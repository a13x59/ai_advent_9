# local_provider.py
"""Локальный провайдер модели: Ollama (llama3.1) по OpenAI-совместимому API.

Никаких облаков: все вызовы идут в локальный сервер Ollama
(OLLAMA_BASE_URL, по умолчанию http://127.0.0.1:11434). Формат запросов и
ответов идентичен DeepSeek (/v1/chat/completions), поэтому провайдер
взаимозаменяем с DeepSeekProvider из main.py и встраивается в тот же
create_app(...) без изменений пайплайна.

Внимание: для инференса нужен именно порт сервера Ollama (11434), а не порт
веб-интерфейса Ollama Desktop (50706) — последний отдаёт только SPA и не
принимает POST /v1/chat/completions.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# Единый источник конфигурации — корневой .env, не зависит от рабочей папки.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from agent_core import (
    AgentProvider,
    AgentRequest,
    parse_json_object,
    normalize_suggestion_list,
    _call_tokens,
    _find_invariant_by_name,
    SUGGEST_SYSTEM,
    SUGGEST_PROMPT,
    SUGGEST_TEMPERATURE,
    SUGGEST_MAX_TOKENS,
    INVARIANT_CHECK_SYSTEM,
    INVARIANT_CHECK_PROMPT,
    INVARIANT_CHECK_TEMPERATURE,
    INVARIANT_CHECK_MAX_TOKENS,
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
# Дефолт — компактная модель под CPU-only VPS (2 ГБ RAM). Локально можно
# переопределить через OLLAMA_MODEL в .env (например, llama3.1).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
# Контекстное окно, передаваемое в Ollama (options.num_ctx). Для 2 ГБ RAM
# держим небольшим (2048), иначе KV-кэш съест память.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
# Ограничение конкурентности вызовов модели (semaphore) и таймаут ожидания слота.
OLLAMA_MAX_CONCURRENCY = int(os.environ.get("OLLAMA_MAX_CONCURRENCY", "2"))
OLLAMA_CONCURRENCY_TIMEOUT = float(os.environ.get("OLLAMA_CONCURRENCY_TIMEOUT", "600"))
# Ретраи на transient-ошибки (модель грузится / 5xx) с линейным backoff.
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_RETRY_BACKOFF = float(os.environ.get("OLLAMA_RETRY_BACKOFF", "2.0"))

# Один семафор на процесс: фактически ограничивает, сколько одновременных
# запросов уходит в Ollama (остальные ждут в очереди на стороне агента).
_MODEL_SEMAPHORE = threading.BoundedSemaphore(OLLAMA_MAX_CONCURRENCY)


def _post_with_retry(url: str, payload: dict, headers: dict, timeout: float) -> requests.Response:
    """POST с ретраями на сетевые ошибки и 5xx (Ollama может перезагружать модель)."""
    last_exc = None
    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            if attempt < OLLAMA_MAX_RETRIES - 1:
                time.sleep(OLLAMA_RETRY_BACKOFF * (attempt + 1))
                continue
            raise
        if resp.status_code in (500, 502, 503) and attempt < OLLAMA_MAX_RETRIES - 1:
            time.sleep(OLLAMA_RETRY_BACKOFF * (attempt + 1))
            continue
        return resp
    raise last_exc

# Имена облачных моделей, которые фронт может прислать по привычке: их нет в
# Ollama, поэтому подставляем локальную модель.
_CLOUD_MODEL_PREFIXES = ("deepseek", "gpt", "claude")


def _resolve_model(requested: str) -> str:
    """Маппинг имени модели, пришедшего от фронта, на модель Ollama."""
    requested = (requested or "").strip()
    if not requested or requested.lower().startswith(_CLOUD_MODEL_PREFIXES):
        return OLLAMA_MODEL
    return requested


def _native_to_compat(data: dict) -> dict:
    """Нативный /api/chat → DeepSeek-совместимый dict (choices + usage)."""
    msg = data.get("message") or {}
    return {
        "choices": [{
            "message": {"role": "assistant", "content": msg.get("content") or ""},
        }],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        },
    }


def _call_openai_compat(messages, model, temperature, top_p, max_tokens, stop, tools):
    """Вызов OpenAI-совместимого /v1/chat/completions (нужен для function calling)."""
    payload = {
        "model": _resolve_model(model),
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop is not None:
        payload["stop"] = stop
    if tools:
        payload["tools"] = tools

    response = _post_with_retry(
        OLLAMA_BASE_URL + "/v1/chat/completions",
        payload,
        {"Content-Type": "application/json"},
        OLLAMA_TIMEOUT,
    )
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    return response.json()


def call_ollama_raw(messages, model, temperature=1.0, top_p=1.0, top_k=0,
                    num_ctx=None, max_tokens=4096, stop=None, tools=None):
    """Один вызов Ollama. Возвращает DeepSeek-совместимый dict.

    Без инструментов используется НАТИВНЫЙ /api/chat: он принимает полный набор
    options, включая num_ctx и top_k (OpenAI-совместимый эндпоинт их игнорирует).
    С инструментами — OpenAI-совместимый /v1/chat/completions, где работает
    нативный function calling (num_ctx/top_k там не передаются — ограничение Ollama).
    """
    # Ограничиваем конкурентность: не больше OLLAMA_MAX_CONCURRENCY одновременных
    # вызовов модели. Остальные запросы ждут слота (или падают по таймауту).
    if not _MODEL_SEMAPHORE.acquire(timeout=OLLAMA_CONCURRENCY_TIMEOUT):
        raise Exception(
            f"Ollama занят: превышен лимит конкурентности "
            f"OLLAMA_MAX_CONCURRENCY={OLLAMA_MAX_CONCURRENCY}. Повторите позже."
        )
    try:
        if tools:
            return _call_openai_compat(messages, model, temperature, top_p, max_tokens, stop, tools)

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if stop is not None:
            options["stop"] = [stop] if isinstance(stop, str) else stop

        payload = {
            "model": _resolve_model(model),
            "messages": messages,
            "stream": False,
            "options": options,
        }

        response = _post_with_retry(
            OLLAMA_BASE_URL + "/api/chat",
            payload,
            {"Content-Type": "application/json"},
            OLLAMA_TIMEOUT,
        )
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        return _native_to_compat(response.json())
    finally:
        _MODEL_SEMAPHORE.release()


class LocalOllamaProvider(AgentProvider):
    """Провайдер: все обращения к модели — в локальный сервер Ollama."""

    def complete(self, messages: List[dict], request: AgentRequest,
                 task: Optional[dict], user_text: str,
                 tools: Optional[List[dict]] = None) -> dict:
        return call_ollama_raw(
            messages,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_ctx=request.num_ctx if request.num_ctx is not None else OLLAMA_NUM_CTX,
            max_tokens=request.max_tokens,
            stop=request.stop,
            tools=tools,
        )

    def suggest_memory(self, working, long_term, user_message, model):
        prompt_text = (
            SUGGEST_PROMPT
            + "\n\nТекущая рабочая память (JSON):\n"
            + json.dumps(working, ensure_ascii=False)
            + "\n\nДолговременная память (JSON):\n"
            + json.dumps(long_term, ensure_ascii=False)
            + "\n\nСообщение пользователя:\n"
            + (user_message or "")
        )
        msgs = [
            {"role": "system", "content": SUGGEST_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        data = call_ollama_raw(
            msgs, model=model, temperature=SUGGEST_TEMPERATURE, max_tokens=SUGGEST_MAX_TOKENS
        )
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = parse_json_object(content) or {}
        result = {
            "working": normalize_suggestion_list(parsed.get("working"), "working"),
            "long_term": normalize_suggestion_list(parsed.get("long_term"), "long_term"),
        }
        inp, out = _call_tokens(prompt_text, content, data.get("usage"))
        return result, inp, out

    def check_invariants(self, invariants, user_text, model) -> dict:
        invariants_text = "\n".join(
            f"- {inv.get('name')}: {inv.get('statement')}"
            + (f" (почему нельзя: {inv.get('rationale')})" if (inv.get('rationale') or "").strip() else "")
            for inv in invariants
        )
        prompt_text = (
            INVARIANT_CHECK_PROMPT
            + "Инварианты:\n" + invariants_text
            + "\n\nЗапрос пользователя:\n" + (user_text or "")
        )
        msgs = [
            {"role": "system", "content": INVARIANT_CHECK_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        data = call_ollama_raw(
            msgs, model=model,
            temperature=INVARIANT_CHECK_TEMPERATURE,
            max_tokens=INVARIANT_CHECK_MAX_TOKENS,
        )
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = parse_json_object(content) or {}
        if not parsed.get("violation"):
            return {"violation": False, "invariant_name": None, "reason": ""}
        name = str(parsed.get("invariant_name") or "").strip()
        return {
            "violation": True,
            "invariant_name": name,
            "invariant": _find_invariant_by_name(invariants, name),
            "reason": str(parsed.get("reason") or "").strip(),
        }
