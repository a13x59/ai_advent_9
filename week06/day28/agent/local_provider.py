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

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") # is port 11434 stable?
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))

# Имена облачных моделей, которые фронт может прислать по привычке: их нет в
# Ollama, поэтому подставляем локальную модель.
_CLOUD_MODEL_PREFIXES = ("deepseek", "gpt", "claude")


def _resolve_model(requested: str) -> str:
    """Маппинг имени модели, пришедшего от фронта, на модель Ollama."""
    requested = (requested or "").strip()
    if not requested or requested.lower().startswith(_CLOUD_MODEL_PREFIXES):
        return OLLAMA_MODEL
    return requested


def call_ollama_raw(messages, model, temperature=1.0, top_p=1.0, max_tokens=4096,
                    stop=None, tools=None):
    """Один вызов Ollama /v1/chat/completions. Возвращает DeepSeek-совместимый dict."""
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

    response = requests.post(
        OLLAMA_BASE_URL + "/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=OLLAMA_TIMEOUT,
    )
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    return response.json()


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
