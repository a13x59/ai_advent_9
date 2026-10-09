"""Клиенты: Ollama (нативный /api/chat) и DeepSeek (судья качества).

Используется НАТИВНЫЙ эндпоинт Ollama /api/chat (не OpenAI-совместимый):
он возвращает серверные тайминги (eval_duration, prompt_eval_duration,
load_duration, total_duration) и точные счётчики токенов (eval_count,
prompt_eval_count) — это даёт честную скорость, в отличие от client-side
замера wall-time.

Потребление ресурсов:
  * модель на диске — /api/tags (size, quantization_level);
  * загруженная модель — /api/ps (size, size_vram);
  * RSS процессов Ollama — psutil (суммарно по процессам).
"""
from __future__ import annotations

import json
import time
from typing import Optional

import psutil
import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    OLLAMA_BASE_URL,
    TIMEOUT_SECONDS,
)


class LlmError(RuntimeError):
    """Ошибка вызова LLM-провайдера."""


# --------------------------------------------------------------------------- #
# Ollama: нативный /api/chat
# --------------------------------------------------------------------------- #
def _ollama_options(options: dict) -> dict:
    """Преобразует наши параметры в options Ollama."""
    out = {}
    if options.get("temperature") is not None:
        out["temperature"] = float(options["temperature"])
    if options.get("top_p") is not None:
        out["top_p"] = float(options["top_p"])
    if options.get("top_k") is not None:
        out["top_k"] = int(options["top_k"])
    if options.get("max_tokens") is not None:
        out["num_predict"] = int(options["max_tokens"])
    if options.get("num_ctx") is not None:
        out["num_ctx"] = int(options["num_ctx"])
    if options.get("seed") is not None:
        out["seed"] = int(options["seed"])
    if options.get("repeat_penalty") is not None:
        out["repeat_penalty"] = float(options["repeat_penalty"])
    return out


def call_ollama_chat(messages: list[dict], model: str,
                     options: Optional[dict] = None) -> dict:
    """Один вызов /api/chat. Возвращает dict с content + серверными метриками."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _ollama_options(options or {}),
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            OLLAMA_BASE_URL + "/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise LlmError(f"Ollama request failed: {type(e).__name__}: {e}") from e

    if resp.status_code != 200:
        raise LlmError(f"Ollama API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    content = (data.get("message") or {}).get("content", "") or ""
    eval_count = data.get("eval_count")
    eval_duration_ns = data.get("eval_duration")
    prompt_eval_count = data.get("prompt_eval_count")
    prompt_eval_duration_ns = data.get("prompt_eval_duration")
    load_duration_ns = data.get("load_duration")
    total_duration_ns = data.get("total_duration")

    result = {
        "content": content,
        "model": data.get("model", model),
        "done_reason": data.get("done_reason"),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if eval_count and eval_duration_ns:
        result["tokens_per_sec"] = round(eval_count / (eval_duration_ns / 1e9), 2)
        result["eval_duration_ms"] = round(eval_duration_ns / 1e6, 1)
    else:
        result["tokens_per_sec"] = None
        result["eval_duration_ms"] = None
    if prompt_eval_count and prompt_eval_duration_ns:
        result["prompt_tokens_per_sec"] = round(
            prompt_eval_count / (prompt_eval_duration_ns / 1e9), 2
        )
    else:
        result["prompt_tokens_per_sec"] = None
    result["load_duration_ms"] = (
        round(load_duration_ns / 1e6, 1) if load_duration_ns else None
    )
    result["total_duration_ms"] = (
        round(total_duration_ns / 1e6, 1) if total_duration_ns else None
    )
    return result


# --------------------------------------------------------------------------- #
# Ollama: реестр моделей и потребление ресурсов
# --------------------------------------------------------------------------- #
def ollama_tags() -> list[dict]:
    resp = requests.get(OLLAMA_BASE_URL + "/api/tags", timeout=30)
    resp.raise_for_status()
    return resp.json().get("models", [])


def ollama_ps() -> dict:
    resp = requests.get(OLLAMA_BASE_URL + "/api/ps", timeout=30)
    resp.raise_for_status()
    return resp.json()


def ollama_rss_mb() -> float:
    """Суммарный RSS процессов Ollama (сервер + runner) в МБ."""
    total = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            info = p.info
            name = (info.get("name") or "").lower()
            cmdline = " ".join(info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmdline:
                total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return round(total / (1024 * 1024), 1)


def model_static_meta() -> dict:
    """tag -> {size_mb, quantization_level, parameter_size}."""
    out = {}
    for m in ollama_tags():
        d = m.get("details") or {}
        out[m["name"]] = {
            "size_mb": round((m.get("size") or 0) / (1024 * 1024), 1),
            "quantization_level": d.get("quantization_level"),
            "parameter_size": d.get("parameter_size"),
            "context_length": d.get("context_length"),
        }
    return out


def loaded_model_meta(model: str) -> dict:
    """Метаданные ЗАГРУЖЕННОЙ модели из /api/ps (size, size_vram)."""
    try:
        ps = ollama_ps()
    except requests.RequestException:
        return {}
    for m in ps.get("models", []):
        if (m.get("name") == model) or (m.get("model") == model):
            return {
                "loaded_size_mb": round((m.get("size") or 0) / (1024 * 1024), 1),
                "loaded_size_vram_mb": round((m.get("size_vram") or 0) / (1024 * 1024), 1),
            }
    return {}


def snapshot_resources(model: str) -> dict:
    """Снимок ресурсов: RSS процессов + загруженная модель (после вызова)."""
    return {
        "ollama_rss_mb": ollama_rss_mb(),
        **loaded_model_meta(model),
    }


# --------------------------------------------------------------------------- #
# DeepSeek: судья качества (независимый арбитр)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "Ты — строгий и беспристрастный судья. Оценивай ответ ассистента относительно "
    "ожидания и ТОЛЬКО по заданным критериям. Не добавляй ничего сверх запрошенного."
)

JUDGE_PROMPT = """Оцени ответ ассистента на вопрос.

Вопрос: {question}

Ожидание (что должно быть в ответе): {expectation}

Ответ ассистента:
{answer}

Критерии оценки (score):
- 2 — ответ полностью соответствует ожиданию: названы все ключевые элементы, фактических ошибок и выдумок нет;
- 1 — ответ частично соответствует: есть часть ключевых элементов, но неполно или с неточностями;
- 0 — ответ не соответствует ожиданию: ключевые элементы отсутствуют, неверны или выдуманы.

Верни СТРОГО JSON-объект вида {{"score": <0|1|2>, "rationale": "<краткое обоснование>"}}.
"""


def judge_answer(question: str, expectation: str, answer: str) -> dict:
    """Оценивает ответ 0–2 независимым судьёй (DeepSeek)."""
    if not DEEPSEEK_API_KEY:
        raise LlmError("Не задан DEEPSEEK_API_KEY")
    prompt = JUDGE_PROMPT.format(question=question, expectation=expectation, answer=answer)
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise LlmError(f"DeepSeek API error {resp.status_code}: {resp.text[:300]}")
    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        obj = json.loads(raw)
    except ValueError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
    score = obj.get("score", 0)
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    score = max(0, min(2, int(score)))
    return {"score": score, "rationale": str(obj.get("rationale") or "")}
