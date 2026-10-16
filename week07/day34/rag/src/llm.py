"""LLM-провайдеры: DeepSeek (облако) и Ollama (локально).

Единая точка вызова модели для генерации ответов, query rewrite и judge.

Выбор бэкенда генерации — переменная окружения RAG_LLM_PROVIDER:
    deepseek | ollama
По умолчанию `deepseek` (обратная совместимость). Локальный провайдер ходит в
Ollama по OpenAI-совместимому /v1/chat/completions (OLLAMA_BASE_URL,
OLLAMA_MODEL, OLLAMA_TIMEOUT) — тот же протокол, что и DeepSeek, поэтому
провайдеры взаимозаменяемы.

Судья качества использует ОТДЕЛЬНЫЙ бэкенд JUDGE_LLM_PROVIDER (по умолчанию
`deepseek`): «независимый арбитр» не должен зависеть от сравниваемого провайдера.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Единый источник ключа/настроек — корневой .env, не зависит от рабочей папки.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = float(os.environ.get("DEEPSEEK_TIMEOUT", "120"))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))

# Бэкенд генерации/rewrite (deepseek | ollama) и бэкенд судьи (по умолчанию deepseek).
LLM_PROVIDER = os.environ.get("RAG_LLM_PROVIDER", "deepseek").strip().lower()
JUDGE_PROVIDER = os.environ.get("JUDGE_LLM_PROVIDER", "deepseek").strip().lower()

PROVIDERS = ("deepseek", "ollama")


class LLMError(RuntimeError):
    """Ошибка вызова LLM-провайдера (нет ключа, сеть, таймаут, HTTP-статус)."""


def resolve_provider(name: str | None) -> str:
    """Нормализует имя провайдера. None → бэкенд по умолчанию (LLM_PROVIDER)."""
    name = (name or LLM_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise LLMError(f"Неизвестный LLM-провайдер: {name!r} (ожидается один из {PROVIDERS})")
    return name


def _call_deepseek(messages, temperature: float, max_tokens: int,
                   model: str | None, timeout: float) -> dict:
    if not DEEPSEEK_API_KEY:
        raise LLMError("Не задан DEEPSEEK_API_KEY")
    model = model or DEEPSEEK_MODEL
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        DEEPSEEK_API_URL, json=payload, headers=headers,
        timeout=timeout or DEEPSEEK_TIMEOUT,
    )
    if resp.status_code != 200:
        raise LLMError(f"DeepSeek API error {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return {"content": content, "usage": data.get("usage", {}), "model": model}


def _call_ollama(messages, temperature: float, max_tokens: int,
                 model: str | None, timeout: float) -> dict:
    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(
        OLLAMA_BASE_URL + "/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout or OLLAMA_TIMEOUT,
    )
    if resp.status_code != 200:
        raise LLMError(f"Ollama API error {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return {"content": content, "usage": data.get("usage", {}), "model": model}


def call_llm_rich(messages, temperature: float = 0.0, max_tokens: int = 1024,
                  provider: str | None = None, model: str | None = None,
                  timeout: float | None = None) -> dict:
    """Один вызов модели с полными метаданными.

    Возвращает dict: content, provider, model, usage, duration_ms, tokens_per_sec.
    tokens_per_sec считается клиентски (completion_tokens / замеренная длительность):
    OpenAI-совместимый эндпоинт Ollama не отдаёт eval_duration.
    """
    provider = resolve_provider(provider)
    t0 = time.perf_counter()
    if provider == "ollama":
        result = _call_ollama(messages, temperature, max_tokens, model, timeout)
    else:
        result = _call_deepseek(messages, temperature, max_tokens, model, timeout)

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    result["provider"] = provider
    result["duration_ms"] = duration_ms

    completion_tokens = (result.get("usage") or {}).get("completion_tokens")
    if completion_tokens and duration_ms > 0:
        result["tokens_per_sec"] = round(completion_tokens / (duration_ms / 1000.0), 1)
    else:
        result["tokens_per_sec"] = None
    return result


def call_llm(messages, temperature: float = 0.0, max_tokens: int = 1024,
             provider: str | None = None, model: str | None = None,
             timeout: float | None = None) -> str:
    """Совместимая обёртка: возвращает только текст ответа."""
    return call_llm_rich(messages, temperature=temperature, max_tokens=max_tokens,
                         provider=provider, model=model, timeout=timeout)["content"]
