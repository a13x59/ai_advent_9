"""LLM-провайдеры для ревью: DeepSeek (облако) и Ollama (локально).

Переключатель провайдера — REVIEW_PROVIDER / --provider (deepseek|ollama),
как договорились: по флагу делегируем ревью локальной модели или DeepSeek.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger("pr_review.providers")

ROOT = Path(__file__).resolve().parent.parent
try:  # python-dotenv опционален (в CI env задаётся секретами/переменными)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))


class ProviderError(RuntimeError):
    """Ошибка вызова LLM-провайдера."""


def _extract_content(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"Непонятный ответ модели: {str(data)[:300]}")


def complete_deepseek(messages: list[dict], model: str = DEEPSEEK_MODEL,
                      temperature: float = 0.2, max_tokens: int = 2048) -> str:
    if not DEEPSEEK_API_KEY:
        raise ProviderError("DEEPSEEK_API_KEY не задан")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=120,
        )
    except requests.RequestException as e:
        raise ProviderError(f"DeepSeek недоступен: {e}") from e
    if resp.status_code != 200:
        raise ProviderError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    usage = data.get("usage") or {}
    logger.info("deepseek usage: %s", usage)
    return _extract_content(data)


def complete_ollama(messages: list[dict], model: str = OLLAMA_MODEL,
                    temperature: float = 0.2, max_tokens: int = 2048) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(
            OLLAMA_BASE_URL.rstrip("/") + "/v1/chat/completions",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ProviderError(f"Ollama недоступен: {e}") from e
    if resp.status_code != 200:
        raise ProviderError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    usage = data.get("usage") or {}
    logger.info("ollama usage: %s", usage)
    return _extract_content(data)


def complete(provider: str, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Единая точка вызова по имени провайдера."""
    provider = (provider or "deepseek").lower()
    if provider == "deepseek":
        return complete_deepseek(messages, model=model or DEEPSEEK_MODEL,
                                 temperature=temperature, max_tokens=max_tokens)
    if provider in ("ollama", "local", "llama"):
        return complete_ollama(messages, model=model or OLLAMA_MODEL,
                               temperature=temperature, max_tokens=max_tokens)
    raise ProviderError(f"Неизвестный провайдер: {provider} (deepseek|ollama)")
