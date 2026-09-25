# main.py
"""Точка входа: реальный агент (DeepSeek API), без моков.

Вся логика обработки запроса живёт в agent_core (единый пайплайн + фабрика
create_app). Здесь определён только РЕАЛЬНЫЙ провайдер модели — DeepSeekProvider,
который ходит по HTTP к api.deepseek.com.

Мок-версия (для тестов без сети и без ключа) лежит в mock_agent.py.
"""
import json
import os
from typing import List, Optional

import requests

from agent_core import (
    AgentProvider,
    AgentRequest,
    create_app,
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
from mcp_client import FortuneMcpClient, CurrencyMcpClient

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-...")


def call_deepseek_raw(messages, model, temperature=1.0, top_p=1.0, max_tokens=4096, top_k=0, stop=None, tools=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop is not None:
        payload["stop"] = stop
    if top_k:
        payload["top_k"] = top_k
    if tools:
        payload["tools"] = tools

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
    return response.json()


class DeepSeekProvider(AgentProvider):
    """Реальный провайдер: все обращения к модели — через DeepSeek API."""

    def complete(self, messages: List[dict], request: AgentRequest,
                 task: Optional[dict], user_text: str,
                 tools: Optional[List[dict]] = None) -> dict:
        return call_deepseek_raw(
            messages,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
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
        data = call_deepseek_raw(
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
        data = call_deepseek_raw(
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


app = create_app(
    DeepSeekProvider(),
    mcp_clients=[FortuneMcpClient(), CurrencyMcpClient()],
)


# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
