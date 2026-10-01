"""Ответ на вопрос: без RAG и с RAG.

Реализует цепочку из задания:
    вопрос → поиск релевантных чанков → объединение с вопросом → запрос к LLM.

answer_plain — прямой запрос к DeepSeek без контекста базы;
answer_rag    — ретрив чанков + промпт с контекстом + запрос к DeepSeek.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from .retrieval import retrieve

# Единый источник ключа — корневой .env (day22/.env), не зависит от рабочей папки.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")

PLAIN_SYSTEM = "Ты — ассистент, отвечаешь кратко и по существу."
RAG_SYSTEM = (
    "Ты — ассистент, отвечающий на основе предоставленного контекста из базы знаний. "
    "Отвечай строго по контексту."
)


def call_llm(messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
    """Один вызов DeepSeek chat/completions, возвращает текст ответа."""
    if not API_KEY:
        raise RuntimeError("Не задан DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Объединяет вопрос и найденные чанки в один промпт для LLM."""
    if not chunks:
        context = "(релевантные фрагменты не найдены)"
    else:
        blocks = []
        for i, c in enumerate(chunks, 1):
            title = c.get("title") or ""
            section = c.get("section") or ""
            src = c.get("source") or ""
            header = f"[{i}] {title}" + (f" — {section}" if section else "")
            blocks.append(f"{header}\nИсточник: {src}\n{c.get('text', '')}")
        context = "\n\n".join(blocks)

    return (
        "Ответь на вопрос пользователя, используя ТОЛЬКО приведённый ниже контекст "
        "(фрагменты из базы знаний).\n"
        "Правила:\n"
        "- Отвечай строго по контексту; не добавляй сведений, которых в нём нет.\n"
        "- В конце перечисли использованные источники в виде [n] (по номерам фрагментов).\n"
        "- Если в контексте нет ответа на вопрос — так и скажи.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}"
    )


def answer_plain(question: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Ответ модели БЕЗ RAG — только вопрос, без контекста базы."""
    messages = [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": question},
    ]
    return call_llm(messages, temperature=temperature, max_tokens=max_tokens)


def answer_rag(question: str, strategy: str = "structural", top_k: int = 5,
               embedder=None, temperature: float = 0.0, max_tokens: int = 1024) -> dict:
    """Ответ модели С RAG: ретрив → промпт с контекстом → LLM.

    Возвращает dict с answer и найденными chunks (для трейса и отчёта).
    """
    chunks = retrieve(question, strategy=strategy, top_k=top_k, embedder=embedder)
    prompt = build_rag_prompt(question, chunks)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages, temperature=temperature, max_tokens=max_tokens)
    return {"answer": answer, "chunks": chunks, "strategy": strategy, "top_k": top_k}
