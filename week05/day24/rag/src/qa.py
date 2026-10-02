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

from .retrieval import retrieve_with_mode

# Единый источник ключа — корневой .env (day22/.env), не зависит от рабочей папки.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "deepseek-chat")

PLAIN_SYSTEM = "Ты — ассистент, отвечаешь кратко и по существу."
RAG_SYSTEM = (
    "Ты — ассистент, отвечающий строго на основе предоставленного контекста из "
    "базы знаний. Следуй обязательному формату ответа (секции «Ответ», «Источники», "
    "«Цитаты»). Каждое утверждение ответа помечай ссылкой [n] и подтверждай дословной "
    "цитатой; источники указывай с source, section и chunk_id. Без дословной цитаты "
    "утверждение писать нельзя."
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
    """Объединяет вопрос и найденные чанки в один промпт для LLM.

    Каждый чанк получает явные метаданные source / section / chunk_id, а сам
    промпт требует строгий трёхсекционный формат ответа (Ответ / Источники /
    Цитаты) — это позволяет детерминированно проверять наличие источников и
    дословных цитат.
    """
    if not chunks:
        context = "(релевантные фрагменты не найдены)"
    else:
        blocks = []
        for i, c in enumerate(chunks, 1):
            title = c.get("title") or ""
            section = c.get("section") or ""
            src = c.get("source") or ""
            cid = c.get("chunk_id") or ""
            header = f"[{i}] {title}" + (f" — {section}" if section else "")
            blocks.append(
                f"{header}\nИсточник: {src}\nchunk_id: {cid}\n{c.get('text', '')}"
            )
        context = "\n\n".join(blocks)

    return (
        "Ответь на вопрос пользователя, используя ТОЛЬКО приведённый ниже контекст "
        "(фрагменты из базы знаний).\n"
        "\n"
        "ФОРМАТ ОТВЕТА — строго по шаблону (три секции):\n"
        "\n"
        "## Ответ\n"
        "<краткий ответ по контексту; каждое утверждение заканчивай ссылкой [n]>\n"
        "\n"
        "## Источники\n"
        "- [n] source: <путь> | section: <раздел> | chunk_id: <id>\n"
        "(по одной строке на каждый использованный фрагмент)\n"
        "\n"
        "## Цитаты\n"
        "- [n] «<дословный фрагмент из соответствующего чанка>»\n"
        "(по цитате на каждое фактическое утверждение из «Ответ»)\n"
        "\n"
        "Правила:\n"
        "- Отвечай только на основе контекста; не добавляй сведений, которых в нём нет.\n"
        "- Каждое утверждение в «Ответ» обязательно помечай ссылкой [n] на фрагмент контекста.\n"
        "- Любое утверждение без дословной цитаты недопустимо: либо процитируй его, либо убери из ответа.\n"
        "- В «Цитаты» приведи ДОСЛОВНЫЙ фрагмент для каждого утверждения из «Ответ» (не пересказ).\n"
        "- Номер [n] должен совпадать с номером фрагмента в контексте.\n"
        "- В «Источники» указывай source, section и chunk_id именно из заголовка фрагмента.\n"
        "- Если в контексте нет ответа на вопрос — так и скажи и попроси уточнить вопрос.\n"
        "\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}"
    )


def build_abstain_answer(question: str, max_score: float | None = None) -> str:
    """Детерминированный ответ «не знаю» при слабом контексте (без вызова LLM)."""
    return (
        "## Ответ\n"
        "Я не знаю ответа на этот вопрос — в базе знаний не нашлось достаточно "
        "релевантной информации.\n\n"
        "Пожалуйста, уточните вопрос или переформулируйте его."
    )


def answer_plain(question: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Ответ модели БЕЗ RAG — только вопрос, без контекста базы."""
    messages = [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": question},
    ]
    return call_llm(messages, temperature=temperature, max_tokens=max_tokens)


REWRITE_SYSTEM = (
    "Ты — помощник поискового движка. Переформулируй вопрос пользователя в "
    "самодостаточный поисковый запрос на русском языке: раскрой местоимения и "
    "неоднозначности, добавь ключевые термины и синонимы, сохрани исходный смысл. "
    "Верни ТОЛЬКО итоговый запрос, без пояснений и кавычек."
)


def rewrite_query(question: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
    """LLM-переформулировка вопроса в поисковый запрос (этап до ретрива)."""
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": question},
    ]
    return call_llm(messages, temperature=temperature, max_tokens=max_tokens).strip()


def answer_rag(question: str, strategy: str = "structural", top_k: int = 5,
               embedder=None, mode: str = "baseline", top_k_candidates: int | None = None,
               min_score: float | None = None, temperature: float = 0.0,
               max_tokens: int = 1024) -> dict:
    """Ответ модели С RAG: (rewrite?) → ретрив → (filter/rerank?) → промпт → LLM.

    mode ∈ {baseline, rewrite, filter, rewrite+filter, rerank}.
    Возвращает dict с answer, chunks и трейсом второго этапа (для отчёта).
    """
    query = question
    rewritten = None
    if mode in ("rewrite", "rewrite+filter"):
        rewritten = rewrite_query(question)
        query = rewritten

    result = retrieve_with_mode(
        query, strategy=strategy, top_k=top_k, top_k_candidates=top_k_candidates,
        min_score=min_score, mode=mode, embedder=embedder,
    )
    chunks = result["chunks"]
    below_relevance = bool(result.get("below_relevance"))

    # Детерминированный гейт «не знаю»: слабый контекст → отказ без вызова LLM.
    if below_relevance or not chunks:
        return {
            "answer": build_abstain_answer(question, max_score=result.get("max_score")),
            "chunks": chunks,
            "strategy": strategy,
            "top_k": top_k,
            "mode": mode,
            "rewritten_query": rewritten,
            "query_used": query,
            "candidates": result.get("candidates"),
            "kept": result.get("kept"),
            "dropped": result.get("dropped", []),
            "max_score": result.get("max_score"),
            "below_relevance": below_relevance,
            "abstained": True,
        }

    # В промпт всегда идёт ИСХОДНЫЙ вопрос пользователя; переформулировка
    # используется только для поиска.
    prompt = build_rag_prompt(question, chunks)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages, temperature=temperature, max_tokens=max_tokens)
    return {
        "answer": answer,
        "chunks": chunks,
        "strategy": strategy,
        "top_k": top_k,
        "mode": mode,
        "rewritten_query": rewritten,
        "query_used": query,
        "candidates": result.get("candidates"),
        "kept": result.get("kept"),
        "dropped": result.get("dropped", []),
        "max_score": result.get("max_score"),
        "below_relevance": below_relevance,
        "abstained": False,
    }
