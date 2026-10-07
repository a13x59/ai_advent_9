"""Ответ на вопрос: без RAG и с RAG.

Реализует цепочку из задания:
    вопрос → поиск релевантных чанков → объединение с вопросом → запрос к LLM.

answer_plain — прямой запрос к модели без контекста базы;
answer_rag    — ретрив чанков + промпт с контекстом + запрос к модели.

Модель выбирается через провайдер из src/llm.py: RAG_LLM_PROVIDER=deepseek
(облако) или ollama (локально). Обе функции принимают параметр `provider` для
явного выбора бэкенда и возвращают dict с полем `answer` + метаданными вызова
(provider, model, usage, duration_ms, tokens_per_sec) — это нужно для сравнения
локальной и облачной генерации (День 28).
"""
from __future__ import annotations

from .llm import call_llm_rich
from .retrieval import retrieve_with_mode

PLAIN_SYSTEM = "Ты — ассистент, отвечаешь кратко и по существу."
RAG_SYSTEM = (
    "Ты — ассистент, отвечающий строго на основе предоставленного контекста из "
    "базы знаний. Следуй обязательному формату ответа (секции «Ответ», «Источники», "
    "«Цитаты»). Каждое утверждение ответа помечай ссылкой [n] и подтверждай дословной "
    "цитатой; источники указывай с source, section и chunk_id. Без дословной цитаты "
    "утверждение писать нельзя."
)
REWRITE_SYSTEM = (
    "Ты — помощник поискового движка. Переформулируй вопрос пользователя в "
    "самодостаточный поисковый запрос на русском языке: раскрой местоимения и "
    "неоднозначности, добавь ключевые термины и синонимы, сохрани исходный смысл. "
    "Верни ТОЛЬКО итоговый запрос, без пояснений и кавычек."
)


def _gen_meta(res: dict) -> dict:
    """Достаёт из rich-результата вызова модели поля для ответа."""
    return {
        "provider": res.get("provider"),
        "model": res.get("model"),
        "usage": res.get("usage", {}),
        "duration_ms": res.get("duration_ms"),
        "tokens_per_sec": res.get("tokens_per_sec"),
    }


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


def answer_plain(question: str, temperature: float = 0.0, max_tokens: int = 1024,
                 provider: str | None = None) -> dict:
    """Ответ модели БЕЗ RAG — только вопрос, без контекста базы."""
    messages = [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": question},
    ]
    res = call_llm_rich(messages, temperature=temperature, max_tokens=max_tokens,
                        provider=provider)
    return {"answer": res["content"], **_gen_meta(res)}


def rewrite_query(question: str, temperature: float = 0.0, max_tokens: int = 256,
                  provider: str | None = None) -> str:
    """LLM-переформулировка вопроса в поисковый запрос (этап до ретрива)."""
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": question},
    ]
    res = call_llm_rich(messages, temperature=temperature, max_tokens=max_tokens,
                        provider=provider)
    return res["content"].strip()


def answer_rag(question: str, strategy: str = "structural", top_k: int = 5,
               embedder=None, mode: str = "baseline", top_k_candidates: int | None = None,
               min_score: float | None = None, temperature: float = 0.0,
               max_tokens: int = 1024, provider: str | None = None) -> dict:
    """Ответ модели С RAG: (rewrite?) → ретрив → (filter/rerank?) → промпт → LLM.

    mode ∈ {baseline, rewrite, filter, rewrite+filter, rerank}.
    provider ∈ {deepseek, ollama, None(по умолчанию)} — выбор генератора.
    Возвращает dict с answer, chunks, трейсом второго этапа и метаданными вызова
    (provider, model, usage, duration_ms, tokens_per_sec).
    """
    query = question
    rewritten = None
    if mode in ("rewrite", "rewrite+filter"):
        rewritten = rewrite_query(question, provider=provider)
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
            "provider": None,
            "model": None,
            "usage": {},
            "duration_ms": 0.0,
            "tokens_per_sec": None,
        }

    # В промпт всегда идёт ИСХОДНЫЙ вопрос пользователя; переформулировка
    # используется только для поиска.
    prompt = build_rag_prompt(question, chunks)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    res = call_llm_rich(messages, temperature=temperature, max_tokens=max_tokens,
                        provider=provider)
    return {
        "answer": res["content"],
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
        **_gen_meta(res),
    }
