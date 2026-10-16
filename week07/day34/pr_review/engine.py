"""Движок AI-ревью (lean, без FastAPI).

Оркестрирует: diff → контекст (детерминированный + RAG) → промпт → LLM →
структурированный результат. Используется headless-скриптом review_pr.py
(подход B) и переиспользуется эндпоинтом /review агента (подход A) через
инъекцию `completion` (так обе ветки получают одинаковый промпт/контекст и
сопоставимы).

`completion` — callable(messages: list[dict]) -> str. По умолчанию — providers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import context as ctx
from . import prompt as pm
from . import providers
from . import rag_client as rag
from .diff_utils import changed_files, estimate_tokens

logger = logging.getLogger("pr_review.engine")

MAX_DIFF_CHARS = 60_000
MAX_RAG_CHUNKS = 10


@dataclass
class ReviewResult:
    review_text: str
    sections: dict = field(default_factory=dict)
    changed_files: list = field(default_factory=list)
    rag_chunks: list = field(default_factory=list)
    context_blocks: list = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens_estimate: int = 0
    diff_chars: int = 0


def run_review(diff_text: str, *, provider: str = "deepseek", model: str | None = None,
               use_rag: bool = True, rag_base_url: str | None = None,
               temperature: float = 0.2, max_tokens: int = 2048,
               completion=None) -> ReviewResult:
    """Выполняет ревью diff и возвращает ReviewResult."""
    if not (diff_text or "").strip():
        raise ValueError("diff пуст — нечего ревьюить")

    files = changed_files(diff_text)

    # 1. Детерминированный контекст: полные файлы + связанный код + документация.
    context_blocks = ctx.build_deterministic_context(files)
    context_text = ctx.render_context(context_blocks)

    # 2. Векторный RAG (документация + код) — если включён.
    rag_chunks: list[dict] = []
    rag_text = ""
    if use_rag:
        try:
            client = rag.RagHttpClient(base_url=rag_base_url) if rag_base_url else rag.RagHttpClient()
            rag_chunks = rag.retrieve_review_context(files, diff_text, client=client,
                                                     top_k=4)[:MAX_RAG_CHUNKS]
            rag_text = rag.render_rag_context(rag_chunks)
        except Exception as e:  # noqa: BLE001 — RAG опционален
            logger.warning("RAG недоступен, продолжаю без него: %s", e)

    # 3. Промпт (с ограничением размера diff).
    bounded_diff = diff_text
    truncated = False
    if len(diff_text) > MAX_DIFF_CHARS:
        bounded_diff = diff_text[:MAX_DIFF_CHARS] + "\n... [diff усечён]"
        truncated = True
    messages = pm.build_review_messages(bounded_diff, files, context_text, rag_text)

    # 4. Генерация.
    if completion is None:
        completion = lambda msgs: providers.complete(  # noqa: E731
            provider, msgs, model=model, temperature=temperature, max_tokens=max_tokens)
    review_text = completion(messages)

    default_model = (providers.DEEPSEEK_MODEL if provider == "deepseek"
                     else providers.OLLAMA_MODEL)
    result = ReviewResult(
        review_text=review_text,
        sections=pm.split_sections(review_text),
        changed_files=files,
        rag_chunks=rag_chunks,
        context_blocks=context_blocks,
        provider=provider,
        model=model or default_model,
        prompt_tokens_estimate=estimate_tokens(messages[0]["content"]) + estimate_tokens(messages[1]["content"]),
        diff_chars=len(diff_text),
    )
    if truncated:
        result.review_text += "\n\n> ⚠️ Diff был усечён из-за размера; ревью покрывает первые 60k символов."
    return result
