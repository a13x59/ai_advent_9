"""Векторный RAG для ревью: HTTP-клиент к rag-сервису (корпуса docs/codebase).

Это «уровень 4» контекста — семантический поиск по отдельному индексу кода
(indexes/codebase) и по документации (indexes). Требует поднятого rag-сервиса
(rag/src/server.py). Если сервис недоступен, retrieve() возвращает пустой
список, и ревью продолжает работать на детерминированном контексте (context.py).
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger("pr_review.rag")

RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://localhost:8891")
RAG_TIMEOUT = float(os.environ.get("RAG_TIMEOUT", "30"))

CORPUS_CODE = "codebase"
CORPUS_DOCS = "docs"


class RagHttpClient:
    """Минимальный HTTP-клиент RAG-ретрива (аналог agent/rag_client.py)."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or RAG_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else RAG_TIMEOUT

    def retrieve(self, query: str, corpus: str = CORPUS_CODE, top_k: int = 5,
                 strategy: str = "structural", mode: str = "baseline") -> list[dict]:
        payload = {
            "query": query, "strategy": strategy, "top_k": top_k,
            "mode": mode, "corpus": corpus,
        }
        try:
            resp = requests.post(self.base_url + "/retrieve", json=payload,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            logger.debug("RAG недоступен: %s", e)
            return []
        if resp.status_code != 200:
            logger.debug("RAG HTTP %s: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json() or {}
        return data.get("chunks") or []


def _query_for_file(path: str, patch: str) -> str:
    """Короткий запрос для поиска похожего кода/документации по файлу."""
    added_lines = [ln[1:].strip() for ln in patch.splitlines()
                   if ln.startswith("+") and not ln.startswith("+++")]
    snippet = " ".join(added_lines)[:200]
    name = path.replace("/", " ").replace("_", " ").replace(".", " ")
    return f"{name} {snippet}".strip()


def retrieve_review_context(changed_files: list[str], diff_text: str,
                            client: RagHttpClient | None = None,
                            top_k: int = 4, max_files: int = 8) -> list[dict]:
    """Семантический контекст: по каждому изменённому файлу ищет похожие чанки
    в корпусе кода (codebase) и в документации (docs), дедуплицирует."""
    client = client or RagHttpClient()
    from .diff_utils import parse_diff  # локальный импорт — нет цикла

    patches = {f.path: f.patch for f in parse_diff(diff_text)}
    seen: set[str] = set()
    merged: list[dict] = []

    for path in changed_files[:max_files]:
        query = _query_for_file(path, patches.get(path, ""))
        if not query:
            continue
        for corpus in (CORPUS_CODE, CORPUS_DOCS):
            for chunk in client.retrieve(query, corpus=corpus, top_k=top_k):
                cid = chunk.get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                merged.append({**chunk, "corpus": corpus})

    merged.sort(key=lambda c: -(float(c.get("score") or 0.0)))
    return merged[: max(1, top_k * 2)]


def render_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["===== RAG-контекст (похожий код и документация) ====="]
    for i, c in enumerate(chunks, 1):
        src = c.get("source") or "?"
        section = c.get("section") or ""
        corpus = c.get("corpus") or ""
        lines.append(f"[{i}] {corpus} :: {src}" + (f" | {section}" if section else ""))
        lines.append((c.get("text") or "").strip()[:1200])
    return "\n\n".join(lines)
