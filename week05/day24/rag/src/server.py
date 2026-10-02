"""HTTP-сервис ретрива поверх готового индекса.

Агент (agent/rag_client.py) ходит сюда за релевантными чанками, поэтому агент
не тянет в свой venv тяжёлые зависимости (faiss / sentence-transformers).

Сервис реализует все режимы ретрива (День 23):
    baseline / rewrite / filter / rewrite+filter / rerank
Параметры: strategy, top_k, top_k_candidates (до фильтра), min_score (порог),
mode. Режимы «rewrite*» переформулируют запрос через DeepSeek (требует
DEEPSEEK_API_KEY из корневого .env).

Запуск (из папки rag, в rag/.venv):
    uvicorn src.server:app --host 0.0.0.0 --port 8891
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .embedder import Embedder
from .qa import rewrite_query
from .retrieval import retrieve_with_mode

logger = logging.getLogger("rag.server")
app = FastAPI(title="RAG Retrieval Service")

_embedder: Embedder | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Вопрос/текст для поиска чанков")
    strategy: str = Field("structural", description="fixed | structural")
    top_k: int = Field(5, ge=1, le=50)
    top_k_candidates: Optional[int] = Field(None, ge=1, le=200,
                                            description="Сколько кандидатов брать ДО фильтра/реранка")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0,
                                       description="Порог отсечения (filter-режим)")
    mode: str = Field("baseline", description="baseline|rewrite|filter|rewrite+filter|rerank")


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@app.get("/health")
def health():
    return {"ok": True, "model": _embedder.model_name if _embedder else None}


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    if req.strategy not in ("fixed", "structural"):
        raise HTTPException(status_code=400, detail="strategy должен быть fixed|structural")
    if req.mode not in config.MODES:
        raise HTTPException(status_code=400, detail=f"mode должен быть один из {config.MODES}")

    query = req.query
    rewritten = None
    if req.mode in ("rewrite", "rewrite+filter"):
        try:
            rewritten = rewrite_query(req.query)
            query = rewritten
        except Exception as e:  # noqa: BLE001 — без LLM переформулировка недоступна
            logger.warning("Query rewrite недоступен (%s), использую исходный запрос", e)
            query = req.query

    result = retrieve_with_mode(
        query, strategy=req.strategy, top_k=req.top_k,
        top_k_candidates=req.top_k_candidates, min_score=req.min_score,
        mode=req.mode, embedder=_get_embedder(),
    )
    result["original_query"] = req.query
    result["rewritten_query"] = rewritten
    return result
