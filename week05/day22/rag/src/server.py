"""HTTP-сервис ретрива поверх готового индекса.

Агент (agent/rag_client.py) ходит сюда за релевантными чанками, поэтому агент
не тянет в свой venv тяжёлые зависимости (faiss / sentence-transformers).

Запуск (из папки rag, в rag/.venv):
    uvicorn src.server:app --host 0.0.0.0 --port 8891
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import Embedder
from .retrieval import retrieve

app = FastAPI(title="RAG Retrieval Service")

_embedder: Embedder | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Вопрос/текст для поиска чанков")
    strategy: str = Field("structural", description="fixed | structural")
    top_k: int = Field(5, ge=1, le=50)


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
    chunks = retrieve(req.query, strategy=req.strategy, top_k=req.top_k,
                      embedder=_get_embedder())
    return {
        "query": req.query,
        "strategy": req.strategy,
        "top_k": req.top_k,
        "chunks": chunks,
    }
