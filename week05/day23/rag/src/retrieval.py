"""Поиск релевантных чанков по вопросу поверх готового индекса.

Использует уже собранные индексы (FAISS + SQLite) и модель эмбеддингов из
embedder.py. Возвращает список чанков с текстом, метаданными и score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .embedder import Embedder
from .indexer import load_index

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "indexes"


def retrieve(query: str, strategy: str = "structural", top_k: int = 5,
             embedder: Embedder | None = None, index_dir: str | Path | None = None) -> list[dict]:
    """Возвращает top_k наиболее релевантных чанков для вопроса.

    embedder можно передать заранее созданный (например, из server.py), чтобы
    не перезагружать модель при каждом вызове. Если None — создаётся новый.
    """
    index_dir = Path(index_dir) if index_dir else INDEX_DIR
    if embedder is None:
        embedder = Embedder()

    index, records, _manifest = load_index(strategy, index_dir)
    emb = embedder.embed([query], batch_size=1, show_progress=False)
    dists, idxs = index.search(np.asarray(emb, dtype="float32"), top_k)

    chunks: list[dict] = []
    for d, row in zip(dists[0], idxs[0]):
        rec = records[int(row)]
        chunks.append({
            "chunk_id": rec["chunk_id"],
            "source": rec["source"],
            "title": rec["title"],
            "section": rec.get("section") or "",
            "score": float(d),
            "text": rec["text"],
        })
    return chunks
