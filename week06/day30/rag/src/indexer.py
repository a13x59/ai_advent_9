"""Сохранение и загрузка индекса: FAISS (векторы) + SQLite (текст/метаданные).

Для каждой стратегии создаются:
  indexes/<strategy>.faiss — векторный индекс (IndexFlatIP, cosine)
  indexes/<strategy>.db    — SQLite: chunk_id, source, title, section,
                             strategy, char_start, char_end, vector_row, text
  indexes/<strategy>.json  — манифест со статистикой
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from .models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    source     TEXT,
    title      TEXT,
    section    TEXT,
    strategy   TEXT,
    char_start INTEGER,
    char_end   INTEGER,
    vector_row INTEGER,
    text       TEXT
);
"""


def _stats(chunks: list[Chunk]) -> dict:
    lengths = np.array([len(c.text) for c in chunks], dtype=float)
    return {
        "n_chunks": int(len(chunks)),
        "total_chars": int(lengths.sum()),
        "avg_len": float(lengths.mean()) if len(lengths) else 0.0,
        "std_len": float(lengths.std()) if len(lengths) else 0.0,
        "min_len": int(lengths.min()) if len(lengths) else 0,
        "max_len": int(lengths.max()) if len(lengths) else 0,
    }


def build_index(chunks: list[Chunk], embeddings: np.ndarray, strategy: str,
                out_dir: str | Path, model_name: str, dim: int) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(out / f"{strategy}.faiss"))

    db_path = out / f"{strategy}.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("DELETE FROM chunks")
    for i, c in enumerate(chunks):
        cur.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                c.chunk_id, c.source, c.title, c.section, c.strategy,
                c.char_start, c.char_end, i, c.text,
            ),
        )
    conn.commit()
    conn.close()

    stats = _stats(chunks)
    manifest = {
        "strategy": strategy,
        "model": model_name,
        "dim": dim,
        "faiss_file": f"{strategy}.faiss",
        "db_file": f"{strategy}.db",
        "faiss_size_bytes": (out / f"{strategy}.faiss").stat().st_size,
        "db_size_bytes": db_path.stat().st_size,
        **stats,
    }
    (out / f"{strategy}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_index(strategy: str, out_dir: str | Path) -> tuple:
    """Возвращает (faiss_index, список чанков в порядке строк индекса, manifest)."""
    out = Path(out_dir)
    index = faiss.read_index(str(out / f"{strategy}.faiss"))
    manifest = json.loads((out / f"{strategy}.json").read_text(encoding="utf-8"))

    conn = sqlite3.connect(out / f"{strategy}.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, source, title, section, strategy, char_start, "
        "char_end, text, vector_row FROM chunks ORDER BY vector_row"
    ).fetchall()
    conn.close()
    records = [dict(r) for r in rows]
    return index, records, manifest
