"""Генерация эмбеддингов через sentence-transformers.

Модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-мерные векторы, поддержка русского языка, локально, без API-ключа).

Кэш модели кладётся внутрь проекта (models_cache/), чтобы не зависеть
от домашнего каталога.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "models_cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(CACHE_DIR))

    def embed(self, texts: list[str], batch_size: int = 32,
              show_progress: bool = True) -> np.ndarray:
        """Возвращает нормализованные (unit) векторы для cosine-поиска."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
