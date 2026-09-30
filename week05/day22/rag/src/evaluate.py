"""Оценка качества поиска: Recall@k, MRR@k, nDCG@k.

Тест-сет: список запросов на русском с указанием релевантных источников
(подстроки из chunk.source / chunk.title). Релевантность задаётся на уровне
документа, поэтому она одинакова для обеих стратегий chunking — сравнение
стратегий получается честным.
"""
from __future__ import annotations

import math

import numpy as np

from .embedder import Embedder

QUERIES: list[dict] = [
    {"query": "Что такое инвертированный индекс и зачем он нужен?",
     "relevant": ["инвертированный_индекс", "информационный_поиск"]},
    {"query": "Как работает модель word2vec?",
     "relevant": ["word2vec"]},
    {"query": "Что такое косинусное сходство векторов?",
     "relevant": ["косинусное_сходство"]},
    {"query": "Что такое генерация, дополненная поиском (RAG)?",
     "relevant": ["генерация_дополненная_поиском"]},
    {"query": "Как установить и использовать библиотеку razdel для токенизации?",
     "relevant": ["razdel_README"]},
    {"query": "Что такое лемматизация в обработке естественного языка?",
     "relevant": ["лемматизация"]},
    {"query": "Как устроена архитектура трансформера?",
     "relevant": ["трансформер_машинное_обучение"]},
    {"query": "Что такое семантический поиск?",
     "relevant": ["семантический_поиск"]},
    {"query": "Что такое n-граммы и как они используются?",
     "relevant": ["n_грамма"]},
    {"query": "Что такое модель «мешок слов»?",
     "relevant": ["мешок_слов"]},
    {"query": "Что такое векторное представление слова?",
     "relevant": ["векторное_представление_слов"]},
    {"query": "Какие возможности есть у библиотеки natasha?",
     "relevant": ["natasha_README"]},
    {"query": "Чем отличается адресный и семантический поиск?",
     "relevant": ["информационный_поиск"]},
    {"query": "Что такое токенизация текста?",
     "relevant": ["токенизация", "razdel_README"]},
]


def _relevant_chunk_ids(records: list[dict], keywords: list[str]) -> set[str]:
    kws = [k.lower() for k in keywords]
    ids = set()
    for r in records:
        hay = f"{r['source']} {r['title']}".lower()
        if any(k in hay for k in kws):
            ids.add(r["chunk_id"])
    return ids


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def _mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, cid in enumerate(retrieved[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if cid in relevant else 0.0 for cid in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    m = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(m))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_strategy(index, records: list[dict], embedder: Embedder,
                      queries: list[dict], top_k: int = 10) -> dict:
    """Прогоняет тест-сет по одному индексу и возвращает метрики."""
    query_texts = [q["query"] for q in queries]
    q_emb = embedder.embed(query_texts, batch_size=8, show_progress=False)
    q_emb = np.asarray(q_emb, dtype="float32")

    per_query = []
    for q, qe in zip(queries, q_emb):
        relevant = _relevant_chunk_ids(records, q["relevant"])
        if not relevant:
            per_query.append({"query": q["query"], "relevant": 0,
                              "recall@5": None, "mrr@10": None, "ndcg@10": None,
                              "warn": "нет релевантных чанков"})
            continue
        _d, _i = index.search(qe.reshape(1, -1), top_k)
        retrieved = [records[row]["chunk_id"] for row in _i[0]]
        per_query.append({
            "query": q["query"],
            "relevant": len(relevant),
            "recall@5": _recall_at_k(retrieved, relevant, 5),
            "mrr@10": _mrr_at_k(retrieved, relevant, 10),
            "ndcg@10": _ndcg_at_k(retrieved, relevant, 10),
        })

    scored = [p for p in per_query if p["recall@5"] is not None]
    avg = {
        "recall@5": float(np.mean([p["recall@5"] for p in scored])) if scored else 0.0,
        "mrr@10": float(np.mean([p["mrr@10"] for p in scored])) if scored else 0.0,
        "ndcg@10": float(np.mean([p["ndcg@10"] for p in scored])) if scored else 0.0,
    }
    return {
        "n_queries": len(queries),
        "n_scored": len(scored),
        "avg": avg,
        "per_query": per_query,
    }
