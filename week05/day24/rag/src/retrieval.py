"""Поиск релевантных чанков + второй этап (фильтрация / реранкинг).

Одностадийный поиск:
    retrieve() — косинусный поиск по FAISS, возвращает top_k чанков.

Двухстадийный пайплайн (День 23):
    retrieve_with_mode() — берёт top_k_candidates, затем применяет второй
    этап в зависимости от режима:

      * filter  — отсечение по порогу similarity (MIN_SCORE);
      * rerank  — лексическая эвристика (пересечение слов запроса и чанка),
                  fusion с косинусом + MMR-разнообразие (убирает дубли).

    rewrite* применяется вызывающим кодом ДО вызова: запрос уже переформулирован,
    а имя режима определяет только наличие второго этапа.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import config
from .embedder import Embedder
from .indexer import load_index

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "indexes"

# --------------------------------------------------------------------------- #
# Лексическая эвристика (русский, без внешних зависимостей)
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "the", "a", "an", "of", "to", "and", "is", "in", "for", "it", "that",
    "with", "on", "as", "by", "are", "at", "be", "this", "or", "from",
}

_WORD_RE = re.compile(r"[a-zа-яё0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def lexical_score(query: str, text: str) -> float:
    """Доля значимых слов запроса, встретившихся в чанке (token recall, 0..1)."""
    q = set(_tokens(query))
    if not q:
        return 0.0
    c = set(_tokens(text))
    return len(q & c) / len(q)


# --------------------------------------------------------------------------- #
# Одностадийный поиск
# --------------------------------------------------------------------------- #
def _search_raw(query: str, strategy: str, top_k: int, embedder, index_dir):
    index_dir = Path(index_dir) if index_dir else INDEX_DIR
    if embedder is None:
        embedder = Embedder()

    index, records, _manifest = load_index(strategy, index_dir)
    emb = embedder.embed([query], batch_size=1, show_progress=False)
    emb = np.asarray(emb, dtype="float32")
    dists, idxs = index.search(emb, top_k)

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
    return index, records, emb[0], chunks


def retrieve(query: str, strategy: str = "structural", top_k: int = 5,
             embedder: Embedder | None = None,
             index_dir: str | Path | None = None) -> list[dict]:
    """Возвращает top_k наиболее релевантных чанков (одностадийный поиск)."""
    _index, _records, _qvec, chunks = _search_raw(query, strategy, top_k, embedder, index_dir)
    return chunks


# --------------------------------------------------------------------------- #
# Второй этап: фильтр и реранкинг
# --------------------------------------------------------------------------- #
def filter_by_threshold(chunks: list[dict], min_score: float,
                        margin: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Отсечение по порогу cosine similarity.

    Абсолютный порог: score >= min_score. При margin > 0 дополнительно
    сохраняются чанки, близкие к лучшему (score >= max_score - margin) —
    это защищает от заниженного масштаба score на коротких запросах.
    """
    kept, dropped = [], []
    max_score = max((c["score"] for c in chunks), default=0.0)
    for c in chunks:
        if c["score"] >= min_score or (margin > 0 and c["score"] >= max_score - margin):
            kept.append(c)
        else:
            dropped.append({**c, "reason": "below_threshold"})
    return kept, dropped


def _minmax_normalize(values: list[float]) -> list[float]:
    values = [float(v) for v in values]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rerank_candidates(query: str, chunks: list[dict], vectors: np.ndarray,
                      query_vec: np.ndarray, top_k: int,
                      lex_weight: float | None = None,
                      mmr_lambda: float | None = None) -> list[dict]:
    """Лексическая эвристика + fusion с косинусом + MMR-разнообразие.

    vectors  — нормализованные векторы чанков-кандидатов (n, dim);
    query_vec — нормализованный вектор запроса (dim,).
    """
    lex_weight = config.LEX_WEIGHT if lex_weight is None else lex_weight
    mmr_lambda = config.MMR_LAMBDA if mmr_lambda is None else mmr_lambda

    lex = [lexical_score(query, c["text"]) for c in chunks]
    cos_norm = _minmax_normalize([c["score"] for c in chunks])
    fused = [(1 - lex_weight) * cn + lex_weight * lx for cn, lx in zip(cos_norm, lex)]

    # MMR: итеративно выбираем следующий чанк, максимизируя
    # λ·релевантность − (1−λ)·(макс. схожесть с уже выбранными).
    selected: list[int] = []
    remaining = list(range(len(chunks)))
    while remaining and len(selected) < top_k:
        best_i, best_val = None, -np.inf
        for i in remaining:
            sim_sel = max(
                (float(np.dot(vectors[i], vectors[j])) for j in selected),
                default=0.0,
            )
            val = mmr_lambda * fused[i] - (1 - mmr_lambda) * sim_sel
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)

    result: list[dict] = []
    for i in selected:
        c = dict(chunks[i])
        c["cosine_score"] = c["score"]
        c["lexical_score"] = round(lex[i], 4)
        c["score"] = round(fused[i], 4)  # итоговая релевантность после fusion
        result.append(c)
    return result


def retrieve_with_mode(query: str, strategy: str = "structural", top_k: int | None = None,
                       top_k_candidates: int | None = None, min_score: float | None = None,
                       mode: str = "baseline", embedder: Embedder | None = None,
                       index_dir: str | Path | None = None) -> dict:
    """Полный пайплайн ретрива с возможным вторым этапом.

    mode ∈ {baseline, rewrite, filter, rewrite+filter, rerank}.
    «rewrite» должен быть применён вызывающим кодом к `query` заранее;
    здесь rewrite* ведут себя как baseline/filter — второй этап зависит
    только от наличия filter/rerank в имени режима.

    Возвращает словарь: chunks (итоговые), candidates/kept (до/после),
    dropped (отсечённые с причиной), mode и фактический query.
    """
    top_k = config.TOP_K_FINAL if top_k is None else top_k
    top_k_candidates = config.TOP_K_CANDIDATES if top_k_candidates is None else top_k_candidates
    min_score = config.MIN_SCORE if min_score is None else min_score

    if mode in ("filter", "rewrite+filter"):
        post = "filter"
    elif mode == "rerank":
        post = "rerank"
    else:
        post = None

    if post is None:
        _i, _r, _q, chunks = _search_raw(query, strategy, top_k, embedder, index_dir)
        return {
            "mode": mode, "query": query, "chunks": chunks,
            "candidates": len(chunks), "kept": len(chunks), "dropped": [],
        }

    index, _records, qvec, candidates = _search_raw(
        query, strategy, top_k_candidates, embedder, index_dir
    )

    if post == "filter":
        kept, dropped = filter_by_threshold(candidates, min_score, margin=config.MAX_SCORE_MARGIN)
        final = kept[:top_k]
        overflow = [{**c, "reason": "over_top_k"} for c in kept[top_k:]]
    else:  # rerank
        vectors = np.asarray(index.reconstruct_n(0, len(candidates)), dtype="float32")
        final = rerank_candidates(query, candidates, vectors, qvec, top_k)
        selected_ids = {c["chunk_id"] for c in final}
        dropped = [{**c, "reason": "not_selected"} for c in candidates
                   if c["chunk_id"] not in selected_ids]
        overflow = []

    return {
        "mode": mode, "query": query, "chunks": final,
        "candidates": len(candidates), "kept": len(final),
        "dropped": dropped + overflow,
    }
