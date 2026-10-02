"""Точка входа пайплайна (CLI).

Использование:
  python -m src.pipeline run                     # собрать индексы обеих стратегий
  python -m src.pipeline eval                     # метрики на тест-сете
  python -m src.pipeline query -s structural "Что такое word2vec?"
  python -m src.pipeline list                     # список документов
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .chunkers import chunk_document
from .embedder import EMBEDDING_DIM, Embedder, MODEL_NAME
from .evaluate import QUERIES, evaluate_strategy, threshold_sweep
from .indexer import build_index, load_index
from .loader import load_documents

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "indexes"
REPORT_DIR = ROOT / "reports"
STRATEGIES = ["fixed", "structural"]


def cmd_list(args: argparse.Namespace) -> None:
    docs = load_documents(DATA_DIR)
    total = sum(len(d.text) for d in docs)
    print(f"Документов: {len(docs)}, суммарно {total:,} симв.")
    for d in docs:
        print(f"  [{d.file_type:7}] {d.source:45} {len(d.text):>8,} симв.")


def cmd_run(args: argparse.Namespace) -> None:
    docs = load_documents(DATA_DIR)
    total_chars = sum(len(d.text) for d in docs)
    print(f"Загружено документов: {len(docs)} ({total_chars:,} симв.)")
    print(f"Модель эмбеддингов: {args.model}")

    embedder = Embedder(args.model)
    summary: dict = {}

    for strategy in STRATEGIES:
        print(f"\n=== Стратегия: {strategy} ===")
        t0 = time.time()
        chunks = []
        for d in docs:
            chunks.extend(chunk_document(d, strategy))
        t_chunk = time.time() - t0

        texts = [c.text for c in chunks]
        t1 = time.time()
        embeddings = embedder.embed(texts, batch_size=args.batch_size)
        t_embed = time.time() - t1

        t2 = time.time()
        manifest = build_index(chunks, embeddings, strategy, INDEX_DIR,
                               embedder.model_name, EMBEDDING_DIM)
        t_build = time.time() - t2

        manifest.update({
            "chunk_time_sec": round(t_chunk, 3),
            "embed_time_sec": round(t_embed, 3),
            "build_time_sec": round(t_build, 3),
            "indexing_time_sec": round(t_chunk + t_embed + t_build, 3),
            "faiss_size_mb": round(manifest["faiss_size_bytes"] / 1e6, 3),
            "db_size_mb": round(manifest["db_size_bytes"] / 1e6, 3),
        })
        summary[strategy] = manifest
        print(
            f"  чанков: {manifest['n_chunks']}, "
            f"средняя длина: {manifest['avg_len']:.0f} симв. "
            f"(std {manifest['std_len']:.0f})"
        )
        print(
            f"  время: chunk {t_chunk:.2f}s | embed {t_embed:.2f}s | "
            f"build {t_build:.2f}s | всего {manifest['indexing_time_sec']:.2f}s"
        )
        print(f"  размер: faiss {manifest['faiss_size_mb']} MB | db {manifest['db_size_mb']} MB")

    (INDEX_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nСводка сохранена в {INDEX_DIR / 'summary.json'}")


def cmd_eval(args: argparse.Namespace) -> None:
    embedder = Embedder(args.model)
    results: dict = {}

    for strategy in STRATEGIES:
        index, records, manifest = load_index(strategy, INDEX_DIR)
        print(f"\n=== Оценка стратегии: {strategy} "
              f"({manifest['n_chunks']} чанков) ===")
        res = evaluate_strategy(index, records, embedder, QUERIES, top_k=args.top_k)
        results[strategy] = res
        a = res["avg"]
        print(f"  оценено запросов: {res['n_scored']}/{res['n_queries']}")
        print(f"  Recall@5 = {a['recall@5']:.4f}  MRR@10 = {a['mrr@10']:.4f}  "
              f"nDCG@10 = {a['ndcg@10']:.4f}")
        print(f"  Precision@5 = {a['precision@5']:.4f}  "
              f"Precision@10 = {a['precision@10']:.4f}")
        for p in res["per_query"]:
            if p["recall@5"] is None:
                print(f"    - [SKIP] {p['query']} ({p['warn']})")
            else:
                print(f"    - R@5={p['recall@5']:.3f} P@5={p['precision@5']:.3f} "
                      f"MRR={p['mrr@10']:.3f} nDCG={p['ndcg@10']:.3f}  {p['query']}")

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nМетрики сохранены в {REPORT_DIR / 'metrics.json'}")


def cmd_sweep(args: argparse.Namespace) -> None:
    embedder = Embedder(args.model)
    results: dict = {}

    for strategy in STRATEGIES:
        index, records, manifest = load_index(strategy, INDEX_DIR)
        print(f"\n=== Sweep порога, стратегия: {strategy} "
              f"({manifest['n_chunks']} чанков) ===")
        res = threshold_sweep(index, records, embedder, QUERIES,
                              top_k_candidates=args.top_k_candidates,
                              top_k_final=args.top_k_final)
        results[strategy] = res
        print(f"  {'threshold':>10} {'precision':>10} {'recall':>8} {'avg_kept':>9}")
        for row in res["thresholds"]:
            print(f"  {row['threshold']:>10.2f} {row['precision']:>10.4f} "
                  f"{row['recall']:>8.4f} {row['avg_kept']:>9.2f}")

    (REPORT_DIR / "threshold_sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSweep сохранён в {REPORT_DIR / 'threshold_sweep.json'}")


def cmd_query(args: argparse.Namespace) -> None:
    embedder = Embedder(args.model)
    index, records, manifest = load_index(args.strategy, INDEX_DIR)
    emb = embedder.embed([args.query], batch_size=1, show_progress=False)
    dists, idxs = index.search(emb, args.top_k)
    print(f"Стратегия: {args.strategy} | запрос: {args.query!r}\n")
    for rank, (d, row) in enumerate(zip(dists[0], idxs[0]), 1):
        rec = records[row]
        print(f"#{rank}  score={d:.4f}")
        print(f"    источник: {rec['source']}")
        print(f"    раздел:   {rec['section'] or '-'}")
        print(f"    текст:    {rec['text'][:240].strip()}")
        print(f"    id:       {rec['chunk_id']}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG-пайплайн индексации")
    parser.add_argument("--model", default=MODEL_NAME)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="собрать индексы")
    p_run.add_argument("--batch-size", type=int, default=32)
    p_eval = sub.add_parser("eval", help="метрики на тест-сете")
    p_eval.add_argument("--top-k", type=int, default=10)
    p_sweep = sub.add_parser("sweep", help="сетка порогов отсечения (precision/recall)")
    p_sweep.add_argument("--top-k-candidates", type=int, default=20)
    p_sweep.add_argument("--top-k-final", type=int, default=5)
    p_query = sub.add_parser("query", help="поисковый запрос")
    p_query.add_argument("--strategy", "-s", choices=STRATEGIES, default="structural")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("query")
    sub.add_parser("list", help="список документов")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "query":
        cmd_query(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
