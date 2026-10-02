"""Сравнение режимов RAG по контрольным вопросам (матрица режимов).

Для каждого вопроса из data/golden.json генерирует ответ без RAG (plain) и
ответы в каждом режиме (baseline / rewrite / filter / rewrite+filter / rerank),
сохраняет сырые данные в reports/rag_modes_answers.json и (если не --no-judge)
запускает LLM-оценку качества (src/judge.py → run_modes_judge).

Использование (из папки rag, в rag/.venv):
    python -m src.compare --strategy structural --top-k 5
    python -m src.compare --modes baseline,filter,rerank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .embedder import Embedder
from .qa import answer_plain, answer_rag

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.json"
REPORT_DIR = ROOT / "reports"


def chunk_stats(relevant_chunk_ids: list[str], chunks: list[dict]) -> dict:
    """Точность найденных чанков на уровне чанков (по точным id)."""
    if not chunks:
        return {"hit": False, "relevant_found": 0, "precision": 0.0}
    ids = set(relevant_chunk_ids or [])
    found = [c for c in chunks if c.get("chunk_id") in ids]
    return {
        "hit": len(found) > 0,
        "relevant_found": len(found),
        "precision": round(len(found) / len(chunks), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение режимов RAG (матрица)")
    parser.add_argument("--strategy", "-s", default="structural", choices=["fixed", "structural"])
    parser.add_argument("--top-k", type=int, default=config.TOP_K_FINAL)
    parser.add_argument("--top-k-candidates", type=int, default=config.TOP_K_CANDIDATES)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--modes", default=",".join(config.MODES),
                        help="режимы через запятую (baseline,rewrite,filter,rewrite+filter,rerank)")
    parser.add_argument("--no-judge", action="store_true", help="не запускать LLM-оценку")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = json.loads(GOLDEN.read_text(encoding="utf-8"))
    embedder = Embedder()
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[{i}/{len(questions)}] {question}")
        plain = answer_plain(question)
        modes_data: dict = {}
        for mode in modes:
            rag = answer_rag(
                question, strategy=args.strategy, top_k=args.top_k, mode=mode,
                top_k_candidates=args.top_k_candidates, min_score=args.min_score,
                embedder=embedder,
            )
            stats = chunk_stats(q.get("relevant_chunk_ids"), rag["chunks"])
            modes_data[mode] = {
                "answer": rag["answer"],
                "chunks": rag["chunks"],
                "rewritten_query": rag["rewritten_query"],
                "candidates": rag["candidates"],
                "kept": rag["kept"],
                "dropped": rag["dropped"],
                **stats,
            }
            print(f"  [{mode}] kept={rag['kept']}/{rag['candidates']} "
                  f"hit={stats['hit']} precision={stats['precision']}")
        results.append({
            "id": q.get("id"),
            "question": question,
            "expectation": q.get("expectation"),
            "sources": q.get("sources"),
            "relevant": q.get("relevant"),
            "relevant_chunk_ids": q.get("relevant_chunk_ids"),
            "strategy": args.strategy,
            "top_k": args.top_k,
            "top_k_candidates": args.top_k_candidates,
            "min_score": args.min_score,
            "modes": modes_data,
            "plain_answer": plain,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "rag_modes_answers.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСырые ответы сохранены: {out}")

    if not args.no_judge:
        from .judge import run_modes_judge
        run_modes_judge(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
