"""Сравнение ответов модели без RAG и с RAG по контрольным вопросам.

Для каждого вопроса из data/golden.json генерирует два ответа (plain и rag),
сохраняет сырые данные в reports/rag_answers.json и (если не --no-judge)
запускает LLM-оценку качества (src/judge.py).

Использование (из папки rag, в rag/.venv):
    python -m src.compare --strategy structural --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embedder import Embedder
from .qa import answer_plain, answer_rag

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.json"
REPORT_DIR = ROOT / "reports"


def _retrieval_hit(relevant: list[str], chunks: list[dict]) -> bool:
    """Попал ли хоть один найденный чанк в релевантный источник (hit@top_k)."""
    if not relevant or not chunks:
        return False
    kws = [k.lower() for k in relevant]
    for c in chunks:
        hay = f"{c.get('source', '')} {c.get('title', '')}".lower()
        if any(k in hay for k in kws):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение без RAG vs с RAG")
    parser.add_argument("--strategy", "-s", default="structural", choices=["fixed", "structural"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="не запускать LLM-оценку")
    args = parser.parse_args(argv)

    questions = json.loads(GOLDEN.read_text(encoding="utf-8"))
    embedder = Embedder()
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[{i}/{len(questions)}] {question}")
        plain = answer_plain(question)
        rag = answer_rag(question, strategy=args.strategy, top_k=args.top_k,
                         embedder=embedder)
        results.append({
            "id": q.get("id"),
            "question": question,
            "expectation": q.get("expectation"),
            "sources": q.get("sources"),
            "relevant": q.get("relevant"),
            "strategy": args.strategy,
            "top_k": args.top_k,
            "plain_answer": plain,
            "rag_answer": rag["answer"],
            "rag_chunks": rag["chunks"],
            "retrieval_hit": _retrieval_hit(q.get("relevant"), rag["chunks"]),
        })
        print(f"  без RAG: {plain[:100].replace(chr(10), ' ')}")
        print(f"  с RAG:   {rag['answer'][:100].replace(chr(10), ' ')}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "rag_answers.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСырые ответы сохранены: {out}")

    if not args.no_judge:
        from .judge import run_judge
        run_judge(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
