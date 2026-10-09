"""Сравнение локальной и облачной LLM на RAG-пайплайне (День 28).

Прогоняет контрольные вопросы (data/golden.json) через два генератора:
  * local — Ollama (llama3.1), генерация полностью локально;
  * cloud — DeepSeek (облако).
Retrieval в обоих случаях локальный и идентичный (тот же индекс/эмбеддер),
поэтому различия в ответах относятся только к генератору.

Для каждого бэкенда делается N повторов на вопрос, замеряются скорость
(длительность, токены/сек) и стабильность (доля успешных вызовов, ошибки,
таймауты). Качество оценивает ФИКСИРОВАННЫЙ независимый судья
(JUDGE_LLM_PROVIDER, по умолчанию deepseek), чтобы не судить модель ею самой.

Итог: reports/llm_comparison.json (сырые данные) и reports/llm_comparison.md.

Использование (из папки rag, в rag/.venv):
    python -m src.compare_llm                 # 3 повтора, local + cloud, с судьёй
    python -m src.compare_llm --runs 1 --no-judge   # быстрый smoke скорости
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .embedder import Embedder
from .qa import answer_rag

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.json"
REPORT_DIR = ROOT / "reports"

# Подпись бэкенда → имя провайдера в src/llm.py.
PROVIDERS = {"local": "ollama", "cloud": "deepseek"}


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def _std(values: list[float]) -> float | None:
    return round(statistics.stdev(values), 2) if len(values) > 1 else None


def run_backend(label: str, questions: list[dict], args, embedder) -> list[dict]:
    """N повторов на вопрос через один бэкенд → список сырых записей."""
    provider = PROVIDERS[label]
    rows: list[dict] = []
    for q in questions:
        for run in range(1, args.runs + 1):
            row = {
                "id": q.get("id"), "question": q["question"],
                "expectation": q.get("expectation"), "backend": label,
                "provider": provider, "run": run,
            }
            t0 = time.perf_counter()
            try:
                rag = answer_rag(
                    q["question"], strategy=args.strategy, top_k=args.top_k,
                    mode=args.mode, min_score=args.min_score,
                    embedder=embedder, provider=provider,
                )
                row.update({
                    "status": "ok",
                    "answer": rag["answer"],
                    "abstained": bool(rag.get("abstained")),
                    "duration_ms": rag.get("duration_ms"),
                    "tokens_per_sec": rag.get("tokens_per_sec"),
                    "completion_tokens": (rag.get("usage") or {}).get("completion_tokens"),
                    "model": rag.get("model"),
                    "max_score": rag.get("max_score"),
                    "kept": rag.get("kept"),
                })
            except Exception as e:  # noqa: BLE001 — фиксируем сбой для метрики стабильности
                row.update({
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "answer": "",
                    "abstained": False,
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                })
            print(f"  [{label}/{run}] {q.get('id')} — {row['status']}"
                  f"{' (' + str(row.get('duration_ms')) + ' ms)' if row.get('duration_ms') is not None else ''}")
            rows.append(row)
    return rows


def judge_rows(rows: list[dict]) -> None:
    """Оценивает качество успешных ответов фиксированным судьёй (на месте)."""
    from .judge import judge_answer
    for r in rows:
        if r["status"] != "ok":
            r["score"] = None
            r["rationale"] = None
            continue
        j = judge_answer(r["question"], r["expectation"], r["answer"])
        r["score"] = j["score"]
        r["rationale"] = j["rationale"]


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] == "error"]
    abstains = [r for r in ok if r.get("abstained")]
    durations = [r["duration_ms"] for r in ok if r.get("duration_ms") is not None]
    tps = [r["tokens_per_sec"] for r in ok if r.get("tokens_per_sec")]
    scores = [r["score"] for r in ok if r.get("score") is not None]
    return {
        "runs": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(rows), 3) if rows else 0.0,
        "abstained": len(abstains),
        "duration_ms_mean": _mean(durations),
        "duration_ms_std": _std(durations),
        "tokens_per_sec_mean": _mean(tps),
        "tokens_per_sec_std": _std(tps),
        "score_mean": _mean([float(s) for s in scores]),
    }


def build_markdown(summary: dict, by_q: dict, labels: list[str], args) -> str:
    order = labels
    lines = [
        "# Сравнение локальной и облачной LLM (RAG, День 28)",
        "",
        f"Стратегия: `{args.strategy}` · top_k={args.top_k} · mode=`{args.mode}` · "
        f"повторов на вопрос: {args.runs}.",
        "",
        "Retrieval в обоих случаях локальный и идентичный; различается только генератор. "
        "Качество оценивает фиксированный судья (DeepSeek).",
        "",
        "## Сводка",
        "",
        "| Бэкенд | Score (0–2) | Успех | Ошибок | Abstain | Длит. (мс) | ±σ | Ток/сек |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in order:
        s = summary[label]
        dur = s["duration_ms_mean"]
        dstd = s["duration_ms_std"]
        tps = s["tokens_per_sec_mean"]
        lines.append(
            f"| {label} | {s['score_mean'] if s['score_mean'] is not None else '—'} | "
            f"{s['ok']}/{s['runs']} | {s['errors']} | {s['abstained']} | "
            f"{dur if dur is not None else '—'} | {dstd if dstd is not None else '—'} | "
            f"{tps if tps is not None else '—'} |"
        )

    lines += [
        "",
        "## По вопросам",
        "",
        "| # | Вопрос | local score | cloud score | local мс | cloud мс |",
        "|---|---|---|---|---|---|---|",
    ]
    for qid, per_q in by_q.items():
        question = per_q.get("question", "")
        ls = per_q.get("local", {})
        cs = per_q.get("cloud", {})
        def sc(d):
            return d["score_mean"] if d.get("score_mean") is not None else "—"
        def dm(d):
            return d["duration_ms_mean"] if d.get("duration_ms_mean") is not None else "—"
        lines.append(
            f"| {qid} | {question} | {sc(ls)} | {sc(cs)} | {dm(ls)} | {dm(cs)} |"
        )

    lines += ["", "## Детали по вопросам", ""]
    for qid, per_q in by_q.items():
        lines.append(f"### {qid}. {per_q.get('question')}")
        lines.append("")
        for label in order:
            d = per_q.get(label, {})
            if not d.get("runs"):
                continue
            lines.append(
                f"**{label}** — score {d.get('score_mean', '—')} · "
                f"успех {d.get('ok')}/{d.get('runs')} · длит. {d.get('duration_ms_mean', '—')} мс · "
                f"abstain {d.get('abstained')}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение local vs cloud LLM на RAG")
    parser.add_argument("--strategy", "-s", default="structural", choices=["fixed", "structural"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", default="baseline",
                        help="baseline|rewrite|filter|rewrite+filter|rerank")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--providers", default="local,cloud",
                        help="бэкенды через запятую (local,cloud)")
    parser.add_argument("--runs", type=int, default=3, help="повторов на вопрос")
    parser.add_argument("--questions", type=int, default=0, help="лимит вопросов (0 — все)")
    parser.add_argument("--no-judge", action="store_true", help="не оценивать качество (только скорость/стабильность)")
    args = parser.parse_args(argv)

    labels = [p.strip() for p in args.providers.split(",") if p.strip()]
    for label in labels:
        if label not in PROVIDERS:
            print(f"Неизвестный бэкенд: {label!r} (ожидается local|cloud)")
            return 1

    questions = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.questions > 0:
        questions = questions[: args.questions]

    embedder = Embedder()
    all_rows: list[dict] = []
    for label in labels:
        print(f"\n=== Бэкенд: {label} ({PROVIDERS[label]}) ===")
        all_rows.extend(run_backend(label, questions, args, embedder))

    if not args.no_judge:
        print("\n=== Оценка качества (фиксированный судья) ===")
        judge_rows(all_rows)

    # Сводка по бэкендам.
    summary: dict = {}
    by_q: dict = {}
    for label in labels:
        summary[label] = summarize([r for r in all_rows if r["backend"] == label])
    for q in questions:
        qid = q.get("id")
        by_q[qid] = {"question": q["question"]}
        for label in labels:
            by_q[qid][label] = summarize(
                [r for r in all_rows if r["backend"] == label and r["id"] == qid]
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "llm_comparison.json"
    out_md = REPORT_DIR / "llm_comparison.md"
    out_json.write_text(
        json.dumps({"args": vars(args), "summary": summary, "by_question": by_q,
                    "rows": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(build_markdown(summary, by_q, labels, args), encoding="utf-8")
    print(f"\nОтчёт: {out_md}")
    print(f"Данные: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
