"""Бенчмарк оптимизации локальной LLM (День 29).

Прогоняет контрольные вопросы golden.json через локальную модель Ollama
(llama3.1) с разными конфигурациями (параметры инференса, квант, промпт-шаблон),
замеряет качество (независимый судья DeepSeek), скорость (серверные тайминги
Ollama) и потребление ресурсов (RSS, загруженная модель), и пишет отчёты.

Retrieval кэшируется один раз (data/chunks.json), поэтому контекст у всех
конфигураций одинаковый — различия относятся только к генератору.

Подкоманды:
  fetch-chunks        — скачать и закэшировать чанки для вопросов golden.json
  run                 — прогнать набор конфигураций (baseline / финалисты)
  sweep               — OFAT-перебор параметров + финалисты
  quant               — сравнение трёх квантов
  prompt-ab           — A/B промпт-шаблона (baseline vs variant)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from client import (
    LlmError,
    call_ollama_chat,
    judge_answer,
    snapshot_resources,
)
from config import (
    BASE_OPTIONS,
    CHUNKS_CACHE,
    COARSE_QUESTION_IDS,
    DATA_DIR,
    GOLDEN_PATH,
    MODEL_Q4,
    QUANTS,
    REPORTS_DIR,
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K,
    RETRIEVAL_URL,
    SWEEP_MAX_TOKENS,
    SWEEP_NUM_CTX,
    SWEEP_TEMPERATURE,
    SWEEP_TOP_K,
)
from prompts import TEMPLATES

import requests


# --------------------------------------------------------------------------- #
# Загрузка golden.json и кэш чанков
# --------------------------------------------------------------------------- #
def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def select_questions(questions: list[dict], spec: str) -> list[dict]:
    spec = (spec or "all").strip()
    if spec == "all":
        return questions
    if spec == "coarse":
        ids = COARSE_QUESTION_IDS
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    by_id = {q["id"]: q for q in questions}
    return [by_id[i] for i in ids if i in by_id]


def fetch_chunks(questions: list[dict]) -> dict:
    """Дёргает локальный retrieval-сервис и возвращает {qid: chunks}."""
    out = {}
    for q in questions:
        resp = requests.post(
            RETRIEVAL_URL + "/retrieve",
            json={
                "query": q["question"],
                "strategy": RETRIEVAL_STRATEGY,
                "top_k": RETRIEVAL_TOP_K,
                "mode": "baseline",
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        out[q["id"]] = {
            "chunks": data.get("chunks", []),
            "max_score": data.get("max_score"),
        }
    return out


def load_chunks() -> dict:
    if CHUNKS_CACHE.exists():
        return json.loads(CHUNKS_CACHE.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Нет кэша чанков {CHUNKS_CACHE}. Запусти fetch-chunks.")


def save_chunks(chunks: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_CACHE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #
def build_messages(question: str, chunks: list[dict], template: str) -> list[dict]:
    tpl = TEMPLATES[template]
    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": tpl["build"](question, chunks)},
    ]


def run_one(q: dict, chunks: list[dict], config: dict, run: int) -> dict:
    row = {
        "id": q["id"],
        "question": q["question"],
        "expectation": q.get("expectation"),
        "label": config["label"],
        "model": config["model"],
        "options": config["options"],
        "template": config["template"],
        "run": run,
    }
    try:
        res = call_ollama_chat(
            build_messages(q["question"], chunks, config["template"]),
            model=config["model"],
            options=config["options"],
        )
        row.update({
            "status": "ok",
            "answer": res["content"],
            "done_reason": res.get("done_reason"),
            "eval_count": res.get("eval_count"),
            "prompt_eval_count": res.get("prompt_eval_count"),
            "tokens_per_sec": res.get("tokens_per_sec"),
            "prompt_tokens_per_sec": res.get("prompt_tokens_per_sec"),
            "eval_duration_ms": res.get("eval_duration_ms"),
            "load_duration_ms": res.get("load_duration_ms"),
            "total_duration_ms": res.get("total_duration_ms"),
            "wall_ms": res.get("wall_ms"),
        })
        row.update(snapshot_resources(config["model"]))
    except (LlmError, requests.RequestException) as e:
        row.update({"status": "error", "error": f"{type(e).__name__}: {e}", "answer": ""})
    return row


def run_configs(configs: list[dict], questions: list[dict], chunks: dict,
                runs: int, judge: bool, verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    for cfg in configs:
        for q in questions:
            qchunks = chunks.get(q["id"], {}).get("chunks", [])
            for r in range(1, runs + 1):
                row = run_one(q, qchunks, cfg, r)
                rows.append(row)
                if verbose:
                    status = row["status"]
                    extra = ""
                    if status == "ok":
                        extra = (f" · {row.get('total_duration_ms')} мс · "
                                 f"{row.get('tokens_per_sec')} ток/с · "
                                 f"ctx {row.get('prompt_eval_count')}")
                    else:
                        extra = f" · {row.get('error', '')[:120]}"
                    print(f"  [{cfg['label']}/{q['id']}/{r}] {status}{extra}")
    if judge:
        print("\n=== Оценка качества (судья DeepSeek) ===")
        judge_rows(rows)
    return rows


def judge_rows(rows: list[dict]) -> None:
    for r in rows:
        if r["status"] != "ok":
            r["score"] = None
            r["rationale"] = None
            continue
        try:
            j = judge_answer(r["question"], r["expectation"], r["answer"])
            r["score"] = j["score"]
            r["rationale"] = j["rationale"]
        except (LlmError, requests.RequestException) as e:
            r["score"] = None
            r["rationale"] = f"judge error: {type(e).__name__}: {e}"
        print(f"  [{r['label']}/{r['id']}/{r['run']}] score={r['score']}")


# --------------------------------------------------------------------------- #
# Агрегация
# --------------------------------------------------------------------------- #
def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def _std(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.stdev(vals), 2) if len(vals) > 1 else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] == "error"]
    scored = [r for r in ok if r.get("score") is not None]
    return {
        "runs": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "score_mean": _mean([r["score"] for r in scored]),
        "score_std": _std([r["score"] for r in scored]),
        "total_duration_ms_mean": _mean([r.get("total_duration_ms") for r in ok]),
        "tokens_per_sec_mean": _mean([r.get("tokens_per_sec") for r in ok]),
        "prompt_tokens_mean": _mean([r.get("prompt_eval_count") for r in ok]),
        "load_ms_mean": _mean([r.get("load_duration_ms") for r in ok]),
        "rss_mb_max": max((r.get("ollama_rss_mb") or 0 for r in ok), default=0),
        "loaded_vram_mb": max((r.get("loaded_size_vram_mb") or 0 for r in ok), default=0),
    }


def group_by_label(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    return out


def save_report(name: str, payload: dict, md: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = REPORTS_DIR / f"{name}.json"
    mp = REPORTS_DIR / f"{name}.md"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(md, encoding="utf-8")
    return jp, mp


# --------------------------------------------------------------------------- #
# Подкоманды
# --------------------------------------------------------------------------- #
def cmd_fetch_chunks(args) -> int:
    questions = load_golden()
    print(f"Скачиваю чанки для {len(questions)} вопросов через {RETRIEVAL_URL}")
    chunks = fetch_chunks(questions)
    save_chunks(chunks)
    for qid, c in chunks.items():
        print(f"  {qid}: {len(c['chunks'])} чанков, max_score={c['max_score']}")
    print(f"Кэш: {CHUNKS_CACHE}")
    return 0


def _configs_from_args(args) -> list[dict]:
    options = json.loads(args.options) if args.options else dict(BASE_OPTIONS)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    templates = [t.strip() for t in (args.template or "baseline").split(",")]
    configs = []
    for model in models:
        for tpl in templates:
            configs.append({
                "label": args.label or f"{model}:{tpl}",
                "model": model,
                "options": options,
                "template": tpl,
            })
    return configs


def cmd_run(args) -> int:
    questions = select_questions(load_golden(), args.questions)
    chunks = load_chunks()
    configs = _configs_from_args(args)
    print(f"Конфигураций: {len(configs)}, вопросов: {len(questions)}, повторов: {args.runs}")
    rows = run_configs(configs, questions, chunks, args.runs, judge=not args.no_judge)
    summary = {k: summarize(v) for k, v in group_by_label(rows).items()}
    payload = {"args": vars(args), "summary": summary, "rows": rows}
    jp, _ = save_report(args.out, payload, _render_summary_md(summary, "run"))
    print(f"\nОтчёт: {jp}")
    for label, s in summary.items():
        print(f"  {label}: score={s['score_mean']} tok/s={s['tokens_per_sec_mean']} "
              f"dur={s['total_duration_ms_mean']} мс rss={s['loaded_vram_mb']} МБ")
    return 0


def _render_summary_md(summary: dict, title: str) -> str:
    lines = [f"# {title}", "",
             "| Конфигурация | Score (0–2) | Ток/с | Длит. (мс) | Вход. токенов | Загрузка (мс) | Память (МБ) |",
             "|---|---|---|---|---|---|---|---|"]
    for label, s in summary.items():
        lines.append(
            f"| {label} | {s['score_mean']} | {s['tokens_per_sec_mean']} | "
            f"{s['total_duration_ms_mean']} | {s['prompt_tokens_mean']} | "
            f"{s['load_ms_mean']} | {s['loaded_vram_mb']} |"
        )
    return "\n".join(lines) + "\n"


def cmd_sweep(args) -> int:
    all_q = load_golden()
    coarse_q = select_questions(all_q, "coarse")
    chunks = load_chunks()

    # OFAT-конфигурации: один параметр за раз от базы.
    base = dict(BASE_OPTIONS)
    configs = []
    for t in SWEEP_TEMPERATURE:
        configs.append({"label": f"temp={t}", "model": MODEL_Q4,
                        "options": {**base, "temperature": t}, "template": "baseline"})
    # top_k осмыслен только при temp > 0 (при temp=0 семплинг жадный).
    for k in SWEEP_TOP_K:
        configs.append({"label": f"top_k={k} (temp=0.2)", "model": MODEL_Q4,
                        "options": {**base, "temperature": 0.2, "top_k": k},
                        "template": "baseline"})
    for m in SWEEP_MAX_TOKENS:
        configs.append({"label": f"max_tokens={m}", "model": MODEL_Q4,
                        "options": {**base, "max_tokens": m}, "template": "baseline"})
    for c in SWEEP_NUM_CTX:
        configs.append({"label": f"num_ctx={c}", "model": MODEL_Q4,
                        "options": {**base, "num_ctx": c}, "template": "baseline"})

    print(f"=== Грубый перебор: {len(configs)} конфигураций × {len(coarse_q)} вопросов × 1 повтор ===")
    coarse_rows = run_configs(configs, coarse_q, chunks, runs=1, judge=True)

    coarse_summary = {k: summarize(v) for k, v in group_by_label(coarse_rows).items()}

    # Выбор финалистов: топ-3 по score + собранная лучшая комбинация.
    winners = _pick_winners(coarse_summary)
    finalists = _top_configs(configs, coarse_summary, winners, limit=4)

    print(f"\n=== Финалисты ({len(finalists)} конфигураций) × {len(all_q)} вопросов × {args.runs} повторов ===")
    final_rows = run_configs(finalists, all_q, chunks, runs=args.runs, judge=True)
    final_summary = {k: summarize(v) for k, v in group_by_label(final_rows).items()}

    winner_label = _best_label(final_summary)
    md = _render_sweep_md(coarse_summary, final_summary, winners, winner_label, args)
    payload = {
        "args": vars(args),
        "coarse_summary": coarse_summary,
        "final_summary": final_summary,
        "winners": winners,
        "winner_label": winner_label,
        "coarse_rows": coarse_rows,
        "final_rows": final_rows,
    }
    jp, mp = save_report("params_sweep", payload, md)
    print(f"\nОтчёт: {mp}")
    print(f"Данные: {jp}")
    print(f"\n>>> Выигрышная комбинация (ТОЛЬКО в отчёте, дефолтом не подставляется): {winner_label}")
    return 0


def _pick_winners(coarse_summary: dict) -> dict:
    """Для каждого измерения выбираем лучший лейбл по score (tie → быстрее)."""
    dims = {
        "temperature": [f"temp={t}" for t in SWEEP_TEMPERATURE],
        "top_k": [f"top_k={k} (temp=0.2)" for k in SWEEP_TOP_K],
        "max_tokens": [f"max_tokens={m}" for m in SWEEP_MAX_TOKENS],
        "num_ctx": [f"num_ctx={c}" for c in SWEEP_NUM_CTX],
    }
    winners = {}
    for dim, labels in dims.items():
        present = {lbl: coarse_summary[lbl] for lbl in labels if lbl in coarse_summary}
        if not present:
            continue
        winners[dim] = _best_label(present)
    return winners


def _best_label(summary: dict) -> str:
    def key(item):
        lbl, s = item
        score = s.get("score_mean")
        score = score if score is not None else -1
        dur = s.get("total_duration_ms_mean")
        dur = dur if dur is not None else float("inf")
        return (score, -dur)
    return max(summary.items(), key=key)[0]


# Ключ опций, который меняет каждое измерение свипа (для сборки best_combo).
_DIM_KEYS = {
    "temperature": "temperature",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "num_ctx": "num_ctx",
}


def _top_configs(configs: list[dict], summary: dict, winners: dict, limit: int) -> list[dict]:
    """Топ-N конфигураций по score + собранная комбинация победителей."""
    ranked = sorted(configs, key=lambda c: _sort_key(summary.get(c["label"])), reverse=True)
    picked = ranked[:limit]
    # Собираем «лучшую комбинацию»: для каждого измерения берём ТОЛЬКО его
    # выигрышное значение, не затирая остальные.
    best_options = dict(BASE_OPTIONS)
    for dim, key in _DIM_KEYS.items():
        label = winners.get(dim)
        if not label:
            continue
        cfg = next((c for c in configs if c["label"] == label), None)
        if cfg is not None and key in cfg["options"]:
            best_options[key] = cfg["options"][key]
    combo = {
        "label": "best_combo",
        "model": MODEL_Q4,
        "options": best_options,
        "template": "baseline",
    }
    if not any(c["label"] == "best_combo" for c in picked):
        picked.append(combo)
    return picked[: limit + 1]


def _sort_key(s):
    if not s:
        return (-1, -float("inf"))
    score = s.get("score_mean")
    score = score if score is not None else -1
    return (score, -(s.get("total_duration_ms_mean") or 0))


def _render_sweep_md(coarse, final, winners, winner_label, args) -> str:
    lines = [
        "# Подбор параметров инференса (OFAT)",
        "",
        f"Грубый перебор: 5 вопросов × 1 повтор. Финалисты: 10 вопросов × {args.runs} повторов.",
        f"Квант: {MODEL_Q4} (Q4_K_M). Шаблон: baseline.",
        "",
        "## Грубый перебор (по одному параметру)",
        "",
        "| Конфигурация | Score (0–2) | Ток/с | Длит. (мс) | Вход. токенов | Память (МБ) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in coarse.items():
        lines.append(
            f"| {label} | {s['score_mean']} | {s['tokens_per_sec_mean']} | "
            f"{s['total_duration_ms_mean']} | {s['prompt_tokens_mean']} | {s['loaded_vram_mb']} |"
        )
    lines += ["", "## Победители по измерениям", ""]
    for dim, label in winners.items():
        lines.append(f"- **{dim}** → `{label}`")
    lines += [
        "",
        "## Финалисты (10 вопросов × N повторов)",
        "",
        "| Конфигурация | Score (0–2) | ±σ | Ток/с | Длит. (мс) | Вход. токенов | Память (МБ) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in final.items():
        lines.append(
            f"| {label} | {s['score_mean']} | {s['score_std']} | {s['tokens_per_sec_mean']} | "
            f"{s['total_duration_ms_mean']} | {s['prompt_tokens_mean']} | {s['loaded_vram_mb']} |"
        )
    lines += [
        "",
        f"## Выигрышная комбинация: `{winner_label}`",
        "",
        "⚠️ Выведена только в отчёт — дефолтом НЕ подставляется.",
        "",
    ]
    return "\n".join(lines)


def cmd_quant(args) -> int:
    all_q = load_golden()
    questions = select_questions(all_q, args.questions or "all")
    chunks = load_chunks()
    configs = []
    for label, model in QUANTS:
        configs.append({"label": label, "model": model,
                        "options": dict(BASE_OPTIONS), "template": "baseline"})
    print(f"=== Кванты: {len(configs)} × {len(questions)} вопросов × {args.runs} повторов ===")
    rows = run_configs(configs, questions, chunks, args.runs, judge=not args.no_judge)
    summary = {k: summarize(v) for k, v in group_by_label(rows).items()}
    md = _render_quant_md(summary, questions, args)
    payload = {"args": vars(args), "summary": summary, "rows": rows}
    jp, mp = save_report("quant_comparison", payload, md)
    print(f"\nОтчёт: {mp}")
    return 0


def _render_quant_md(summary, questions, args) -> str:
    lines = [
        "# Сравнение квантования llama3.1 (RAG-QA)",
        "",
        f"Вопросов: {len(questions)} × {args.runs} повторов. Параметры: дефолт. Шаблон: baseline.",
        "",
        "| Квант | Score (0–2) | Ток/с | Длит. (мс) | Вход. токенов | Загрузка (мс) | Память (МБ) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in summary.items():
        lines.append(
            f"| {label} | {s['score_mean']} | {s['tokens_per_sec_mean']} | "
            f"{s['total_duration_ms_mean']} | {s['prompt_tokens_mean']} | "
            f"{s['load_ms_mean']} | {s['loaded_vram_mb']} |"
        )
    return "\n".join(lines) + "\n"


def cmd_prompt_ab(args) -> int:
    all_q = load_golden()
    questions = select_questions(all_q, args.questions or "all")
    chunks = load_chunks()
    configs = [
        {"label": "baseline", "model": MODEL_Q4, "options": dict(BASE_OPTIONS), "template": "baseline"},
        {"label": "variant", "model": MODEL_Q4, "options": dict(BASE_OPTIONS), "template": "variant"},
    ]
    print(f"=== A/B шаблона: 2 × {len(questions)} вопросов × {args.runs} повторов ===")
    rows = run_configs(configs, questions, chunks, args.runs, judge=not args.no_judge)
    summary = {k: summarize(v) for k, v in group_by_label(rows).items()}
    md = _render_summary_md(summary, "A/B промпт-шаблона (baseline vs variant)")
    payload = {"args": vars(args), "summary": summary, "rows": rows}
    jp, mp = save_report("prompt_ab", payload, md)
    print(f"\nОтчёт: {mp}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Бенчмарк оптимизации локальной LLM")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-chunks", help="закэшировать чанки retrieval")

    r = sub.add_parser("run", help="прогнать конфигурации")
    r.add_argument("--label")
    r.add_argument("--models", default=MODEL_Q4)
    r.add_argument("--options", help="JSON-строка параметров (иначе BASE_OPTIONS)")
    r.add_argument("--template", default="baseline")
    r.add_argument("--questions", default="all", help="all|coarse|id1,id2")
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--out", default="run")
    r.add_argument("--no-judge", action="store_true")

    s = sub.add_parser("sweep", help="OFAT-перебор параметров")
    s.add_argument("--runs", type=int, default=3)

    q = sub.add_parser("quant", help="сравнение квантов")
    q.add_argument("--questions", default="all")
    q.add_argument("--runs", type=int, default=3)
    q.add_argument("--no-judge", action="store_true")

    a = sub.add_parser("prompt-ab", help="A/B шаблона")
    a.add_argument("--questions", default="all")
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--no-judge", action="store_true")

    args = p.parse_args(argv)
    handlers = {
        "fetch-chunks": cmd_fetch_chunks,
        "run": cmd_run,
        "sweep": cmd_sweep,
        "quant": cmd_quant,
        "prompt-ab": cmd_prompt_ab,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
