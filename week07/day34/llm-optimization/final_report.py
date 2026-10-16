"""Сборка итогового отчёта «до/после» из отчётов этапов.

Читает reports/{baseline,params_sweep,quant_comparison,prompt_ab,optimized}.json
и собирает reports/final_report.md:

  * baseline   — «до» (текущий Q4_K_M, дефолтные параметры, baseline-шаблон);
  * optimized  — «после» (лучший квант + лучшие параметры + лучший шаблон).

Также печатает рекомендуемую итоговую конфигурацию (JSON) для финального
прогона `bench.py run --label optimized ...`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import QUANTS, REPORTS_DIR


def _load(name: str):
    p = REPORTS_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _best(summary: dict) -> str:
    def key(item):
        lbl, s = item
        score = s.get("score_mean")
        score = score if score is not None else -1
        dur = s.get("total_duration_ms_mean")
        dur = dur if dur is not None else float("inf")
        return (score, -dur)
    return max(summary.items(), key=key)[0]


_DIM_KEYS = {
    "temperature": "temperature",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "num_ctx": "num_ctx",
}


def _best_options(sweep: dict) -> dict:
    """Достаёт options «best_combo» из финальных строк свипа (fallback: winners)."""
    for r in sweep.get("final_rows", []):
        if r.get("label") == "best_combo":
            return r.get("options", {})
    # fallback: собрать из winners по одному ключу на измерение
    winners = sweep.get("winners", {})
    opts = {}
    for dim, key in _DIM_KEYS.items():
        label = winners.get(dim)
        if not label:
            continue
        for r in sweep.get("coarse_rows", []):
            if r.get("label") == label and key in r.get("options", {}):
                opts[key] = r["options"][key]
                break
    return opts


def main() -> int:
    baseline = _load("baseline")
    sweep = _load("params_sweep")
    quant = _load("quant_comparison")
    prompt = _load("prompt_ab")
    optimized = _load("optimized")

    lines = ["# Итоговый отчёт: оптимизация локальной LLM (День 29)", ""]

    # Рекомендуемая конфигурация.
    best_quant_label = _best(quant["summary"]) if quant else None
    best_template = _best(prompt["summary"]) if prompt else None
    best_opts = _best_options(sweep) if sweep else None
    best_model = dict(QUANTS).get(best_quant_label) if best_quant_label else None

    lines += [
        "## Рекомендуемая конфигурация («после»)",
        "",
        f"- Квант: `{best_quant_label}` → `{best_model}`",
        f"- Шаблон: `{best_template}`",
        f"- Параметры: `{json.dumps(best_opts, ensure_ascii=False)}`",
        "",
    ]
    if sweep and sweep.get("winner_label"):
        lines.append(f"- Выигрышная комбинация параметров (свип): `{sweep['winner_label']}`")
        lines.append("")

    # Сводка «до/после».
    if baseline and optimized:
        b = baseline["summary"].get("baseline", {})
        o = optimized["summary"].get("optimized", {})
        lines += [
            "## До vs После",
            "",
            "| Метрика | До (baseline) | После (optimized) |",
            "|---|---|---|",
            f"| Score (0–2) | {b.get('score_mean')} | {o.get('score_mean')} |",
            f"| Ток/с | {b.get('tokens_per_sec_mean')} | {o.get('tokens_per_sec_mean')} |",
            f"| Длительность (мс) | {b.get('total_duration_ms_mean')} | {o.get('total_duration_ms_mean')} |",
            f"| Вход. токенов | {b.get('prompt_tokens_mean')} | {o.get('prompt_tokens_mean')} |",
            f"| Память (МБ) | {b.get('loaded_vram_mb')} | {o.get('loaded_vram_mb')} |",
            "",
        ]

    # Кванты.
    if quant:
        lines += ["## Квантование", "", "| Квант | Score | Ток/с | Длит. (мс) | Память (МБ) |",
                  "|---|---|---|---|---|"]
        for lbl, s in quant["summary"].items():
            lines.append(f"| {lbl} | {s['score_mean']} | {s['tokens_per_sec_mean']} | "
                         f"{s['total_duration_ms_mean']} | {s['loaded_vram_mb']} |")
        lines.append("")

    # Шаблоны.
    if prompt:
        lines += ["## Промпт-шаблон", "", "| Шаблон | Score | Ток/с | Длит. (мс) |",
                  "|---|---|---|---|"]
        for lbl, s in prompt["summary"].items():
            lines.append(f"| {lbl} | {s['score_mean']} | {s['tokens_per_sec_mean']} | "
                         f"{s['total_duration_ms_mean']} |")
        lines.append("")

    # Ссылки на исходные отчёты.
    lines += ["## Исходные отчёты", ""]
    for name in ["baseline", "params_sweep", "quant_comparison", "prompt_ab", "optimized"]:
        if (REPORTS_DIR / f"{name}.json").exists():
            lines.append(f"- `reports/{name}.md` · `reports/{name}.json`")

    out = REPORTS_DIR / "final_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)

    # Печатаем рекомендуемый прогон оптимизированной конфигурации.
    print("\nЗапустите финальный прогон:")
    print(f".venv/bin/python bench.py run --label optimized --models {best_model} "
          f"--options '{json.dumps(best_opts, ensure_ascii=False)}' "
          f"--template {best_template} --questions all --runs 3 --out optimized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
