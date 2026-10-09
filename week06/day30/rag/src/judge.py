"""LLM-as-judge: оценка качества ответов без RAG и с RAG против ожидания.

Каждый ответ оценивается от 0 до 2 по полю expectation контрольного вопроса.
Итог — reports/rag_comparison.md (сводная таблица + примеры) и
reports/rag_comparison.json (полные данные с оценками).

Может запускаться отдельно от уже сохранённых reports/rag_answers.json:
    python -m src.judge
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .llm import JUDGE_PROVIDER, call_llm

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

JUDGE_SYSTEM = (
    "Ты — строгий и беспристрастный судья. Оценивай ответ ассистента относительно "
    "ожидания и ТОЛЬКО по заданным критериям. Не добавляй ничего сверх запрошенного."
)

JUDGE_PROMPT = """Оцени ответ ассистента на вопрос.

Вопрос: {question}

Ожидание (что должно быть в ответе): {expectation}

Ответ ассистента:
{answer}

Критерии оценки (score):
- 2 — ответ полностью соответствует ожиданию: названы все ключевые элементы, фактических ошибок и выдумок нет;
- 1 — ответ частично соответствует: есть часть ключевых элементов, но неполно или с неточностями;
- 0 — ответ не соответствует ожиданию: ключевые элементы отсутствуют, неверны или выдуманы.

Верни СТРОГО JSON-объект вида {{"score": <0|1|2>, "rationale": "<краткое обоснование>"}}.
"""


def judge_answer(question: str, expectation: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, expectation=expectation, answer=answer)
    raw = call_llm(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=256, provider=JUDGE_PROVIDER,
    )
    try:
        obj = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
    score = obj.get("score", 0)
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    score = max(0, min(2, int(score)))
    return {"score": score, "rationale": str(obj.get("rationale") or "")}


GROUNDED_PROMPT = """Оцени, соответствует ли смысл ответа ассистента приведённым цитатам из базы знаний.

Вопрос: {question}

Ответ ассистента (раздел «## Ответ»):
{answer}

Цитаты (дословные фрагменты из найденных чанков):
{citations}

Критерии (score):
- 2 — ответ полностью опирается на цитаты: все ключевые утверждения подтверждаются цитатами, утверждений вне цитат нет;
- 1 — ответ в основном опирается на цитаты, но есть незначительные отступления или детали, не подтверждённые цитатами;
- 0 — ответ противоречит цитатам либо в основном не подтверждается ими.

Верни СТРОГО JSON-объект вида {{"score": <0|1|2>, "rationale": "<краткое обоснование>"}}.
"""


def judge_groundedness(question: str, answer: str, citations: str) -> dict:
    """LLM-judge: насколько смысл ответа соответствует дословным цитатам."""
    prompt = GROUNDED_PROMPT.format(question=question, answer=answer, citations=citations)
    raw = call_llm(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=256, provider=JUDGE_PROVIDER,
    )
    try:
        obj = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
    score = obj.get("score", 0)
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    score = max(0, min(2, int(score)))
    return {"score": score, "rationale": str(obj.get("rationale") or "")}


def _winner(plain: int, rag: int) -> str:
    if rag > plain:
        return "rag"
    if plain > rag:
        return "plain"
    return "tie"


def build_report(scored: list[dict]) -> str:
    n = len(scored)
    avg_plain = sum(r["judge_plain"]["score"] for r in scored) / n if n else 0.0
    avg_rag = sum(r["judge_rag"]["score"] for r in scored) / n if n else 0.0
    wins_rag = sum(1 for r in scored if r["winner"] == "rag")
    wins_plain = sum(1 for r in scored if r["winner"] == "plain")
    ties = sum(1 for r in scored if r["winner"] == "tie")
    hits = sum(1 for r in scored if r.get("retrieval_hit"))

    lines = [
        "# Сравнение: ответ без RAG vs ответ с RAG",
        "",
        f"Вопросов: {n}. Средняя оценка (0–2): **без RAG — {avg_plain:.2f}**, "
        f"**с RAG — {avg_rag:.2f}**.",
        f"С RAG лучше: {wins_rag} | без RAG лучше: {wins_plain} | ничья: {ties}.",
        f"Релевантный источник попал в top-k: {hits}/{n}.",
        "",
        "| # | Вопрос | Без RAG | С RAG | Итог |",
        "|---|---|---|---|---|",
    ]
    for r in scored:
        label = {"rag": "✅ RAG лучше", "plain": "⚠️ без RAG лучше", "tie": "➖ одинаково"}[r["winner"]]
        lines.append(
            f"| {r.get('id')} | {r['question']} | {r['judge_plain']['score']}/2 | "
            f"{r['judge_rag']['score']}/2 | {label} |"
        )
    lines += [
        "",
        "## Детали по вопросам",
        "",
    ]
    for r in scored:
        lines += [
            f"### {r.get('id')}. {r['question']}",
            "",
            f"**Ожидание:** {r['expectation']}",
            f"**Источники:** {', '.join(r.get('sources') or [])}",
            "",
            f"**Без RAG ({r['judge_plain']['score']}/2):** {r['judge_plain']['rationale']}",
            f"**С RAG ({r['judge_rag']['score']}/2):** {r['judge_rag']['rationale']}",
            "",
            "Ответ без RAG:",
            "",
            f"> {r['plain_answer'].strip()}",
            "",
            "Ответ с RAG:",
            "",
            f"> {r['rag_answer'].strip()}",
            "",
            "Использованные чанки (источники):",
            "",
        ]
        for c in r.get("rag_chunks") or []:
            lines.append(f"- `{c.get('source')}` — {c.get('title')} (score {c.get('score', 0):.3f})")
        lines.append("")
    return "\n".join(lines)


def run_judge(results: list[dict]) -> list[dict]:
    for r in results:
        print(f"\nОценка [{r.get('id')}] {r['question']}")
        plain = judge_answer(r["question"], r["expectation"], r["plain_answer"])
        rag = judge_answer(r["question"], r["expectation"], r["rag_answer"])
        r["judge_plain"] = plain
        r["judge_rag"] = rag
        r["winner"] = _winner(plain["score"], rag["score"])
        print(f"  без RAG: {plain['score']}/2 | с RAG: {rag['score']}/2 | {r['winner']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rag_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "rag_comparison.md").write_text(
        build_report(results), encoding="utf-8"
    )
    print(f"\nОтчёт: {REPORT_DIR / 'rag_comparison.md'}")
    print(f"Данные: {REPORT_DIR / 'rag_comparison.json'}")
    return results


# --------------------------------------------------------------------------- #
# Матрица режимов (День 23): plain vs baseline/rewrite/filter/rewrite+filter/rerank
# --------------------------------------------------------------------------- #
def _modes_order(scored: list[dict]) -> list[str]:
    if not scored:
        return []
    from . import config
    order = []
    for m in config.MODES:
        if any(m in (r.get("modes") or {}) for r in scored):
            order.append(m)
    for m in scored[0].get("modes", {}):
        if m not in order:
            order.append(m)
    return order


def build_modes_report(scored: list[dict]) -> str:
    n = len(scored)
    modes = _modes_order(scored)
    avg_plain = sum(r["judge_plain"]["score"] for r in scored) / n if n else 0.0

    stats = {m: {"avg": 0.0, "better": 0, "worse": 0, "same": 0, "hits": 0,
                 "precision": 0.0, "kept": 0.0} for m in modes}
    baseline = "baseline"
    for r in scored:
        bscore = None
        if baseline in (r.get("modes") or {}):
            bscore = r["judge_modes"][baseline]["score"]
        for mode in modes:
            m = r["modes"].get(mode)
            if m is None:
                continue
            st = stats[mode]
            st["avg"] += r["judge_modes"][mode]["score"]
            st["hits"] += 1 if m.get("hit") else 0
            st["precision"] += m.get("precision", 0.0)
            st["kept"] += m.get("kept", 0)
            if bscore is not None:
                ms = r["judge_modes"][mode]["score"]
                if ms > bscore:
                    st["better"] += 1
                elif ms < bscore:
                    st["worse"] += 1
                else:
                    st["same"] += 1

    lines = [
        "# Сравнение режимов RAG (реранкинг и фильтрация)",
        "",
        f"Вопросов: {n}. Средняя оценка (0–2): **без RAG — {avg_plain:.2f}**.",
        "",
        "Режимы: baseline (без второго этапа) · rewrite (LLM-переформулировка) · "
        "filter (порог similarity) · rewrite+filter · rerank (лексическая эвристика + MMR).",
        "",
        "| Режим | Средний score | Лучше baseline | Хуже baseline | Hit@top-k | Precision | Ср. kept |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in modes:
        st = stats[m]
        avg = st["avg"] / n if n else 0.0
        prec = st["precision"] / n if n else 0.0
        kept = st["kept"] / n if n else 0.0
        vs = f"+{st['better']}/−{st['worse']}" if baseline in modes else "—"
        lines.append(
            f"| {m} | {avg:.2f} | {st['better']} | {st['worse']} | "
            f"{st['hits']}/{n} | {prec:.2f} | {kept:.2f} |"
        )
    lines += [
        "",
        "## По вопросам",
        "",
        "| # | Вопрос | plain | " + " | ".join(modes) + " |",
        "|---|---|---|" + "---|" * len(modes),
    ]
    for r in scored:
        cells = [f"{r.get('id')}", r["question"], f"{r['judge_plain']['score']}/2"]
        for m in modes:
            if m in r.get("modes", {}):
                cells.append(f"{r['judge_modes'][m]['score']}/2")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Детали по вопросам", ""]
    for r in scored:
        lines.append(f"### {r.get('id')}. {r['question']}")
        lines.append("")
        lines.append(f"**Ожидание:** {r['expectation']}")
        lines.append("")
        lines.append(f"**Без RAG ({r['judge_plain']['score']}/2):** "
                     f"{r['judge_plain']['rationale']}")
        lines.append("")
        for m in modes:
            md = r.get("modes", {}).get(m)
            if md is None:
                continue
            j = r["judge_modes"][m]
            rewrite = ""
            if md.get("rewritten_query"):
                rewrite = f" · rewrite: «{md['rewritten_query']}»"
            kept = md.get("kept")
            dropped = len(md.get("dropped") or [])
            meta = f"kept {kept}" + (f" / отсечено {dropped}" if dropped else "")
            lines += [
                f"**{m} ({j['score']}/2)** — {meta}{rewrite}",
                "",
                f"> {md['answer'].strip()[:400]}",
                "",
                "Чанки:",
                "",
            ]
            for c in md.get("chunks") or []:
                lines.append(f"- `{c.get('source')}` — {c.get('title')} "
                             f"(score {c.get('score', 0):.3f})")
            lines.append("")
    return "\n".join(lines)


def run_modes_judge(results: list[dict]) -> list[dict]:
    for r in results:
        print(f"\nОценка [{r.get('id')}] {r['question']}")
        r["judge_plain"] = judge_answer(r["question"], r["expectation"], r["plain_answer"])
        r["judge_modes"] = {}
        for mode, m in r.get("modes", {}).items():
            r["judge_modes"][mode] = judge_answer(r["question"], r["expectation"], m["answer"])
            print(f"  {mode}: {r['judge_modes'][mode]['score']}/2")
        print(f"  plain: {r['judge_plain']['score']}/2")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rag_modes.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "rerank_comparison.md").write_text(
        build_modes_report(results), encoding="utf-8"
    )
    print(f"\nОтчёт: {REPORT_DIR / 'rerank_comparison.md'}")
    print(f"Данные: {REPORT_DIR / 'rag_modes.json'}")
    return results


def main() -> int:
    answers_path = REPORT_DIR / "rag_answers.json"
    if not answers_path.exists():
        print(f"Не найден {answers_path}. Сначала запустите: python -m src.compare --no-judge")
        return 1
    results = json.loads(answers_path.read_text(encoding="utf-8"))
    run_judge(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
