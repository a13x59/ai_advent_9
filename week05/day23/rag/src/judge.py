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

from .qa import call_llm

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
        temperature=0.0, max_tokens=256,
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
