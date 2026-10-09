"""Проверка обязательных источников и цитат (День 24).

Прогоняет контрольные вопросы из data/golden.json (10 вопросов) и
слаборелевантные вопросы из data/weak.json через answer_rag, после чего:

  1. детерминированно проверяет, что в ответе есть секции «Источники» и
     «Цитаты», что каждая цитата — дословный фрагмент найденного чанка
     (подстрока после нормализации), а источник содержит chunk_id;
  2. LLM-judge оценивает, совпадает ли смысл ответа с цитатами;
  3. для слабых вопросов проверяет срабатывание режима «не знаю».

Результат — reports/citations_report.md и reports/citations_report.json.

Использование (из папки rag, в rag/.venv):
    python -m src.citations
    python -m src.citations --mode rerank --top-k 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config
from .embedder import Embedder
from .judge import judge_groundedness
from .qa import answer_rag

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.json"
WEAK = ROOT / "data" / "weak.json"
REPORT_DIR = ROOT / "reports"

# --------------------------------------------------------------------------- #
# Парсинг markdown-ответа (три секции)
# --------------------------------------------------------------------------- #
_SECTION_RE = re.compile(r"^#{2,4}\s*(Ответ|Источники|Цитаты)\s*$", re.M)

_SOURCE_RE = re.compile(r"source\s*[:=]\s*([^\s|,;]+)")
_CHUNK_ID_RE = re.compile(r"chunk_id\s*[:=]\s*([^\s|,;]+)")
_SECTION_FIELD_RE = re.compile(
    r"section\s*[:=]\s*(.*?)(?=\s*\|\s*chunk_id\b|\s*\|\s*$|\s*$)", re.S
)
_SRC_ITEM_RE = re.compile(r"^\s*[-*]\s*\[(\d+)\]\s*(.*)$")
_QUOTES = "«»\u201c\u201d\u0022\u0027"
_QUOTE_ITEM_RE = re.compile(
    rf"^\s*[-*]\s*\[(\d+)\]\s*[{_QUOTES}](.+?)[{_QUOTES}]\s*$"
)

_EDGE_PUNCT = " \t\n\r.,;:!?…«»\"'()[]—-–"


def _field(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _normalize(s: str) -> str:
    """Нормализация для проверки «цитата — подстрока чанка» (уровень 1)."""
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"\\[A-Za-z]+", " ", s)  # LaTeX-команды (\displaystyle и т.п.)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(_EDGE_PUNCT)


def _canonical(s: str) -> str:
    """Каноническая форма: только буквы и цифры (уровень 2, устойчив к LaTeX).

    Позволяет считать дословной цитату «W_Q», когда в чанке формула записана
    как «W_{Q}» или разбита переносами строк.
    """
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^a-z0-9]+", "", s)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower().replace("ё", "е"))


def _ordered_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(t in it for t in needle)


def _contains(needle: str, hay: str) -> bool:
    """Есть ли needle (цитата) в hay (текст чанка) после нормализации.

    Три ступени: (1) подстрока после нормализации whitespace/пунктуации;
    (2) подстрока канонической формы (только буквы/цифры) — устойчиво к LaTeX;
    (3) упорядоченная подпоследовательность токенов — на случай, когда формула
    в чанке записана дважды (видимый символ + LaTeX-альтернатива), как в
    википедийных статьях.
    """
    if not needle:
        return False
    if _normalize(needle) in _normalize(hay):
        return True
    c = _canonical(needle)
    if c and c in _canonical(hay):
        return True
    nt, ht = _tokens(needle), _tokens(hay)
    return len(nt) >= 3 and len(nt) <= len(ht) and _ordered_subsequence(nt, ht)


def section_blocks(answer: str) -> dict[str, str]:
    """Делит ответ на блоки «Ответ» / «Источники» / «Цитаты»."""
    blocks: dict[str, str] = {"Ответ": "", "Источники": "", "Цитаты": ""}
    matches = list(_SECTION_RE.finditer(answer or ""))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        blocks[name] = answer[start:end].strip()
    return blocks


def parse_sources(answer: str) -> list[dict]:
    """Извлекает источники из секции «Источники» → [{n, source, section, chunk_id}]."""
    sec = section_blocks(answer)["Источники"]
    items: list[dict] = []
    for line in sec.splitlines():
        m = _SRC_ITEM_RE.match(line.strip())
        if not m:
            continue
        rest = m.group(2)
        items.append({
            "n": int(m.group(1)),
            "source": _field(_SOURCE_RE, rest),
            "section": _field(_SECTION_FIELD_RE, rest),
            "chunk_id": _field(_CHUNK_ID_RE, rest),
            "raw": line.strip(),
        })
    return items


def parse_citations(answer: str) -> list[dict]:
    """Извлекает цитаты из секции «Цитаты» → [{n, quote}]."""
    sec = section_blocks(answer)["Цитаты"]
    items: list[dict] = []
    for line in sec.splitlines():
        m = _QUOTE_ITEM_RE.match(line.strip())
        if m:
            items.append({"n": int(m.group(1)), "quote": m.group(2).strip()})
    return items


# --------------------------------------------------------------------------- #
# Детерминированные проверки
# --------------------------------------------------------------------------- #
def _chunk_by_n(chunks: list[dict]) -> dict[int, dict]:
    return {i: c for i, c in enumerate(chunks, 1)}


def check_sources(sources: list[dict], chunks: list[dict]) -> dict:
    """Есть ли источники и совпадают ли их chunk_id с найденными чанками."""
    chunk_ids = {c.get("chunk_id") for c in chunks}
    matched = []
    for s in sources:
        ok = s["chunk_id"] in chunk_ids or s["source"] in {c.get("source") for c in chunks}
        matched.append({**s, "matched": bool(ok)})
    present = len(sources) > 0
    return {
        "present": present,
        "total": len(sources),
        "matched": sum(1 for m in matched if m["matched"]),
        "items": matched,
    }


def check_citations(citations: list[dict], chunks: list[dict]) -> dict:
    """Есть ли цитаты и является ли каждая дословным фрагментом чанка."""
    by_n = _chunk_by_n(chunks)
    checked = []
    for q in citations:
        n = q["n"]
        hay_candidates = []
        if n in by_n:
            hay_candidates.append((n, by_n[n].get("text", "")))
        # fallback: цитата может ссылаться на другой номер — проверим все чанки
        hay_candidates.extend(
            (i, c.get("text", "")) for i, c in by_n.items() if i != n
        )
        matched_n = None
        for i, hay in hay_candidates:
            if _contains(q["quote"], hay):
                matched_n = i
                break
        checked.append({**q, "matched": matched_n is not None, "matched_chunk": matched_n})
    present = len(citations) > 0
    return {
        "present": present,
        "total": len(citations),
        "matched": sum(1 for c in checked if c["matched"]),
        "items": checked,
    }


def _abstain_ok(answer: str) -> bool:
    return "не знаю" in (answer or "").lower() and "уточн" in (answer or "").lower()


# --------------------------------------------------------------------------- #
# Прогон и отчёт
# --------------------------------------------------------------------------- #
def run(questions: list[dict], weak: list[dict], strategy: str, top_k: int,
        mode: str, min_score: float | None, do_judge: bool = True) -> list[dict]:
    embedder = Embedder()
    results: list[dict] = []

    def process(q: dict, weak_flag: bool) -> dict:
        question = q["question"]
        rag = answer_rag(
            question, strategy=strategy, top_k=top_k, mode=mode,
            min_score=min_score, embedder=embedder,
        )
        answer = rag["answer"]
        chunks = rag["chunks"]
        blocks = section_blocks(answer)
        sources = parse_sources(answer)
        citations = parse_citations(answer)
        src_check = check_sources(sources, chunks)
        cit_check = check_citations(citations, chunks)

        grounded = None
        if not rag.get("abstained") and do_judge:
            grounded = judge_groundedness(
                question, blocks["Ответ"], blocks["Цитаты"]
            )

        return {
            "id": q.get("id"),
            "question": question,
            "weak": weak_flag,
            "mode": mode,
            "strategy": strategy,
            "top_k": top_k,
            "abstained": bool(rag.get("abstained")),
            "max_score": rag.get("max_score"),
            "below_relevance": rag.get("below_relevance"),
            "answer": answer,
            "answer_block": blocks["Ответ"],
            "chunks": chunks,
            "sources": src_check,
            "citations": cit_check,
            "grounded": grounded,
            "abstain_ok": _abstain_ok(answer) if rag.get("abstained") else None,
        }

    for q in questions:
        print(f"[in-domain] {q['question']}")
        results.append(process(q, weak_flag=False))
    for q in weak:
        print(f"[weak] {q['question']}")
        results.append(process(q, weak_flag=True))

    return results


def build_markdown(results: list[dict]) -> str:
    domain = [r for r in results if not r["weak"]]
    weak = [r for r in results if r["weak"]]

    def yesno(v: bool) -> str:
        return "✅" if v else "❌"

    lines = [
        "# Проверка обязательных источников и цитат (День 24)",
        "",
        f"Контрольных вопросов: {len(domain)} · слаборелевантных: {len(weak)}.",
        "",
        "## Контрольные вопросы (in-domain)",
        "",
        "| # | Вопрос | Источники | Цитаты | Цитаты — подстроки | Смысл = цитаты |",
        "|---|---|---|---|---|---|",
    ]
    for r in domain:
        grounded = r["grounded"]
        g = f"{grounded['score']}/2" if grounded else "—"
        lines.append(
            f"| {r['id']} | {r['question']} | "
            f"{yesno(r['sources']['present'])} ({r['sources']['matched']}/{r['sources']['total']}) | "
            f"{yesno(r['citations']['present'])} ({r['citations']['total']}) | "
            f"{yesno(r['citations']['present'] and r['citations']['matched'] == r['citations']['total'])} "
            f"({r['citations']['matched']}/{r['citations']['total']}) | {g} |"
        )

    lines += [
        "",
        "## Режим «не знаю» (слабый контекст)",
        "",
        "| # | Вопрос | max_score | abstain | «не знаю»+уточнение |",
        "|---|---|---|---|---|",
    ]
    for r in weak:
        lines.append(
            f"| {r['id']} | {r['question']} | {r['max_score']} | "
            f"{yesno(r['abstained'])} | {yesno(bool(r['abstain_ok']))} |"
        )

    lines += ["", "## Детали по контрольным вопросам", ""]
    for r in domain:
        grounded = r["grounded"]
        lines.append(f"### {r['id']}. {r['question']}")
        lines.append("")
        lines.append(f"**Ответ:**")
        lines.append("")
        lines.append(f"> {r['answer_block'].strip()[:600]}")
        lines.append("")
        lines.append(f"**Источники** (matched {r['sources']['matched']}/{r['sources']['total']}):")
        for s in r["sources"]["items"]:
            lines.append(f"- `{s['chunk_id']}` section=`{s['section']}` "
                         f"{'✅' if s['matched'] else '❌'}")
        lines.append("")
        lines.append(f"**Цитаты** (подстроки {r['citations']['matched']}/{r['citations']['total']}):")
        for c in r["citations"]["items"]:
            lines.append(f"- [{c['n']}] «{c['quote'][:200]}» "
                         f"{'✅' if c['matched'] else '❌'}")
        if grounded:
            lines.append("")
            lines.append(f"**Смысл = цитаты:** {grounded['score']}/2 — {grounded['rationale']}")
        lines.append("")

    lines += ["", "## Детали по слабым вопросам", ""]
    for r in weak:
        lines.append(f"### {r['id']}. {r['question']} (max_score {r['max_score']})")
        lines.append("")
        lines.append(f"> {r['answer'].strip()[:300]}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка источников и цитат RAG")
    parser.add_argument("--strategy", "-s", default="structural", choices=["fixed", "structural"])
    parser.add_argument("--top-k", type=int, default=config.TOP_K_FINAL)
    parser.add_argument("--mode", default="baseline", choices=config.MODES)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--no-judge", action="store_true", help="не запускать LLM-judge")
    args = parser.parse_args(argv)

    questions = json.loads(GOLDEN.read_text(encoding="utf-8"))
    weak = json.loads(WEAK.read_text(encoding="utf-8")) if WEAK.exists() else []
    results = run(questions, weak, args.strategy, args.top_k, args.mode,
                  args.min_score, do_judge=not args.no_judge)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "citations_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "citations_report.md").write_text(
        build_markdown(results), encoding="utf-8"
    )
    print(f"\nОтчёт: {REPORT_DIR / 'citations_report.md'}")
    print(f"Данные: {REPORT_DIR / 'citations_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
