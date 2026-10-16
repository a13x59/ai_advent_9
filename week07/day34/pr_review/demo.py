#!/usr/bin/env python3
"""Сценарий демонстрации Дня 32 «Автоматизация ревью кода».

Пошагово показывает, как пайплайн:
  1) получает diff «нового PR»;
  2) находит изменённые файлы;
  3) собирает контекст (полные файлы + связанный код + документация);
  4) делает векторный RAG по корпусам docs/codebase (если поднят rag-сервис);
  5) формирует промпт и генерирует ревью (3 секции);
  6) показывает, как ревью публикуется комментарием в PR.

Режимы:
  --provider mock     (по умолчанию) детерминированная эвристика, БЕЗ сети/ключа;
  --provider deepseek реальное ревью через DeepSeek (нужен DEEPSEEK_API_KEY);
  --provider ollama   реальное ревью через локальную модель (нужен Ollama).

Примеры:
  python -m pr_review.demo                          # демо на примере, без ключа
  python -m pr_review.demo --provider deepseek      # реальное AI-ревью
  python -m pr_review.demo --show-prompt            # показать собранный промпт
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pr_review import context as ctx  # noqa: E402
from pr_review import rag_client as rag  # noqa: E402
from pr_review.diff_utils import changed_files, estimate_tokens  # noqa: E402
from pr_review.prompt import REVIEW_SYSTEM, build_review_messages  # noqa: E402

SAMPLE_DIFF = ROOT / "pr_review" / "samples" / "example_pr.diff"


def _hrule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _section(n: int, title: str) -> None:
    print(f"\n--- Шаг {n}. {title} ---")


# --------------------------------------------------------------------------- #
# Детерминированный «ревьюер» для режима mock: находит типовые засеянные баги.
# --------------------------------------------------------------------------- #
def _heuristic_review(diff_text: str) -> str:
    added = "\n".join(ln[1:] for ln in diff_text.splitlines() if ln.startswith("+"))

    bugs: list[str] = []
    arch: list[str] = []
    reco: list[str] = []

    if re.search(r"execute\(\s*f[\"']", added) or re.search(r"execute\(\s*[\"'][^\"]*%s", added):
        bugs.append("- **SQL-инъекция**: имя пользователя подставляется в строку запроса "
                    "(`execute(f\"... WHERE username = '{username}'\")`). Использовать "
                    "параметризованный запрос `?`.")
        reco.append("- Перейти на параметризованные запросы (`?` / `:param`).")
    if re.search(r"except\s+Exception:", added):
        bugs.append("- **Проглатывание исключений**: `except Exception: return 0` скрывает "
                    "реальные ошибки БД. Логировать и различать «нет данных» и «ошибка».")
        reco.append("- Сузить `except`, логировать traceback, не маскировать ошибки нулём.")
    if re.search(r"for\b.*\n.*connect\(", added):
        arch.append("- **N+1 запросов**: внутри цикла открывается соединение и выполняется "
                    "запрос на каждую итерацию. Вынести `IN (...)` в один запрос или "
                    "переиспользовать одно соединение.")
        reco.append("- Переиспользовать соединение/пул и делать один запрос `WHERE user_id IN (...)`.")
    if re.search(r"fetchone\(\)\[\d+\]", added):
        arch.append("- **Нет обработки None**: `fetchone()[0]` упадёт при отсутствии строки. "
                    "Проверять результат перед обращением к индексу.")

    bugs = "\n".join(bugs) if bugs else "- не обнаружено"
    arch = "\n".join(arch) if arch else "- не обнаружено"
    reco = "\n".join(reco) if reco else "- не обнаружено"

    return (
        "## Потенциальные баги\n" + bugs + "\n\n"
        "## Архитектурные проблемы\n" + arch + "\n\n"
        "## Рекомендации\n" + reco + "\n\n"
        "> ⚠️ Режим mock: вывод сгенерирован детерминированной эвристикой, не LLM."
    )


def _show_context_blocks(blocks) -> None:
    if not blocks:
        print("(нет)")
        return
    for b in blocks:
        preview = b.content.replace("\n", " ")[:90]
        print(f"  • [{b.kind}] {b.title}  ({len(b.content)} симв.)  «{preview}…»")


def _show_rag(chunks) -> None:
    if not chunks:
        print("  Векторный RAG недоступен (rag-сервис не запущен) — пропущен.\n"
              "  Детерминированный контекст (код + документация) уже собран выше.")
        return
    for i, c in enumerate(chunks, 1):
        score = float(c.get("score") or 0.0)
        print(f"  [{i}] {c.get('corpus','')} :: {c.get('source','?')}  (score={score:.3f})")
        print("      " + (c.get("text") or "").strip()[:140].replace("\n", " "))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Демо AI-ревью (День 32)")
    parser.add_argument("--diff-file", default=str(SAMPLE_DIFF))
    parser.add_argument("--provider", choices=["mock", "deepseek", "ollama"], default="mock")
    parser.add_argument("--no-rag", action="store_true", help="отключить векторный RAG")
    parser.add_argument("--show-prompt", action="store_true", help="показать собранный промпт")
    args = parser.parse_args(argv)

    diff_text = Path(args.diff_file).read_text(encoding="utf-8")

    _hrule("ДЕМО: автоматическое AI-ревью кода (День 32)")
    print(f"Режим провайдера: {args.provider}")

    _section(1, "Получен diff «нового PR»")
    print(f"diff: {len(diff_text)} симв. (~{estimate_tokens(diff_text)} токенов)")
    print("```diff")
    print(diff_text[:900] + ("\n..." if len(diff_text) > 900 else ""))
    print("```")

    files = changed_files(diff_text)
    _section(2, "Изменённые файлы (после фильтрации сгенерированных)")
    for f in files:
        print(f"  • {f}")
    if not files:
        print("  (нет файлов для ревью)")
        return 0

    _section(3, "Контекст: полные файлы + связанный код + документация")
    blocks = ctx.build_deterministic_context(files)
    _show_context_blocks(blocks)
    context_text = ctx.render_context(blocks)

    _section(4, "RAG: семантический поиск по корпусам docs и codebase")
    rag_chunks: list[dict] = []
    rag_text = ""
    if not args.no_rag:
        rag_chunks = rag.retrieve_review_context(files, diff_text, top_k=4)[:8]
        _show_rag(rag_chunks)
        rag_text = rag.render_rag_context(rag_chunks)
    else:
        print("  отключён флагом --no-rag")

    _section(5, "Промпт для модели")
    messages = build_review_messages(diff_text, files, context_text, rag_text)
    print(f"  system-промпт: {len(REVIEW_SYSTEM)} симв. (3 обязательные секции)")
    print(f"  user-промпт: {len(messages[1]['content'])} симв. "
          f"(~{estimate_tokens(messages[1]['content'])} токенов)")
    if args.show_prompt:
        print("\n  --- SYSTEM ---\n" + REVIEW_SYSTEM[:800] + "\n  …")
        print("\n  --- USER (начало) ---\n" + messages[1]["content"][:800] + "\n  …")

    _section(6, "Ревью (потенциальные баги / архитектура / рекомендации)")
    if args.provider == "mock":
        review_text = _heuristic_review(diff_text)
        provider_label = "mock (эвристика)"
    else:
        try:
            from pr_review.engine import run_review
            result = run_review(diff_text, provider=args.provider,
                                use_rag=not args.no_rag)
            review_text = result.review_text
            provider_label = f"{args.provider} ({result.model})"
        except Exception as e:  # noqa: BLE001
            print(f"  Не удалось вызвать {args.provider}: {e}")
            print("  Совет: укажите DEEPSEEK_API_KEY или запустите Ollama, "
                  "либо используйте --provider mock.")
            return 1

    print(f"  Провайдер: {provider_label}")
    print("  " + review_text.replace("\n", "\n  "))

    _section(7, "Публикация в PR")
    print("  Локально: вывод выше печатается в консоль/файл.")
    print("  В GitHub Action: `.github/workflows/pr-review.yml` получает diff через")
    print("  GitHub API, запускает `review_pr.py --post-comment`, и ревью уходит")
    print("  комментарием в PR (секции те же).")

    _hrule("КОНЕЦ ДЕМО")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
