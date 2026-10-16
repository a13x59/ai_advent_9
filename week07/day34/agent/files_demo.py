#!/usr/bin/env python3
"""Демо файлового ассистента (День 34).

Прогоняет четыре сценария против работающего агента (эндпоинт POST /files) и
показывает: поиск использования API, проверку файлов по инварианту, генерацию
нового файла (ADR) и применение изменений с diff.

Сценарии задаются на уровне ЦЕЛИ — ассистент сам решает, какие файлы читать,
искать и менять (через files-mcp: list_dir / read_file / search_files /
write_file / edit_file).

Требует запущенные сервисы:
    - files-mcp   (http://localhost:8894)  FILES_ROOT=.. ./gradlew run  (из files-mcp/)
    - агент       (http://localhost:8002)  uvicorn local_main:app --port 8002

Использование:
    python files_demo.py [BASE_URL] [--model MODEL] [--apply]
    # пример:
    python files_demo.py http://127.0.0.1:8002 --model llama3.1
"""
import json
import sys
import time

import requests

DEFAULT_BASE = "http://127.0.0.1:8002"
DEFAULT_MODEL = "llama3.1"

# (название, session_id, apply, цель)
SCENARIOS = [
    (
        "1. Поиск использования компонента/API (dry-run)",
        "demo-1",
        False,
        "Найди все места в каталоге agent, где используется FilesMcpClient, "
        "и перечисли их с номерами строк.",
    ),
    (
        "2. Проверка файлов по инварианту (dry-run)",
        "demo-2",
        False,
        "Проверь, что каждый Python-файл в каталоге agent начинается с комментария "
        "или docstring. Перечисли файлы, которые этому правилу не соответствуют.",
    ),
    (
        "3. Генерация ADR — предпросмотр diff (dry-run)",
        "demo-3",
        False,
        "Сгенерируй файл ADR-034-file-assistant.md в корне проекта: архитектурное "
        "решение о модели dry-run/apply файлового ассистента (сервис files-mcp, "
        "инструменты read_file/search_files/write_file/edit_file).",
    ),
    (
        "4. Генерация ADR — применить на диск (apply)",
        "demo-3",
        True,
        "Сгенерируй файл ADR-034-file-assistant.md в корне проекта: архитектурное "
        "решение о модели dry-run/apply файлового ассистента (сервис files-mcp).",
    ),
]


def call_files(base: str, model: str, session_id: str, message: str, apply: bool) -> dict:
    resp = requests.post(
        base.rstrip("/") + "/files",
        json={"session_id": session_id, "message": message, "model": model, "apply": apply},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _short(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n… (обрезано)"


def main() -> int:
    args = [a for a in sys.argv[1:]]
    base = DEFAULT_BASE
    model = DEFAULT_MODEL
    if args and not args[0].startswith("--"):
        base = args.pop(0)
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model = args[i + 1]

    print("=" * 74)
    print("  Файловый ассистент — демонстрация (цель → разведка → diff/apply)")
    print(f"  агент: {base} | модель: {model}")
    print("=" * 74)

    for name, session, apply, message in SCENARIOS:
        print(f"\n── {name} ──────────────────────────────")
        print(f"  👤 {message}")
        t0 = time.time()
        try:
            r = call_files(base, model, session, message, apply)
        except Exception as e:  # noqa: BLE001 — показать ошибку подключения
            print(f"  ⚠ ошибка: {e}")
            print("  (проверьте, что files-mcp / агент запущены)")
            continue

        changed = r.get("changed_files") or []
        if changed:
            tag = "СОХРАНЕНО" if r.get("applied") else "предпросмотр"
            print(f"  📁 файлы ({tag}): {', '.join(changed)}")
        print(f"  🤖 {_short(r.get('response', ''))}")

        calls = r.get("tool_calls") or []
        if calls:
            tools = ", ".join(
                f"{c['name']}({(c.get('arguments') or {}).get('path') or (c.get('arguments') or {}).get('query') or ''})"
                for c in calls
            )
            print(f"  🔧 инструменты: {tools}")

        diff = r.get("diff") or ""
        if diff and diff != "No pending changes.":
            print("  ── diff ───────────────────────────────")
            print("  " + _short(diff, 900).replace("\n", "\n  "))

        print(f"  ⏱ {r.get('duration', 0):.2f}s")

    print("\n" + "=" * 74)
    print("Готово. Откройте agent/files.html для интерактивной версии.")
    print("Для apply-сценария (№4) файл ADR-034-file-assistant.md записан в корень проекта.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
