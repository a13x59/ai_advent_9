#!/usr/bin/env python3
"""Демо AI-ассистента поддержки пользователей (День 33).

Прогоняет четыре сценария против работающего агента (эндпоинт POST /support) и
наглядно показывает: создание тикета, офтопик-гейт, продолжение диалога в
контексте тикета и подгрузку тикета по номеру.

Требует запущенные сервисы:
    - support-mcp   (http://localhost:8893)  java -jar support-mcp-all.jar
    - RAG           (http://localhost:8891)  uvicorn src.server:app --port 8891
    - агент         (http://localhost:8002)  uvicorn local_main:app --port 8002

Использование:
    python support_demo.py [BASE_URL] [--model MODEL]
    # пример:
    python support_demo.py http://127.0.0.1:8003 --model llama3.1
"""
import json
import sys
import time

import requests

DEFAULT_BASE = "http://127.0.0.1:8002"
DEFAULT_MODEL = "llama3.1"

SCENARIOS = [
    ("1. Создание тикета (вопрос о продукте)", "demo-1",
     "/support Почему не работает авторизация?"),
    ("2. Офтопик (не по теме домена)", "demo-2",
     "/support Кто был 44 президентом США?"),
    ("3. Продолжение диалога (та же сессия)", "demo-1",
     "А если письмо с подтверждением не пришло?"),
    ("4. Подгрузка тикета по номеру", "demo-3",
     "/support --ticket_id=T-1001 Что делать с подтверждением аккаунта?"),
]


def call_support(base: str, model: str, session_id: str, message: str) -> dict:
    resp = requests.post(
        base.rstrip("/") + "/support",
        json={"session_id": session_id, "message": message, "model": model},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    args = [a for a in sys.argv[1:]]
    base = DEFAULT_BASE
    model = DEFAULT_MODEL
    if args and not args[0].startswith("--"):
        base = args.pop(0)
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            model = args[i + 1]

    print("=" * 72)
    print("  AI-ассистент поддержки «ВелоДом» — демонстрация")
    print(f"  агент: {base} | модель: {model}")
    print("=" * 72)

    for name, session, message in SCENARIOS:
        print(f"\n── {name} ──────────────────────────────")
        print(f"  👤 {message}")
        t0 = time.time()
        try:
            r = call_support(base, model, session, message)
        except Exception as e:  # noqa: BLE001 — показать ошибку подключения
            print(f"  ⚠ ошибка: {e}")
            print("  (проверьте, что support-mcp / RAG / агент запущены)")
            continue
        ticket = r.get("ticket_number")
        if ticket:
            tag = "создан" if r.get("ticket_created") else "контекст"
            print(f"  🎫 тикет №{ticket} ({tag})")
        print(f"  🤖 {r.get('response', '')}")
        if r.get("abstained"):
            print("  ↳ (офтопик-гейт: тикет не создавался)")
        if r.get("sources"):
            srcs = ", ".join(s["source"] for s in r["sources"][:3])
            print(f"  📎 источники: {srcs}")
        print(f"  ⏱ {r.get('duration', 0):.2f}s")

    print("\n" + "=" * 72)
    print("Готово. Откройте agent/support.html для интерактивной версии.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
