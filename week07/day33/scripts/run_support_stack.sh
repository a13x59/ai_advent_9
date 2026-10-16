#!/usr/bin/env bash
# Запускает весь стек поддержки в ОДНОМ терминале:
#   RAG          http://127.0.0.1:8891  (ретрив по корпусу support)
#   support-mcp  http://127.0.0.1:8893  (тикеты/покупатели, SQLite)
#   агент        http://127.0.0.1:8002  (llama3.1, эндпоинт POST /support)
#
# Запускайте ЭТИМ скриптом из своего терминала, чтобы сервисы жили независимо
# от фоновых задач ассистента. Ctrl+C — остановить все три.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIDS=()
cleanup() {
    echo ""
    echo "Останавливаю сервисы…"
    for p in "${PIDS[@]:-}"; do
        kill "$p" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

echo "▶ RAG          http://127.0.0.1:8891"
(cd "$ROOT/rag" && exec ./.venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8891) &
PIDS+=($!)

echo "▶ support-mcp  http://127.0.0.1:8893"
(cd "$ROOT/support-mcp" && exec java -jar build/libs/support-mcp-all.jar) &
PIDS+=($!)

echo "▶ агент        http://127.0.0.1:8002  (llama3.1)"
(cd "$ROOT/agent" && exec ./.venv/bin/uvicorn local_main:app --host 0.0.0.0 --port 8002) &
PIDS+=($!)

echo ""
echo "Стек запущен. Откройте agent/support.html или выполните:"
echo "  python agent/support_demo.py http://127.0.0.1:8002 --model llama3.1"
echo "Ctrl+C — остановить все сервисы."
wait
