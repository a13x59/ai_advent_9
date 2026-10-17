#!/usr/bin/env bash
#
# Останавливает сервисы, запущенные через run-local.sh.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT/.run"

stop_one() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"

  if [ ! -f "$pidfile" ]; then
    echo "$name: не запущен (PID-файл отсутствует)."
    return 0
  fi

  local pid
  pid="$(cat "$pidfile")"

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$name: процесс $pid уже остановлен."
    rm -f "$pidfile"
    return 0
  fi

  echo "Останавливаю $name (PID $pid)..."
  kill "$pid" 2>/dev/null || true

  local i
  for i in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.3
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "$name: не завершился, завершаю принудительно."
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}

# Сначала приложение (оно зависит от воркера), затем воркер.
stop_one app
stop_one worker

echo "Сервисы остановлены. Логи: $RUN_DIR/app.log, $RUN_DIR/worker.log"
