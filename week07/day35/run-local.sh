#!/usr/bin/env bash
#
# Локальный запуск сервиса (без Docker): Python-воркер + Kotlin-приложение.
# Для продакшена на VPS используйте deploy.sh (Docker).
#
# Использование:
#   ./run-local.sh     # запустить (создаёт venv/собирает при первом запуске)
#   ./stop-local.sh    # остановить
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT/.run"
mkdir -p "$RUN_DIR"

WORKER_PORT="${WORKER_PORT:-8000}"
APP_PORT="${APP_PORT:-8080}"

is_running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

# Ждёт, пока HTTP-эндпоинт ответит; при неудаче печатает хвост лога.
wait_http() {
  local url="$1" name="$2" log="$3" tries="${4:-40}"
  local i
  for i in $(seq 1 "$tries"); do
    if curl -s --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Ошибка: $name не ответил на $url за ${tries} с." >&2
  if [ -f "$log" ]; then
    echo "Последние строки лога ($log):" >&2
    tail -n 15 "$log" >&2 || true
  fi
  return 1
}

# --- проверка зависимостей ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "Ошибка: python3 не найден" >&2
  exit 1
fi
if ! command -v java >/dev/null 2>&1; then
  echo "Ошибка: java не найден (нужен JDK 17+)" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Ошибка: curl не найден" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Внимание: ffmpeg не найден — MP3/M4A не поддерживаются (только WAV)."
fi

# --- воркер ---
if is_running "$RUN_DIR/worker.pid"; then
  echo "Воркер уже запущен (PID $(cat "$RUN_DIR/worker.pid"))."
else
  if [ ! -d "$ROOT/worker/.venv" ]; then
    echo "==> Создание venv и установка зависимостей воркера (первый запуск может быть долгим)..."
    python3 -m venv "$ROOT/worker/.venv"
    "$ROOT/worker/.venv/bin/pip" install --quiet --upgrade pip
    "$ROOT/worker/.venv/bin/pip" install --quiet -r "$ROOT/worker/requirements.txt"
  fi
  echo "==> Запуск воркера (127.0.0.1:${WORKER_PORT})"
  (
    cd "$ROOT/worker"
    PYTHONDONTWRITEBYTECODE=1 nohup ./.venv/bin/uvicorn main:app --host 127.0.0.1 --port "$WORKER_PORT" \
      > "$RUN_DIR/worker.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/worker.pid"
  )
  echo "    Ожидание готовности воркера (загрузка модели)…"
  if ! wait_http "http://127.0.0.1:${WORKER_PORT}/health" "воркер" "$RUN_DIR/worker.log" 90; then
    exit 1
  fi
  echo "Воркер готов."
fi

# --- приложение ---
if [ ! -x "$ROOT/app/build/install/guitar-tab-service/bin/guitar-tab-service" ]; then
  echo "==> Сборка приложения (первый запуск может быть долгим)..."
  (cd "$ROOT/app" && ./gradlew installDist --no-daemon)
fi

if is_running "$RUN_DIR/app.pid"; then
  echo "Приложение уже запущено (PID $(cat "$RUN_DIR/app.pid"))."
else
  echo "==> Запуск приложения (127.0.0.1:${APP_PORT})"
  (
    cd "$ROOT/app/build/install/guitar-tab-service"
    PORT="$APP_PORT" WORKER_URL="http://127.0.0.1:${WORKER_PORT}" \
      nohup ./bin/guitar-tab-service \
      > "$RUN_DIR/app.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/app.pid"
  )
  echo "    Ожидание готовности приложения…"
  if ! wait_http "http://127.0.0.1:${APP_PORT}/api/health" "приложение" "$RUN_DIR/app.log" 20; then
    exit 1
  fi
  echo "Приложение готово."
fi

echo ""
echo "Готово:"
echo "  Веб-интерфейс: http://localhost:${APP_PORT}"
echo "  API:           http://localhost:${APP_PORT}/api/transcribe"
echo "  Воркер:        http://localhost:${WORKER_PORT}/health"
echo "  Остановить:    ./stop-local.sh"
echo "  Логи:          $RUN_DIR/worker.log, $RUN_DIR/app.log"
