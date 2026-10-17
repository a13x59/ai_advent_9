#!/usr/bin/env bash
#
# Развёртывание сервиса на VPS (Ubuntu + Docker).
# Запускать из корня проекта (там, где лежит docker-compose.yml).
#
# Скрипт НЕ запрашивает DEEPSEEK_API_KEY — ключ задаётся в переменных окружения
# или в файле .env (см. .env.example), который вы заполняете сами.
#
set -euo pipefail

cd "$(dirname "$0")"

# Команда от root или через sudo
as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "==> [1/5] Проверка Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "    Docker не найден — устанавливаю…"
  curl -fsSL https://get.docker.com | as_root sh
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "    Плагин docker compose не найден. Проверьте установку Docker." >&2
  exit 1
fi
# Добавляем текущего пользователя в группу docker (действует после перелогина).
# Текущий запуск всё равно идёт через sudo (as_root), поэтому работает сразу.
as_root usermod -aG docker "$(id -un)" 2>/dev/null || true

echo "==> [2/5] Проверка .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    Создан .env из .env.example."
  echo "    Заполните DEEPSEEK_API_KEY в .env (или экспортируйте переменную) и перезапустите скрипт."
fi

echo "==> [3/5] Сборка образов"
as_root docker compose build

echo "==> [4/5] Запуск сервисов"
as_root docker compose up -d

echo "==> [5/5] Готово"
APP_PORT=$(grep -E '^APP_PORT=' .env 2>/dev/null | cut -d= -f2 || true)
APP_PORT=${APP_PORT:-8080}
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
IP=${IP:-127.0.0.1}

echo ""
echo "Сервис доступен по адресу: http://${IP}:${APP_PORT}"
echo "Проверка: curl http://localhost:${APP_PORT}/api/health"
echo ""
echo "Первый запуск может занять 1–2 минуты (загрузка образа и инициализация модели)."
echo "Логи: sudo docker compose logs -f worker   (после перелогина — без sudo)"
