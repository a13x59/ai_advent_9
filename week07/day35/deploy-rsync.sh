#!/usr/bin/env bash
#
# Копирует файлы проекта на удалённый сервер через rsync.
# Исключает всё, что не нужно для деплоя: git, кэши сборки, venv,
# локальные секреты (.env), логи/PID (.run), IDE-файлы.
#
# Если целевой директории на сервере нет (или она не на запись),
# скрипт создаст её через sudo (запросит пароль) и отдаст владельцу.
#
# Использование:
#   ./deploy-rsync.sh              # загрузить на сервер
#   ./deploy-rsync.sh --dry-run    # показать, что будет скопировано (без изменений)
#
# После загрузки на сервере:
#   ssh -i ~/.ssh/ai-agent/ai-agent user@91.188.212.37 'cd /opt/tab-generator && ./deploy.sh'
#
set -euo pipefail

# --- настройки (поправьте при необходимости) ---
SSH_HOST="user@91.188.212.37"                  # пользователь@сервер
SSH_KEY="$HOME/.ssh/ai-agent/ai-agent"         # ключ (сертификат) для SSH
SSH_PORT=""                                    # например "2222", если SSH на нестандартном порту
REMOTE_DIR="/opt/tab-generator/"               # целевая директория на сервере

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR_CLEAN="${REMOTE_DIR%/}"
RSYNC_TARGET="${SSH_HOST}:${REMOTE_DIR}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "Ошибка: rsync не установлен." >&2
  exit 1
fi

SSH_OPTS="-i ${SSH_KEY}"
if [ -n "$SSH_PORT" ]; then
  SSH_OPTS="${SSH_OPTS} -p ${SSH_PORT}"
fi

DRY=""
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ]; then
  DRY="-n"
fi

# Создаём целевую директорию на сервере, если её ещё нет или она не на запись.
# test -w проверяет существование и доступность на запись; при первом запуске
# запросит пароль sudo (если sudo не беспарольный).
if [ -z "$DRY" ] && ! ssh $SSH_OPTS "$SSH_HOST" "test -d '$REMOTE_DIR_CLEAN' && test -w '$REMOTE_DIR_CLEAN'" 2>/dev/null; then
  echo "==> Создаю директорию ${REMOTE_DIR_CLEAN} на сервере (запросит пароль sudo)…"
  ssh -t $SSH_OPTS "$SSH_HOST" "sudo mkdir -p '$REMOTE_DIR_CLEAN' && sudo chown \$USER '$REMOTE_DIR_CLEAN'"
fi

# --delete держит сервер в синхроне с локальным деревом (удаляет лишнее).
# Исключённые из копирования файлы (.env и пр.) при этом НЕ удаляются.
# --no-owner --no-group, чтобы не падать на chown при разных именах пользователей.
rsync -avzh --no-owner --no-group --delete $DRY -e "ssh $SSH_OPTS" \
  --exclude='.git/' \
  --exclude='.gradle/' \
  --exclude='.kotlin/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='*.iml' \
  --exclude='build/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.class' \
  --exclude='.run/' \
  --exclude='.env' \
  --exclude='.DS_Store' \
  --exclude='examples/' \
  "$SRC_DIR/" "$RSYNC_TARGET"

echo ""
echo "Готово: файлы скопированы в ${RSYNC_TARGET}"
echo "Дальше на сервере:"
echo "  ssh -i ${SSH_KEY} ${SSH_HOST} 'cd ${REMOTE_DIR_CLEAN} && ./deploy.sh'"
