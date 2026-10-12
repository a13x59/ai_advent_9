#!/usr/bin/env bash
# deploy.sh — развёртывание приватного AI-сервиса (агент + Ollama) на VPS.
#
# Запуск от root:   sudo bash deploy/deploy.sh
# Требования: Ubuntu/Debian, >= 2 ГБ RAM (CPU-only), доступ в интернет.
#
# Что делает:
#   1) ставит пакеты, Ollama, nginx, ufw;
#   2) закрывает Ollama на loopback и ограничивает её ресурсы;
#   3) готовит .env (модель qwen2.5:1.5b, rate limit 5/min) и API-ключ;
#   4) качает модель, ставит venv и systemd-юнит агента (uvicorn 127.0.0.1:8002);
#   5) настраивает nginx (front + API с X-Api-Key) и firewall.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT_DIR="$PROJECT_ROOT/agent"

echo "==> Проект: $PROJECT_ROOT"

if [ "$(id -u)" -ne 0 ]; then
    echo "Запускайте от root: sudo bash deploy/deploy.sh" >&2
    exit 1
fi

# --- 1. Пакеты ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl python3 python3-venv python3-pip nginx ufw openssl

# --- 2. Ollama ---
if ! command -v ollama >/dev/null 2>&1; then
    echo "==> Установка Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# --- 3. Ollama: loopback + ограничение ресурсов ---
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=5m"
EOF
systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama

# --- 4. Конфиг агента (.env) ---
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$PROJECT_ROOT/.env"
fi
# API-ключ: генерируем, если не задан.
if ! grep -q '^AGENT_API_KEY=.' "$PROJECT_ROOT/.env"; then
    echo "AGENT_API_KEY=$(openssl rand -hex 24)" >> "$PROJECT_ROOT/.env"
fi
# Фиксируем целевые значения (компактная модель, маленький контекст, rate limit 5).
sed -i 's|^OLLAMA_MODEL=.*|OLLAMA_MODEL=qwen2.5:1.5b|' "$PROJECT_ROOT/.env"
sed -i 's|^OLLAMA_NUM_CTX=.*|OLLAMA_NUM_CTX=2048|' "$PROJECT_ROOT/.env"
sed -i 's|^AGENT_RATE_LIMIT_PER_MINUTE=.*|AGENT_RATE_LIMIT_PER_MINUTE=5|' "$PROJECT_ROOT/.env"
sed -i 's|^AGENT_MAX_PROMPT_TOKENS=.*|AGENT_MAX_PROMPT_TOKENS=1536|' "$PROJECT_ROOT/.env"

# --- 4.5 Конфиг фронта для VPS (same-origin, модель qwen) ---
cat > "$AGENT_DIR/config.js" <<'EOF'
// config.js — сгенерирован deploy.sh для VPS.
// Фронт раздаётся тем же nginx, поэтому base = '' (same-origin).
window.APP_CONFIG = {
    cloudBase: '',
    localBase: '',
    cloudModel: 'deepseek-chat',
    localModel: 'qwen2.5:1.5b',
    defaultBackend: 'local',
    apiKey: '',   // ключ вводится в интерфейсе (X-Api-Key)
};
EOF

MODEL="$(grep -E '^OLLAMA_MODEL=' "$PROJECT_ROOT/.env" | cut -d= -f2-)"
API_KEY="$(grep -E '^AGENT_API_KEY=' "$PROJECT_ROOT/.env" | cut -d= -f2-)"

# --- 5. Модель ---
echo "==> Загрузка модели Ollama: $MODEL"
ollama pull "$MODEL"

# --- 6. Python venv агента ---
python3 -m venv "$AGENT_DIR/.venv"
"$AGENT_DIR/.venv/bin/pip" install --upgrade pip
"$AGENT_DIR/.venv/bin/pip" install -r "$AGENT_DIR/requirements.txt"

# --- 7. systemd unit агента ---
sed "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" "$SCRIPT_DIR/agent.service" \
    > /etc/systemd/system/ai-agent.service
systemctl daemon-reload
systemctl enable ai-agent
systemctl restart ai-agent

# --- 8. nginx ---
sed -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__API_KEY__|$API_KEY|g" "$SCRIPT_DIR/nginx.conf" \
    > /etc/nginx/sites-available/ai-agent
ln -sf /etc/nginx/sites-available/ai-agent /etc/nginx/sites-enabled/ai-agent
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx

# --- 9. Firewall ---
ufw allow OpenSSH || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true
ufw deny 11434/tcp || true
ufw --force enable || true

echo ""
echo "============================================================"
echo " Готово."
echo " API key (X-Api-Key): $API_KEY"
echo "============================================================"
echo "Проверка (на самом VPS):"
echo "  curl http://127.0.0.1:8002/healthz"
echo "  curl http://127.0.0.1:8002/limits"
echo "Проверка извне (публичный IP):"
echo "  curl -H 'X-Api-Key: $API_KEY' http://<ПУБЛИЧНЫЙ_IP>/healthz"
echo "  curl -H 'X-Api-Key: $API_KEY' http://<ПУБЛИЧНЫЙ_IP>/limits"
echo "Фронт:  http://<ПУБЛИЧНЫЙ_IP>/   (в поле «API-ключ» введите ключ выше)"
echo "Нагрузочный тест:  python scripts/load_test.py --base http://<ПУБЛИЧНЫЙ_IP> --api-key $API_KEY"
