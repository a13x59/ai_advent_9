# Приватный AI-сервис на базе локальной LLM (VPS)

Разворачивает на VPS приватный сервис: агент (FastAPI, порт `8002` внутри)
поверх **Ollama** с компактной моделью **qwen2.5:1.5b** (CPU-only, 2 ГБ RAM).
Публично доступен только агентский API через nginx (front + `/agent*` с `X-Api-Key`),
а Ollama остаётся на loopback.

## Архитектура

```
Интернет (80/443)
    │
    ▼
nginx (front + API, X-Api-Key)          ← единственная публичная граница
    │
    ▼
агент FastAPI (127.0.0.1:8002)          ← rate limit per-IP, /healthz, /limits
    │
    ▼
Ollama (127.0.0.1:11434, loopback)      ← наружу НЕ открывается
```

## Быстрый старт

1. Скопируйте проект на VPS (git clone / scp).
2. Запустите от root:

   ```bash
   sudo bash deploy/deploy.sh
   ```

3. Заберите из вывода **API key** и публичный IP/домен.

## Проверка

```bash
# здоровье и лимиты (с ключом)
curl -H 'X-Api-Key: KEY' http://<IP>/healthz
curl -H 'X-Api-Key: KEY' http://<IP>/limits

# чат (один запрос к /agent)
curl -s -H 'X-Api-Key: KEY' -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"привет"}]}' \
  http://<IP>/agent

# нагрузочный тест (стабильность + rate limit + max context)
python scripts/load_test.py --base http://<IP> --api-key KEY --n 8 --concurrency 3
```

Фронт: `http://<IP>/` — в поле «API-ключ» введите ключ и выберите бэкенд
«Локальная LLM (Ollama)».

## Ключевые параметры (через .env / deploy/.env.example)

| Переменная | Значение на VPS | Смысл |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | компактная модель (фолбэки: `llama3.2:1b`, `qwen2.5:0.5b`) |
| `OLLAMA_NUM_CTX` | `2048` | контекстное окно (KV-кэш) |
| `OLLAMA_MAX_CONCURRENCY` | `2` | одновременных вызовов модели |
| `AGENT_RATE_LIMIT_PER_MINUTE` | `5` | rate limit per-IP |
| `AGENT_MAX_PROMPT_TOKENS` | `1536` | бюджет промпта (защита от переполнения) |
| `AGENT_API_KEY` | авто | ключ `X-Api-Key` для nginx |

## HTTPS (рекомендуется для публичного VPS)

После деплоя получите сертификат Let's Encrypt:

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d your.domain.com   # или: certbot certonly --standalone -d your.domain.com
```

## Примечания

- **Почему 1 worker uvicorn:** rate limit держится в памяти процесса. Для
  нескольких воркеров нужен общий redis-бэкенд.
- **Почему Ollama на loopback:** у Ollama нет авторизации; наружу она не должна
  торчать. Весь доступ — только через агента с `X-Api-Key`.
- **RAG опционально:** если поднять `rag/src/server.py` на 8891, агент сможет
  отвечать по базе знаний (`RAG_BASE_URL=http://127.0.0.1:8891`). MCP-сервисы
  на VPS не разворачиваются.
- Службы: `ai-agent` (агент), `ollama` (модель), `nginx` (прокси).
