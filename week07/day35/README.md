# Guitar Tab Service

Веб-сервис: принимает аудиофайл с гитарной партией и возвращает табулатуру в
стандартном строе шестиструнной гитары (EADGBE) + MIDI.

## Архитектура

```
браузер → app (Kotlin + Ktor) → worker (Python FastAPI + Basic Pitch)
              │                        └─ аудио.wav → MIDI
              ├─ ffmpeg (конвертация в WAV)
              ├─ парсинг MIDI (javax.sound.midi)
              ├─ раскладка по струнам EADGBE (Витерби)
              ├─ рендер ASCII-таба
              └─ гейт качества: детерминированные проверки + DeepSeek
```

- **`app/`** — Kotlin + Ktor. Приём файла, конвертация, вызов воркера, раскладка
  по струнам, табулатура, оценка качества, раздача фронтенда.
- **`worker/`** — Python FastAPI + [Basic Pitch](https://github.com/spotify/basic-pitch)
  (используется «как есть»). `POST /predict` принимает WAV-байты, возвращает MIDI.
- **`deploy.sh`** — развёртывание на VPS (Ubuntu + Docker).
- **`docker-compose.yml`** — два сервиса: `app` и `worker`.

## Запуск

### Локально (без Docker)

Проще всего через скрипты:

```bash
./run-local.sh    # запустить (воркер + приложение)
./stop-local.sh   # остановить
```

Требуется: JDK 17+ (рекомендуется 21), Python 3.9–3.11, ffmpeg (нужен только для MP3/M4A; для WAV не обязателен).
Открыть http://localhost:8080.

### Продакшен (VPS, Ubuntu)

Два шага: загрузить файлы (rsync) и запустить на сервере.

```bash
# 1) локально — загрузить файлы (исключая git/кэши/venv/.env)
./deploy-rsync.sh

# 2) на сервере — собрать и запустить
ssh user@91.188.212.37 'cd /opt/tab-generator && ./deploy.sh'
```

На сервере `deploy.sh` сам создаст `.env` из `.env.example` — заполните там
`DEEPSEEK_API_KEY`. Альтернатива rsync — `git clone <репозиторий>` в ту же папку.

Сервис будет доступен на `http://<IP_VPS>:8080`.

## API

### `POST /api/transcribe`

Тело запроса — сырые байты аудиофайла (WAV/MP3/M4A), заголовок `Content-Type` —
MIME-тип файла. Пример:

```bash
curl --data-binary @solo.wav -H "Content-Type: audio/wav" \
     http://localhost:8080/api/transcribe
```

Ответ (200):

```json
{
  "tab": "e| 0-- 3-- 5--|\nB|-----------|\n…",
  "midiBase64": "TVRoZAAAAAYAAQ…",
  "quality": "ok",
  "qualityReason": null,
  "meta": { "durationSec": 12.4, "noteCount": 42, "tuning": "EADGBE" }
}
```

- `quality` — `ok` или `warning`.
- Если DeepSeek-гейт выносит вердикт «не OK», результат всё равно отдаётся, но
  `quality: "warning"` и `qualityReason` содержат причину.

Ошибки: `400` (неподдерживаемый формат), `502` (воркер недоступен), `500`.

### `GET /api/health`

Возвращает `OK`.

## Переменные окружения

| Переменная          | По умолчанию          | Описание |
|---------------------|-----------------------|----------|
| `PORT`              | `8080`                | Порт приложения |
| `WORKER_URL`        | `http://localhost:8000` | Адрес Python-воркера |
| `DEEPSEEK_API_KEY`  | *(пусто)*             | Ключ DeepSeek (заполняете сами) |
| `DEEPSEEK_MODEL`    | `deepseek-chat`       | Модель DeepSeek |
| `ENABLE_LLM_GATE`   | `true`                | Включить LLM-гейт качества |
| `FFMPEG_PATH`       | `ffmpeg`              | Путь к ffmpeg |

## Гейт качества

1. Детерминированные проверки: непустой результат, ноты в диапазоне гитары
   (лады 0–24, строй EADGBE).
2. Если включён `ENABLE_LLM_GATE` и задан ключ — DeepSeek оценивает
   правдоподобность/играбельность табулатуры (JSON-вердикт `ok`/`reason`).

При сбое LLM — fail-open (результат отдаётся по детерминированным проверкам).

## Ограничения (вне скоупа MVP)

- Точная нотация бэндов/слайдов/вибрато (питч-бенды игнорируются).
- Строгая ритмическая нотация (таб по ладам точный, по таймингу — приблизительный).
- Плотные аккорды, дисторшн, выделение гитары из полного микса.
- Гейт DeepSeek проверяет *правдоподобие*, а не точность нот относительно аудио.

## Структура

```
app/                  # Kotlin + Ktor
  src/main/kotlin/com/guitartab/
    Application.kt    # точка входа, DI, роутинг, статика
    TranscriptionService.kt
    routes/Routes.kt
    audio/            # определение формата, ffmpeg
    worker/           # HTTP-клиент к Basic Pitch
    midi/             # парсер MIDI
    guitar/           # строй, раскладка по струнам, рендер таба
    quality/          # детерминированные проверки + DeepSeek
  src/main/resources/static/index.html
worker/               # Python FastAPI + Basic Pitch
docker-compose.yml
deploy.sh
.env.example
```
