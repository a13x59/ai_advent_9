# pr_review — автоматическое AI-ревью кода (День 32)

Пайплайн, который на каждый PR получает diff, собирает контекст через RAG (документация + код) и возвращает текст ревью
по трём секциям:

- **Потенциальные баги**
- **Архитектурные проблемы**
- **Рекомендации**

## Два подхода + переключатели

| Флаг                             | Значения               | Что делает                                                                       |
|----------------------------------|------------------------|----------------------------------------------------------------------------------|
| `--backend` / `REVIEW_BACKEND`   | `script` (по умолч.)   | headless/lean: движок `engine.py` напрямую, без FastAPI                          |
|                                  | `agent`                | POST в работающий агент (`agent/` endpoint `/review`), полный агентский пайплайн |
| `--provider` / `REVIEW_PROVIDER` | `deepseek` (по умолч.) | ревью через облачный DeepSeek                                                    |
|                                  | `ollama`               | ревью через локальную модель (llama3.1 на `OLLAMA_BASE_URL`)                     |

Оба подхода используют один и тот же промпт и сборщик контекста (`prompt.py`,
`context.py`, `rag_client.py`), поэтому их результаты сопоставимы — это и есть «переключатель логики» для последующего
сравнения.

## Структура

```
pr_review/
├── review_pr.py     # CLI (подход B) + переключатель --backend/--provider
├── engine.py        # оркестратор: diff → контекст → промпт → LLM → результат
├── prompt.py        # системный промпт и контракт вывода (3 секции)
├── context.py       # детерминированный контекст: полные файлы + импорты + docs
├── rag_client.py    # векторный RAG: HTTP к rag-сервису (корпуса docs/codebase)
├── diff_utils.py    # разбор diff, фильтрация сгенерированных файлов, нарезка
├── providers.py     # DeepSeek + Ollama (HTTP), переключатель по имени
├── github.py        # публикация комментария в PR (GitHub API)
├── tests/           # unit-тесты (без сети)
└── samples/         # example_service.py + example_pr.diff (демо с багами)
```

## RAG: документация + код (отдельный индекс)

Корпус кода проекта лежит в **отдельном** индексе `rag/indexes/codebase/`
(симлинки на исходники — `rag/data/codebase/code/`), документация — в
`rag/indexes/` (корпус `docs`). Сборка:

```bash
cd rag && ./.venv/bin/python -m src.pipeline run --corpus codebase
```

Симлинки синхронизируются **автоматически** перед сборкой: `run --corpus codebase`
(и `list --corpus codebase`) вызывает `rag/src/codebase.py::sync_codebase`, которая добавляет ссылки на новые файлы и
удаляет устаревшие/битые. Вручную можно прогнать `python rag/scripts/sync_codebase.py` (обёртка над той же логикой).

Ревью собирает контекст в 4 уровня:

1. полный текущий текст изменённых файлов;
2. локально импортируемые модули (связанный код);
3. документация, релевантная изменённым папкам;
4. векторный RAG по корпусам `codebase` + `docs` (если поднят rag-сервис).

Уровни 1–3 работают всегда (даже в CI без torch/faiss); уровень 4 включается, когда запущен rag-сервис:
`cd rag && uvicorn src.server:app --port 8891`.

## Демонстрация (пошаговый сценарий)

```bash
cd week07/day32
python -m pr_review.demo                 # без ключа (режим mock, детерминированная эвристика)
python -m pr_review.demo --provider deepseek   # реальное AI-ревью
python -m pr_review.demo --provider ollama     # на локальной модели
python -m pr_review.demo --show-prompt         # показать собранный промпт
```

Сценарий проходит шаги: diff PR → изменённые файлы → контекст (файлы+импорты+docs)
→ векторный RAG (docs+codebase) → промпт → ревью (3 секции) → публикация в PR.

## Локальный запуск

```bash
cd week07/day32

# демо-ревью примера с багами (нужен DEEPSEEK_API_KEY в .env)
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff \
    --provider deepseek --output /tmp/review.md

# тот же пример на локальной модели
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff \
    --provider ollama

# подход A — через работающего агента (порт 8000)
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff \
    --backend agent

# dry-run (без вызова LLM)
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff --dry-run
```

## GitHub Action

`.github/workflows/pr-review.yml` (в корне репозитория) запускается на
`pull_request` (opened/synchronize/reopened): скачивает diff через GitHub API, вызывает `review_pr.py` и публикует ревью
комментарием в PR. Переключатели задаются как Repository variables: `REVIEW_BACKEND`, `REVIEW_PROVIDER`.

## Тесты

```bash
cd agent
./.venv/bin/python -m pytest test_review_endpoint.py ../pr_review/tests/test_review.py -q
```
