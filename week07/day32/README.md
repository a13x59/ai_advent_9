# День 32. Автоматизация ревью кода

Автоматическое AI-ревью PR: пайплайн получает diff, использует RAG
(документация + код) и возвращает текст ревью по трём секциям — потенциальные
баги, архитектурные проблемы, рекомендации.

## Результат

- **Пайплайн (GitHub Action)** — `.github/workflows/pr-review.yml` (в корне
  репозитория) запускается на `pull_request`, получает diff через GitHub API,
  вызывает ревью и публикует комментарий в PR.
- **Движок ревью** — новый пакет `pr_review/`:
  - `review_pr.py` — CLI (headless/lean, подход B);
  - `engine.py` / `prompt.py` / `context.py` / `rag_client.py` / `providers.py`
    / `github.py` / `diff_utils.py`.
- **Подход A (через агента)** — эндпоинт `POST /review` в `agent/agent_core.py`,
  полный агентский пайплайн (провайдер агента + RagClient).
- **RAG (документация + код)** — отдельный индекс кода
  `rag/indexes/codebase/` + корпус документации `rag/indexes/`; поддержка
  `--corpus codebase` в `rag/src/pipeline.py`, `retrieval.py`, `server.py`.
- **Переключатели** — `REVIEW_BACKEND` (`script|agent`) и `REVIEW_PROVIDER`
  (`deepseek|ollama`), чтобы управлять обеими логиками и сравнивать их.
- **Пример + тесты** — `pr_review/samples/` (демо-файл с багами) и
  `pr_review/tests/` + `agent/test_review_endpoint.py`.

## Соответствие требованиям

| Требование | Где |
|---|---|
| пайплайн запускается на PR | `.github/workflows/pr-review.yml` |
| ассистент получает diff | `review_pr.py` (git / API / файл) + `diff_utils.py` |
| RAG (документация + код) | `rag_client.py` + корпуса `docs`/`codebase` |
| баги / архитектура / рекомендации | `prompt.py` (3 обязательные секции) |
| текст ревью | вывод в консоль, файл, комментарий в PR |

## Быстрый старт

```bash
cd week07/day32

# демо (dry-run, без LLM)
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff --dry-run

# полное ревью (нужен DEEPSEEK_API_KEY в .env)
python -m pr_review.review_pr --diff-file pr_review/samples/example_pr.diff \
    --provider deepseek --output /tmp/review.md
```

Подробности — в [`pr_review/README.md`](pr_review/README.md).

## Правки шебангов

Консольные скрипты венвов (`agent/.venv`, `rag/.venv`, `llm-optimization/.venv`)
ссылались на старые пути `week05/day23` / `week06/day29`; исправлены на
`week07/day32` (46 файлов: шебанги `bin/*` и `VIRTUAL_ENV` в `activate*`).
