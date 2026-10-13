# MCP-сервисы и RAG

## MCP-клиенты (`agent/mcp_client.py`)

Синхронные клиенты MCP по протоколу **Streamable HTTP / JSON-RPC 2.0**
(эндпоинт `/mcp`, заголовок `Mcp-Session-Id`). Интерфейс — `list_tools()` /
`call_tool(name, args)`. Каждый клиент хранит `SERVER_NAME` для маршрутизации инструментов в цикле function calling
(`agent_core._run_tool_loop`).

| Клиент              | Сервер                | Адрес   | Инструменты                            |
|---------------------|-----------------------|---------|----------------------------------------|
| `FortuneMcpClient`  | `fortune-telling-mcp` | `:8888` | `magic_8_ball`, `bibliomancy`          |
| `CurrencyMcpClient` | `currency-mcp`        | `:8889` | `get_rate`, `get_summary`              |
| `PipelineMcpClient` | `pipeline-mcp`        | `:8890` | `search`, `summarize`, `save_to_file`  |
| `GitMcpClient`      | `git-mcp`             | `:8892` | `git_branch`, `list_files`, `git_diff` |

Инструменты включаются флагом `enable_tools: true` и моделью, поддерживающей function calling (DeepSeek `deepseek-chat`
или локальные `llama*`/`qwen*`).

### git-mcp (ассистент разработчика)

Сервис для доступа к git-репозиторию проекта:

- `git_branch` — текущая ветка и список веток.
- `list_files` — список файлов проекта (`path`, `max_files`).
- `git_diff` — изменения рабочей копии (`staged`, `stat`, `path`).

Скоуп операций — каталог проекта `GIT_PROJECT_PATH` (по умолчанию рабочий каталог сервиса); команды выполняются через
`git -C <path> …` с pathspec `.`.

## RAG (`agent/rag_client.py`)

Агент не тянет тяжёлые зависимости (faiss / sentence-transformers): ретрив делает отдельный сервис `rag/src/server.py`
(порт `:8891`), клиент ходит к нему по HTTP (`POST /retrieve`).

- `retrieve(query, strategy, top_k, mode, top_k_candidates, min_score)` →
  `{chunks, mode, candidates, kept, dropped, rewritten_query, max_score,
  below_relevance, ...}`.
- Режимы: `baseline | rewrite | filter | rewrite+filter | rerank`.
- Стратегии chunking: `fixed` (по размеру) и `structural` (по заголовкам).

Документация проекта подключена в RAG через симлинки в `rag/data/raw/docs/`
(README агента + `agent/docs/*.md`). Индекс пересобирается командой
`python -m src.pipeline run` из папки `rag/`.

## `/help` — как это работает

1. Фронт или пользователь отправляет сообщение, начинающееся с `/help`.
2. Бэкенд распознаёт команду и переводит пайплайн в режим help.
3. Принудительно выполняется RAG-ретрив по вопросу после `/help`
   (документация проекта).
4. Детерминированно вызываются git-инструменты (`git_branch`, `list_files`).
5. В контекст подставляется системный блок «ассистент разработчика» (документация + git-контекст), модель генерирует
   свободный ответ со списком источников. Ответ возвращается с полем `help_context`.
