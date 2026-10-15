# MCP-сервисы и RAG

## MCP-клиенты (`agent/mcp_client.py`)

Синхронные клиенты MCP по протоколу **Streamable HTTP / JSON-RPC 2.0**
(эндпоинт `/mcp`, заголовок `Mcp-Session-Id`). Интерфейс — `list_tools()` /
`call_tool(name, args)`. Каждый клиент хранит `SERVER_NAME` для маршрутизации инструментов в цикле function calling
(`agent_core._run_tool_loop`).

| Клиент              | Сервер                | Адрес   | Инструменты                                                                           |
|---------------------|-----------------------|---------|---------------------------------------------------------------------------------------|
| `FortuneMcpClient`  | `fortune-telling-mcp` | `:8888` | `magic_8_ball`, `bibliomancy`                                                         |
| `CurrencyMcpClient` | `currency-mcp`        | `:8889` | `get_rate`, `get_summary`                                                             |
| `PipelineMcpClient` | `pipeline-mcp`        | `:8890` | `search`, `summarize`, `save_to_file`                                                 |
| `GitMcpClient`      | `git-mcp`             | `:8892` | `git_branch`, `list_files`, `git_diff`                                                |
| `SupportMcpClient`  | `support-mcp`         | `:8893` | `create_ticket`, `get_ticket`, `find_user`, `list_tickets_for_user`, `search_tickets` |

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

### support-mcp (ассистент поддержки)

Сервис поддержки (мини-CRM) на SQLite: таблицы `users` и `tickets`, инструменты
`create_ticket` (write) и read-инструменты (`get_ticket`, `find_user`,
`list_tickets_for_user`, `search_tickets`). Номер тикета — читаемый (`T-1000+id`).

Эндпоинт `POST /support` использует его детерминированно (не через function calling), как `/help` использует git-mcp:
резолвит/создаёт тикет и подставляет его контекст в системный блок.

### RAG-корпус `support`

Для поддержки заведён отдельный корпус `support` (`rag/data/support/` →
`rag/indexes/support/`): FAQ (`faq/*.md`) и документация продукта (`docs/*.md`). Он изолирован от `docs`/`codebase`,
чтобы ответы поддержки не смешивались с NLP-статьями и кодом проекта. Индекс собирается командой
`python -m src.pipeline run --corpus support`. Офтопик-гейт `/support` использует поле `below_relevance` из результата
ретрива.

## `/help` — как это работает

1. Фронт или пользователь отправляет сообщение, начинающееся с `/help`.
2. Бэкенд распознаёт команду и переводит пайплайн в режим help.
3. Принудительно выполняется RAG-ретрив по вопросу после `/help`
   (документация проекта).
4. Детерминированно вызываются git-инструменты (`git_branch`, `list_files`).
5. В контекст подставляется системный блок «ассистент разработчика» (документация + git-контекст), модель генерирует
   свободный ответ со списком источников. Ответ возвращается с полем `help_context`.
