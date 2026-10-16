# Архитектура агента

## Общая схема

Агент — FastAPI-приложение, собранное фабрикой `create_app(...)` в
`agent_core.py`. Вся логика обработки запроса живёт в одном модуле
`agent_core.py` и не зависит от конкретной модели: точки взаимодействия с LLM вынесены за интерфейс `AgentProvider`.

```
index.html (SPA)
   │  POST /agent (JSON)
   ▼
create_app(provider, store, mcp_clients, rag_retriever)
   │
   ├─ RateLimitMiddleware (per-IP, скользящее окно)
   ├─ CORS
   └─ /agent — пайплайн обработки (см. ниже)
```

## Провайдеры модели (`AgentProvider`)

Абстрактный класс с тремя методами:

- `complete(messages, request, task, user_text, tools=None)` — основная генерация; возвращает DeepSeek-совместимый
  `dict` (`choices[0].message.content`,
  `tool_calls`, `usage`).
- `suggest_memory(working, long_term, user_message, model)` — предложения памяти.
- `check_invariants(invariants, user_text, model)` — вердикт нарушения инварианта.

Реализации:

- `DeepSeekProvider` (`main.py`) — HTTP к `https://api.deepseek.com/v1/chat/completions`.
- `LocalOllamaProvider` (`local_provider.py`) — HTTP к Ollama (`http://127.0.0.1:11434`); без инструментов использует
  нативный `/api/chat` (поддержка `num_ctx`, `top_k`), с инструментами — OpenAI-совместимый `/v1/chat/completions`
  (function calling). Ограничивает конкурентность семафором `OLLAMA_MAX_CONCURRENCY`.
- `MockProvider` (`mock_agent.py`) — детерминированный провайдер для тестов.

## Пайплайн обработки запроса (`/agent`)

1. **Ввод** — последнее сообщение пользователя из `request.messages`.
2. **Профиль** — явный `profile_id` → привязанный к сессии → активный.
3. **Сессия** — создаётся при первом обращении; хранит `strategy`, `window_size`.
4. **Память** — явные `memory_ops` + загрузка рабочей/постоянной памяти.
5. **Классификация** — задача (`_detect_task` по глаголам), продолжение, подтверждение плана/валидации, вопрос, обычный
   чат, команда `/help`.
6. **Инварианты** — `provider.check_invariants(...)`; при нарушении — отказ.
7. **Задача (FSM)** — при новой задаче создаётся оболочка `planning`; прогресс модель сообщает маркером `[STATE] {...}`,
   который валидируется
   `task_state.validate_and_apply` (см. `data-model.md`).
8. **Контекст** — `build_request_messages(...)`: системные блоки (инварианты, профиль, задача, память) + окно диалога по
   стратегии.
9. **RAG** — при `rag: true` (или `/help`) релевантные чанки вставляются в контекст.
10. **Бюджет контекста** — `_trim_messages_to_budget` обрезает старый диалог до
    `AGENT_MAX_PROMPT_TOKENS`; при переполнении — HTTP 413.
11. **Генерация** — обычный чат / цикл MCP-инструментов / ход задачи / отказ.
12. **Сохранение** — ответ в историю (сессии или задачи), обновление памяти.

## Управление контекстом (три стратегии)

- `sliding_window` — окно последних `N` сообщений.
- `sticky_facts` — факты из постоянной памяти + окно `N`.
- `branching` — ветки диалога: checkpoint делит историю, можно переключаться.

## Команда `/help`

Сообщение, начинающееся с `/help`, переключает пайплайн в режим «ассистент разработчика»: принудительный RAG-ретрив по
документации проекта, сбор git-контекста через git-MCP (ветка, список файлов, diff) и свободный ответ со списком
источников. Подробности — в `mcp-and-rag.md`.
