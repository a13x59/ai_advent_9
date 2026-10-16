# День 33. Ассистент поддержки пользователей

AI-ассистент поддержки интернет-магазина велосипедов «ВелоДом»: отвечает на вопросы о продукте по базе знаний
(RAG), учитывает контекст тикета и покупателя (MCP) и детерминированно отказывается от офтопика.

## Результат

- **Эндпоинт `POST /support`** — в `agent/agent_core.py`. Парсит команду
  (`/support <вопрос>` / `/support --ticket_id=<N> <вопрос>` / продолжение диалога), делает офтопик-гейт через RAG,
  резолвит/создаёт тикет через MCP и отвечает «человеческим» языком с учётом тикета.
- **MCP-сервис `support-mcp`** (Kotlin + SQLite, порт 8893) — мини-CRM: таблицы `users` и `tickets`, инструменты
  `create_ticket` (write) и `get_ticket` / `find_user` / `list_tickets_for_user` / `search_tickets` (read).
- **RAG-корпус `support`** — FAQ + документация продукта (`rag/data/support/` → `rag/indexes/support/`), отдельный от
  `docs`/`codebase`; сборка `python -m src.pipeline run --corpus support`.
- **Интеграция в агента** — `SupportMcpClient` (`agent/mcp_client.py`), регистрация в `main.py`/`local_main.py`,
  привязка «сессия → тикет» (`support_ticket_id` в `storage.py`).
- **Интерфейс** — `agent/support.html` (чат поддержки) + ссылка на главной `agent/index.html`.
- **Демо и тесты** — `agent/support_demo.py` (4 сценария в консоли), `agent/test_support_endpoint.py`,
  `support-mcp/src/test/.../SupportDbTest.kt`.

## Соответствие требованиям

| Требование                                   | Где                                                              |
|----------------------------------------------|------------------------------------------------------------------|
| отвечать на вопросы о продукте               | `POST /support` + персона в `build_support_system_block`          |
| использовать RAG (FAQ, документация)         | корпус `support` + `rag_retriever.retrieve(..., corpus="support")` |
| учитывать контекст пользователя/тикета       | `create_ticket`/`get_ticket` → контекст тикета и покупателя в промпте |
| через MCP: CRM или JSON с пользователями/тикетами | `support-mcp` (SQLite `users` + `tickets`)                    |
| пример «Почему не работает авторизация?»     | `rag/data/support/faq/avtorizaciya.md` + тикет в `support-mcp`   |

## Быстрый старт

```bash
cd week07/day33

# 1. RAG-индекс поддержки + сервис ретрива
cd rag && python -m src.pipeline run --corpus support
uvicorn src.server:app --host 0.0.0.0 --port 8891 &

# 2. support-mcp (тикеты/покупатели)
cd ../support-mcp && ./gradlew run &          # http://localhost:8893

# 3. агент (локальная llama3.1 или DeepSeek)
cd ../agent
uvicorn local_main:app --host 0.0.0.0 --port 8002   # локальная модель
# uvicorn main:app --host 0.0.0.0 --port 8000        # DeepSeek

# 4. демо в консоли
python support_demo.py http://127.0.0.1:8002 --model llama3.1

# или откройте agent/support.html в браузере
```

## Сценарии поддержки

| Сообщение                                   | Поведение                                                        |
|---------------------------------------------|------------------------------------------------------------------|
| `/support Почему не работает авторизация?`  | создаёт тикет, выводит номер, отвечает по FAQ                    |
| обычное сообщение (та же сессия)            | продолжает диалог в контексте привязанного тикета                |
| `/support --ticket_id=T-1001 <вопрос>`      | подгружает тикет N и отвечает в его контексте                    |
| `/support Кто был 44 президентом США?`      | «Нет информации по этому вопросу» (тикет не создаётся)           |

Подробности — в `agent/README.md`, `agent/docs/api.md`, `support-mcp/README.md`.
