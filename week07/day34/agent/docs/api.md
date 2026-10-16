# API агента

## POST `/agent`

Обработка сообщения. Основное тело запроса — модель `AgentRequest`.

| Поле                   | Тип    | По умолчанию     | Назначение                                              |
|------------------------|--------|------------------|---------------------------------------------------------|
| `messages`             | list   | —                | История сообщений `{role, content}`                     |
| `session_id`           | str?   | —                | ID сессии (при первом обращении создаётся)              |
| `model`                | str    | —                | Название модели (`deepseek-chat`, `llama3.1`, …)        |
| `strategy`             | str    | `sliding_window` | `sliding_window` / `sticky_facts` / `branching`         |
| `window_size`          | int    | `10`             | Размер окна последних сообщений                         |
| `temperature`          | float  | `1.0`            | 0..2                                                    |
| `top_k`                | int    | `0`              | 0 = не задано                                           |
| `top_p`                | float  | `1.0`            | 0..1                                                    |
| `max_tokens`           | int    | `4096`           | 1..8192                                                 |
| `num_ctx`              | int?   | —                | Контекстное окно (локальные модели Ollama)              |
| `memory_ops`           | list   | `[]`             | Явные операции памяти `save/delete/move`                |
| `auto_suggest_memory`  | bool   | `false`          | Генерировать предложения памяти                         |
| `enable_tools`         | bool   | `false`          | Разрешить вызов MCP-инструментов                        |
| `rag`                  | bool   | `false`          | Режим RAG (дополнить запрос чанками)                    |
| `rag_mode`             | str    | `baseline`       | `baseline`/`rewrite`/`filter`/`rewrite+filter`/`rerank` |
| `rag_top_k`            | int    | `5`              | Топ-K чанков после второго этапа                        |
| `rag_top_k_candidates` | int?   | —                | Топ-K до фильтра/реранка                                |
| `rag_min_score`        | float? | —                | Порог отсечения (filter-режим)                          |
| `profile_id`           | str?   | —                | Профиль пользователя для запроса                        |
| `task_id`              | str?   | —                | Задача, к которой относится запрос                      |

Ответ (`/agent`) содержит: `response`, `session_id`, `usage`, `context`,
`memory`, `task`, `invariants`, `refused`, `tool_calls`, `rag_context`,
`sources`, `help_context` (для `/help`), `cost`, `duration`.

## Память

- `GET /agent/{session_id}/memory` — список слоёв памяти сессии.
- `POST /agent/{session_id}/memory` — явные операции `{ops: [...]}`.
- `POST /agent/{session_id}/memory/suggest` — предложения памяти от модели.
- `GET/POST /memory/longterm` — постоянная память (глобально).

## Профили

- `GET /profiles`, `GET /profiles/{id}`.
- `POST /profiles` — создать/обновить (`name` обязателен; `display_name`, `style`,
  `format`, `constraints`, `notes`; `is_active`).
- `POST /profiles/{id}/activate`, `DELETE /profiles/{id}`.

## Инварианты

- `GET /invariants`, `POST /invariants` (поля: `name`, `statement`, `category`,
  `rationale`, `deny_examples`, `is_active`, скоуп `profile_id`/`session_id`/`task_id`).
- `POST /invariants/{id}/toggle`, `DELETE /invariants/{id}`.

## Задачи (FSM)

- `GET /agent/{id}/tasks`, `POST /agent/{id}/tasks` (создать).
- `GET /agent/{id}/tasks/{task_id}` (+ `transitions`).
- `POST /agent/{id}/tasks/{task_id}/transition` — ручной переход (для отладки/UI).
- `POST .../pause`, `POST .../resume`, `DELETE ...`.

## Сессии, ветки, история

- `GET /agent/{session_id}` — полное состояние сессии (история, память, задачи, ветки).
- `GET /agent/{id}/branches`, `POST /agent/{id}/branch` (checkpoint + 2 ветки),
  `POST /agent/{id}/switch`.
- `GET /agents` — список сессий; `DELETE /agent/{id}` — удалить сессию.

## POST `/support`

Ассистент поддержки пользователей (День 33). Тело — модель `SupportRequest`.

| Поле            | Тип    | По умолчанию   | Назначение                                                       |
|-----------------|--------|----------------|------------------------------------------------------------------|
| `message`       | str    | —              | `/support <вопрос>` или `/support --ticket_id=<N> <вопрос>`, либо продолжение |
| `session_id`    | str?   | —              | ID сессии поддержки (привязка «сессия → тикет»)                 |
| `model`         | str    | `deepseek-chat`| Название модели                                                  |
| `temperature`   | float  | `0.3`          | 0..2                                                             |
| `max_tokens`    | int    | `1024`         | 1..8192                                                          |
| `customer_email`| str?   | —              | E-mail покупателя для привязки нового тикета                    |
| `rag_top_k`     | int    | `5`            | Топ-K чанков FAQ                                                 |
| `rag_mode`      | str    | `baseline`     | `baseline`/`filter`/`rerank`                                     |

Флоу: офтопик-гейт (RAG по корпусу `support`) → резолв/создание тикета через
support-mcp → привязка к сессии → ответ с учётом тикета. Ответ содержит:
`response`, `session_id`, `ticket`, `ticket_number`, `ticket_created`,
`abstained`, `sources`, `duration`.

## Служебные

- `GET /healthz` — `{ok, service, provider}`.
- `GET /limits` — лимиты (rate limit, бюджет промпта, параметры Ollama).
