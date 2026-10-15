# 🛠 Support MCP

MCP-сервис поддержки пользователей (мини-CRM) на **Kotlin** (Ktor + официальный
[MCP Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)). Хранит
покупателей и тикеты в **SQLite** и отдаёт их агенту как MCP-инструменты.

Агент поддержки (`agent/agent_core.py`, эндпоинт `POST /support`) использует этот
сервис, чтобы создавать тикеты и подгружать контекст обращения при ответе клиенту.

## Инструменты

| Инструмент | Описание | Write |
| --- | --- | --- |
| `create_ticket` | Создать тикет, вернуть номер (например `T-1002`) | ✅ |
| `get_ticket` | Тикет по номеру `T-1001` или числовому id (с покупателем) | — |
| `find_user` | Покупатель по e-mail (`id, email, name, phone, plan`) | — |
| `list_tickets_for_user` | Тикеты покупателя по e-mail | — |
| `search_tickets` | Поиск по номеру/теме/сообщению | — |

## Схема БД

- `users` — покупатели: `id, email, name, phone, plan`.
- `tickets` — обращения: `id, number (T-1000+id), subject, status, priority,
  customer_id → users.id, first_message, created_at`.

## Конфигурация

| Переменная | По умолчанию | Смысл |
| --- | --- | --- |
| `PORT` | `8893` | HTTP-порт |
| `DB_PATH` | `support.db` | путь к SQLite-файлу |
| `SEED_DATA` | `true` | заполнять демо-данными при пустой БД |

## Запуск

```bash
./gradlew run            # http://localhost:8893
# или fat-jar:
./gradlew shadowJar
java -jar build/libs/support-mcp-all.jar
```

## Эндпоинты

| Эндпоинт | Описание |
| --- | --- |
| `GET /` | HTML-страница для ручного теста |
| `GET /api/methods` | Список инструментов |
| `GET /api/users` | Покупатели |
| `GET /api/tickets` | Тикеты |
| `POST /api/tickets` | Создать тикет (`{subject, message, customer_email?}`) |
| `GET /api/stats` | Статистика БД |
| `POST /mcp` | MCP Streamable HTTP |

## Проверка в MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector
```

Выберите **Streamable HTTP** и подключитесь к `http://localhost:8893/mcp`.
