# День 34. Ассистент для работы с файлами проекта

AI-ассистент, который **активно работает с файлами** проекта, а не просто отвечает текстом. Через MCP-сервис
`files-mcp` он умеет читать файлы, искать по нескольким файлам, анализировать содержимое и создавать/изменять файлы —
причём по задаче на уровне **цели** («найди все места использования X», «сгенерируй ADR»), а не по команде «открой
файл X».

## Результат

- **MCP-сервис `files-mcp`** (Kotlin + Ktor, порт 8894) — файловые инструменты с **dry-run по умолчанию**:
  `list_dir`, `read_file`, `search_files`, `write_file`, `edit_file`, `diff`, `apply_changes`, `discard_changes`.
  Запись ничего не трогает на диске, пока не вызван `apply_changes`; пути вне `FILES_ROOT` и `..` отвергаются.
- **Эндпоинт `POST /files`** — в `agent/agent_core.py` (`FilesRequest` + персона `build_files_system_block` +
  `_find_files_client`/`_files_tool_call`). Ассистент сам ведёт цикл «разведка → анализ → правки» через function calling
  (`_run_tool_loop`), а эндпоинт детерминированно применяет (`apply=true`) или показывает diff (`apply=false`).
- **Клиент `FilesMcpClient`** — `agent/mcp_client.py`, зарегистрирован в `main.py`/`local_main.py`.
- **Интерфейс** — `agent/files.html` (в стиле DeepSeek Harness): цель в textarea, переключатель «Предпросмотр /
  Применить», лента действий с файлами и раскрашенный diff; ссылка на главной `agent/index.html`.
- **Демо и тесты** — `agent/files_demo.py` (4 сценария), `agent/test_files_endpoint.py`,
  `files-mcp/src/test/.../FileServiceTest.kt` (12 тестов: path traversal, dry-run/apply, search, diff).

## Сценарии

| Цель (сообщение)                                          | Поведение                                                                  |
|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| «найди все места использования FilesMcpClient»            | `search_files` по коду → отчёт `file:line` (dry-run, без изменений)          |
| «проверь, что каждый .py начинается с комментария»        | `read_file` файлов → детерминированная валидация → список нарушений          |
| «сгенерируй ADR-034 по модели dry-run/apply»              | `write_file` нового файла → diff (dry-run) или запись на диск (`apply=true`) |
| «обнови README files-mcp на основе кода»                  | `read_file` кода/доки → `edit_file` → diff                                   |

## Соответствие требованиям

| Требование                                        | Где                                                                 |
|---------------------------------------------------|---------------------------------------------------------------------|
| читать файлы проекта                              | `read_file` (`files-mcp` → `FilesMcpClient`)                        |
| искать по нескольким файлам                       | `search_files` (grep, `path`/`glob`/`regex`)                        |
| анализировать содержимое                          | автономный цикл `/files` + персона `build_files_system_block`       |
| создавать/изменять файлы                          | `write_file` / `edit_file` + `apply_changes`                        |
| ассистент сам инициирует работу                   | function calling (`_run_tool_loop`), задача задаётся на уровне цели |
| изменения сохраняются или выводятся как diff      | dry-run (diff) по умолчанию + `apply=true` (запись + итоговый diff) |
| результат воспроизводим                           | `files_demo.py` + `test_files_endpoint.py` + `FileServiceTest.kt`   |
| отдельный экран                                   | `agent/files.html` (стиль DeepSeek Harness) + ссылка в `index.html` |

## Быстрый старт

```bash
cd week07/day34

# 1. files-mcp (файловые операции; скоуп по умолчанию — корень day34)
cd files-mcp && ./gradlew run &          # http://localhost:8894

# 2. агент (локальная llama3.1 или DeepSeek)
cd ../agent
uvicorn local_main:app --host 0.0.0.0 --port 8002     # локальная модель
# uvicorn main:app --host 0.0.0.0 --port 8000          # DeepSeek

# 3. демо в консоли (4 сценария)
python files_demo.py http://127.0.0.1:8002 --model llama3.1

# или откройте agent/files.html в браузере
```

Проверка сервиса вручную:

```bash
curl http://localhost:8894/api/methods
curl "http://localhost:8894/api/list?path=.&max_files=20"
curl "http://localhost:8894/api/search?query=FileService&path=."
curl -s -X POST http://localhost:8002/files -H 'Content-Type: application/json' \
  -d '{"message":"найди все места использования FilesMcpClient","apply":false}'
```

Подробности — в `files-mcp/README.md`, `agent/README.md`, `agent/docs/api.md`, `agent/docs/mcp-and-rag.md`.
