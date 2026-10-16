# 📁 Files MCP

**MCP (Model Context Protocol) server** для файлового ассистента (День 34): чтение, поиск по нескольким файлам и «сухая»
правка файлов проекта. Написан на Kotlin (Ktor + официальный MCP Kotlin SDK) — по аналогии с `git-mcp` и `pipeline-mcp`.

Ключевая особенность — **dry-run по умолчанию**: инструменты записи не трогают диск, а кладут правку в «стейджинг» и
возвращают unified diff. Запись происходит только по явному вызову `apply_changes`. Это даёт агенту безопасный цикл
«разведай → проанализируй → покажи diff → примени».

## Инструменты

| Инструмент        | Вход                                                | Что делает                                                         |
|-------------------|-----------------------------------------------------|--------------------------------------------------------------------|
| `list_dir`        | `path` (`.`), `max_files` (`200`)                   | список файлов подкаталога (файловая система, включая untracked)    |
| `read_file`       | `path`, `start_line?`, `end_line?`                  | чтение файла целиком или диапазона строк                           |
| `search_files`    | `query`, `path?`, `glob?`, `regex?`, `max_results?` | grep по нескольким файлам → `file:line:text`                       |
| `write_file`      | `path`, `content`                                   | создать/перезаписать файл (в стейджинг, diff)                      |
| `edit_file`       | `path`, `old_string`, `new_string`, `replace_all?`  | точечная замена (в стейджинг, diff); ошибка при 0 или >1 вхождений |
| `diff`            | —                                                   | unified diff всех стейджинговых изменений                          |
| `apply_changes`   | —                                                   | записать стейджинг на диск                                         |
| `discard_changes` | —                                                   | отбросить стейджинг без записи                                     |

## Скоуп и безопасность

- Операции ограничены каталогом `FILES_ROOT` (по умолчанию — родительский каталог
  сервиса, т.е. `week07/day34` при запуске из `files-mcp/`).
- Абсолютные пути и path traversal (`..`) отвергаются на этапе резолва.
- Чтение/поиск ограничены по объёму (`FILES_MAX_READ_CHARS`, `FILES_MAX_SEARCH_RESULTS`).
- Запись всегда идёт через стейджинг: `write_file`/`edit_file` только готовят diff, `apply_changes` — сохраняет.

## Конфигурация (переменные окружения)

| Переменная                 | Default    | Назначение                                |
|----------------------------|------------|-------------------------------------------|
| `PORT`                     | `8894`     | HTTP-порт (не пересекается с другими MCP) |
| `FILES_ROOT`               | родитель сервиса (`week07/day34`) | корень файлового скоупа        |
| `FILES_MAX_READ_CHARS`     | `20000`    | лимит символов при чтении файла           |
| `FILES_MAX_SEARCH_RESULTS` | `100`      | лимит результатов grep                    |

## Запуск

Требуется JDK 17+ (Gradle wrapper скачает остальное).

```bash
./gradlew run                                     # порт 8894, скоуп = week07/day34
# или (для другого скоупа):
FILES_ROOT=/path/to/project ./gradlew run
# или fat-jar:
./gradlew shadowJar
java -jar build/libs/files-mcp-all.jar            # запускать из files-mcp/ (скоуп = parent)
```

## Эндпоинты

| Endpoint                                                | Описание                      |
|---------------------------------------------------------|-------------------------------|
| `GET /`                                                 | Интерактивная HTML-страница   |
| `GET /api/methods`                                      | Список инструментов           |
| `GET /api/list?path=.&max_files=200`                    | Список файлов                 |
| `GET /api/read?path=...&start_line=&end_line=`          | Чтение файла                  |
| `GET /api/search?query=...&regex=false`                 | Grep-поиск                    |
| `GET /api/write?path=...&content=...`                   | Стейджинг нового файла        |
| `GET /api/edit?path=...&old=...&new=...`                | Стейджинг правки              |
| `GET /api/diff` / `GET /api/apply` / `GET /api/discard` | diff / применить / отбросить  |
| `POST /mcp`                                             | MCP Streamable HTTP transport |

## Проверка

```bash
curl http://localhost:8894/api/methods
curl "http://localhost:8894/api/list?path=.&max_files=20"
curl "http://localhost:8894/api/search?query=FileService&path=."
```

## Тесты

```bash
./gradlew test
```
