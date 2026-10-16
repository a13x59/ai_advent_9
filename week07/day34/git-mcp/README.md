# 🌿 Git MCP

**MCP (Model Context Protocol) server** для доступа ассистента к git-репозиторию проекта. Написан на Kotlin (Ktor +
официальный MCP Kotlin SDK) — по аналогии с
`fortune-telling-mcp`, `currency-mcp` и `pipeline-mcp`.

## Инструменты

### 1. `git_branch`

Текущая ветка и список локальных веток.

- Вход: без аргументов.
- Выход: `{"current_branch": "...", "branches": ["..."]}`.

### 2. `list_files`

Список отслеживаемых (git-tracked) файлов проекта.

- Вход: `path` (по умолчанию `.`), `max_files` (по умолчанию `200`).
- Выход: `{"count": N, "truncated": bool, "files": ["..."]}`.

### 3. `git_diff`

Изменения рабочей копии (или staged-изменения).

- Вход: `staged` (bool), `stat` (bool), `path` (по умолчанию `.`).
- Выход: текст диффа (или `No changes (working tree clean).`).

## Скоуп

Операции выполняются через `git -C <GIT_PROJECT_PATH> …` с pathspec `.`, поэтому вывод ограничен каталогом проекта, даже
если git-репозиторий больше (например, весь репозиторий курса). `GIT_PROJECT_PATH` по умолчанию равен рабочему каталогу
сервиса; переопределяется переменной окружения.

## Запуск

Требуется JDK 17+ (Gradle wrapper скачает остальное).

```bash
GIT_PROJECT_PATH=/path/to/project ./gradlew run    # порт 8892
# или:
./gradlew shadowJar
java -jar build/libs/git-mcp-all.jar
```

## Эндпоинты

| Endpoint                                | Описание                      |
|-----------------------------------------|-------------------------------|
| `GET /`                                 | Интерактивная HTML-страница   |
| `GET /api/methods`                      | Список инструментов           |
| `GET /api/branch`                       | Текущая ветка                 |
| `GET /api/files?path=.&max_files=100`   | Список файлов                 |
| `GET /api/diff?staged=false&stat=false` | Дифф                          |
| `POST /mcp`                             | MCP Streamable HTTP transport |

## Проверка

```bash
curl http://localhost:8892/api/branch
curl "http://localhost:8892/api/files?path=.&max_files=10"
curl "http://localhost:8892/api/diff?stat=true"
```
