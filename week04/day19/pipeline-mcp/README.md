# 🔗 Pipeline MCP

An **MCP (Model Context Protocol) server** written in **Kotlin** (Ktor + the official
[MCP Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)) that exposes a
three-tool **data pipeline**:

```
search ──► summarize ──► save_to_file
(fetch)    (process)     (store)
```

This is the "Композиция MCP-инструментов" demo: the first tool fetches data, the second
processes it, and the third saves the result to a file.

## Tools

### 1. `search` — fetch data
Web search via the **Yandex Search API**. Returns the top results as:

```json
{"query":"deepseek","count":3,"results":[{"title":"...","url":"...","passage":"..."}]}
```

- Input: `query` (string, required), `limit` (integer, optional).
- Requires `YANDEX_API_KEY` (and `YANDEX_FOLDER_ID` for the Yandex Cloud folder).

### 2. `summarize` — process data
LLM summarization via the **DeepSeek** chat-completions API. Returns:

```json
{"summary":"...","model":"deepseek-chat","input_chars":1234,"usage":{"prompt_tokens":"...","completion_tokens":"..."}}
```

- Input: `text` (string, required).
- Requires `DEEPSEEK_API_KEY`.

### 3. `save_to_file` — store the result
Writes text content to a file under the output directory and returns:

```json
{"path":"/abs/output/summary.txt","filename":"summary.txt","bytes":543,"preview":"..."}
```

- Input: `filename` (string, required), `content` (string, required).
- The file name is sanitized and path traversal outside the output dir is rejected.

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8890` | HTTP/SSE port (distinct from fortune 8888 and currency 8889) |
| `YANDEX_SEARCH_URL` | `https://searchapi.yandex.ru/v1/web/search` | Yandex Search API base URL |
| `YANDEX_API_KEY` | `` | Yandex Search API key (`Authorization: Api-Key <key>`) |
| `YANDEX_FOLDER_ID` | `` | Yandex Cloud folder id (`folderid` query param) |
| `SEARCH_MAX_RESULTS` | `5` | default `limit` for `search` |
| `DEEPSEEK_API_URL` | `https://api.deepseek.com/v1/chat/completions` | DeepSeek completions endpoint |
| `DEEPSEEK_API_KEY` | `` | DeepSeek API key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | model used by `summarize` |
| `SUMMARIZE_MAX_TOKENS` | `512` | max tokens for the summarization completion |
| `SUMMARIZE_TEMPERATURE` | `0.3` | temperature for the summarization completion |
| `OUTPUT_DIR` | `output` | directory where `save_to_file` writes files |

## Run

Requires JDK 17+ (the Gradle wrapper downloads everything else).

```bash
./gradlew run            # start on http://localhost:8890
# or build a fat jar:
./gradlew shadowJar
java -jar build/libs/pipeline-mcp-all.jar
```

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Interactive HTML test page |
| `GET /api/methods` | List of tools |
| `GET /api/search?query=...&limit=...` | Web search |
| `GET /api/summarize?text=...` | LLM summarization |
| `GET /api/save?filename=...&content=...` | Save text to a file |
| `POST /mcp` | MCP Streamable HTTP transport |

## Test in MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector
```

Choose **Streamable HTTP** and connect to `http://localhost:8890/mcp`. The three tools
(`search`, `summarize`, `save_to_file`) should appear under *Tools*.

## Chaining with the agent

The Python agent (`agent/`) calls these tools automatically through function calling. A
request like *"найди информацию про X и сохрани сводку в файл"* makes the model run the
chain: `search` → `summarize` (fed with the search output) → `save_to_file` (fed with the
summary).
