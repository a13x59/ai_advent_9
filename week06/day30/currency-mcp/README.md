# 💱 Currency MCP

An **MCP (Model Context Protocol) server** written in **Kotlin** (Ktor + the official
[MCP Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)) that periodically
fetches the **EUR/USD** exchange rate and aggregates it into daily summaries.

It runs 24/7: a background job polls the live rate every `POLL_INTERVAL_MS` (default
**2 minutes**) and stores each sample in a **SQLite** database; a second background job
pre-aggregates a daily summary every `SUMMARY_INTERVAL_MS` (default **5 minutes**).

## Tools

### 1. `get_rate`
Returns the latest EUR/USD rate at the current time.

```json
{"source":"EUR","target":"USD","rate":1.13895,"time":"2026-09-25T08:44:11+0000"}
```

### 2. `get_summary`
Returns a daily summary for a given date (`date`, `YYYY-MM-DD`):

```json
{"date":"2026-09-25","count":12,"avg":1.1389,"trend":"up","last_n":[{"rate":1.13895,"time":"..."}]}
```

- `count` — number of stored rate samples for the date;
- `avg` — average rate over the date;
- `trend` — `up` / `down` / `flat` vs. the previous stored day;
- `last_n` — the last N rate samples (N is configurable via `LAST_N`).

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8889` | HTTP/SSE port |
| `DB_PATH` | `currency.db` | SQLite file path |
| `POLL_INTERVAL_MS` | `120000` | rate poll interval (2 min) |
| `SUMMARY_INTERVAL_MS` | `300000` | summary aggregation interval (5 min) |
| `RATES_API_URL` | `https://allratestoday.com/api/v1/rates` | rates API base URL |
| `RATES_API_KEY` | `` | bearer token for the rates API |
| `SOURCE` / `TARGET` | `USD` / `EUR` | supported pair |
| `LAST_N` | `5` | number of last rates kept in `last_n` |

## Run

Requires JDK 17+ (the Gradle wrapper downloads everything else).

```bash
./gradlew run            # start on http://localhost:8889
# or build a fat jar:
./gradlew shadowJar
java -jar build/libs/currency-mcp-all.jar
```

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Interactive HTML test page |
| `GET /api/methods` | List of tools |
| `GET /api/rate` | Latest EUR/USD rate |
| `GET /api/summary?date=YYYY-MM-DD` | Daily summary |
| `GET /api/stats` | Service/DB stats |
| `POST /mcp` | MCP Streamable HTTP transport |

## Test in MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector
```

Choose **Streamable HTTP** and connect to `http://localhost:8889/mcp`. The two tools
(`get_rate`, `get_summary`) should appear under *Tools*.

## Supported currencies

Only **EUR** and **USD** are supported (requirement 3). Unsupported pairs return an error
from the tool.
