# 🔮 Fortune Telling MCP

A "fortune telling" **MCP (Model Context Protocol) server** written in **Kotlin** (Ktor + the official
[MCP Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)). It exposes two prediction
methods and also ships a plain **REST API** so it can be tested both in the browser and in the
MCP Inspector.

## Methods

### 1. Magic 8 Ball — `magic_8_ball`
Classic Magic 8 Ball with the standard set of 20 answers.

- Input: `question` (string)
- Output format: `❓$question 🔮 $randomAnswer`
- Example: `❓Will it rain tomorrow? 🔮 Signs point to yes.`

### 2. Bibliomancy — `bibliomancy`
Divination by books: open a random page of a chosen book and read a passage from the top or the bottom.

- Input: `question` (string), `book` (enum), `line` (enum: `top` | `bottom`)
- Output format: `❓Question: $question 📚 From book: $book.title 📄 Page: $page 📖 Answer: $passage`
- Example: `❓Question: What does the future hold? 📚 From book: The Art of War 📄 Page: 2 📖 Answer: In the midst of chaos, there is also opportunity.`

Books (enum): `the_bible`, `war_and_peace`, `alice_in_wonderland`, `the_art_of_war`, `hamlet`, `the_little_prince`.

## Run

Requires JDK 17+ (the Gradle wrapper downloads everything else).

```bash
./gradlew run            # start on http://localhost:8888
# or build a fat jar:
./gradlew shadowJar
java -jar build/libs/fortune-telling-mcp-all.jar
```

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Interactive HTML test page |
| `GET /api/methods` | List of prediction methods |
| `GET /api/books` | List of books available for bibliomancy |
| `GET /api/lines` | `top` / `bottom` |
| `GET /api/magic8ball?question=...` | Magic 8 Ball |
| `GET /api/bibliomancy?question=...&book=...&line=...` | Bibliomancy |
| `POST /mcp` | MCP Streamable HTTP transport |

CORS is enabled (`anyHost`), and DNS-rebinding protection is disabled on `/mcp` so the endpoint
accepts requests from any origin.

## Test in the browser

Open http://localhost:8888 and use the two forms, or call the REST API directly, e.g.:

```bash
curl "http://localhost:8888/api/magic8ball?question=Will%20I%20pass%20the%20interview%3F"
curl "http://localhost:8888/api/bibliomancy?question=What%20next%3F&book=the_art_of_war&line=top"
```

## Test in MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector
```

In the inspector, choose **Streamable HTTP** and connect to `http://localhost:8888/mcp`.
The two tools (`magic_8_ball`, `bibliomancy`) should appear under *Tools*.
