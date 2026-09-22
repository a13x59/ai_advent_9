# 🔮 Fortune Telling MCP — Client

Console client (Kotlin) that connects to the [`fortune-telling-mcp`](../fortune-telling-mcp) MCP server
over **Streamable HTTP** and lists its available tools.

## What it does

1. Establishes an MCP connection (performs the `initialize` handshake).
2. Fetches the tool list via `tools/list` and prints each tool's name, description, and input schema.

## Run

Requires JDK 17+. First start the server (see `../fortune-telling-mcp`):

```bash
cd ../fortune-telling-mcp && ./gradlew run   # server on http://localhost:8888
```

Then run the client:

```bash
./gradlew run
# or build a fat jar first:
./gradlew shadowJar
java -jar build/libs/fortune-telling-mcp-client-all.jar
```

By default it connects to `http://localhost:8888/mcp`. Pass a custom URL as the first argument:

```bash
./gradlew run --args="http://localhost:8888/mcp"
java -jar build/libs/fortune-telling-mcp-client-all.jar http://localhost:8888/mcp
```

## Example output

```
Connecting to MCP server at http://localhost:8888/mcp ...
Connected to: fortune-telling-mcp v1.0.0

Available tools: 2

1. magic_8_ball
   Description: Answers a yes/no question by consulting the classic Magic 8 Ball. ...
   Input schema:
     {
       "type": "object",
       "properties": {
         "question": { "type": "string", ... }
       },
       "required": ["question"]
     }
...
```
