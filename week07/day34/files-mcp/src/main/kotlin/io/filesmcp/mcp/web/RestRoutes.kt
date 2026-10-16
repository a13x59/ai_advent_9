package io.filesmcp.mcp.web

import io.filesmcp.mcp.files.FileService
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * REST-эндпоинты для проверки в браузере (по аналогии с остальными MCP-сервисами).
 * Основной транспорт — POST /mcp (Streamable HTTP); REST — только для ручной отладки.
 */
fun Route.fileRoutes(fileService: FileService) {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "files-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(buildJsonObject { put("name", "list_dir"); put("endpoint", "/api/list?path=.&max_files=200") })
                            add(buildJsonObject { put("name", "read_file"); put("endpoint", "/api/read?path=...&start_line=&end_line=") })
                            add(buildJsonObject { put("name", "search_files"); put("endpoint", "/api/search?query=...&path=.&regex=false&max_results=50") })
                            add(buildJsonObject { put("name", "write_file"); put("endpoint", "/api/write?path=...&content=...") })
                            add(buildJsonObject { put("name", "edit_file"); put("endpoint", "/api/edit?path=...&old=...&new=...&replace_all=false") })
                            add(buildJsonObject { put("name", "diff"); put("endpoint", "/api/diff") })
                            add(buildJsonObject { put("name", "apply_changes"); put("endpoint", "/api/apply") })
                            add(buildJsonObject { put("name", "discard_changes"); put("endpoint", "/api/discard") })
                        },
                    )
                }.toString(),
            )
        }

        get("/list") {
            val path = call.request.queryParameters["path"]?.ifBlank { "." } ?: "."
            val maxFiles = call.request.queryParameters["max_files"]?.toIntOrNull() ?: 200
            val (count, files) = fileService.listDir(path, maxFiles)
            call.respondJson(
                buildJsonObject {
                    put("count", count)
                    put("truncated", count > files.size)
                    put("files", buildJsonArray { files.forEach { add(it) } })
                }.toString(),
            )
        }

        get("/read") {
            val path = call.request.queryParameters["path"]
            if (path.isNullOrBlank()) {
                call.respondError("'path' is required")
                return@get
            }
            val r = fileService.readFile(
                path,
                call.request.queryParameters["start_line"]?.toIntOrNull(),
                call.request.queryParameters["end_line"]?.toIntOrNull(),
            )
            call.respondJson(
                buildJsonObject {
                    put("path", r.path)
                    put("total_lines", r.totalLines)
                    put("start_line", r.startLine)
                    put("end_line", r.endLine)
                    put("truncated", r.truncated)
                    put("content", r.content)
                }.toString(),
            )
        }

        get("/search") {
            val query = call.request.queryParameters["query"]
            if (query.isNullOrBlank()) {
                call.respondError("'query' is required")
                return@get
            }
            val r = fileService.searchFiles(
                query,
                call.request.queryParameters["path"]?.ifBlank { "." } ?: ".",
                call.request.queryParameters["glob"]?.ifBlank { "*" } ?: "*",
                call.request.queryParameters["regex"]?.toBooleanStrictOrNull() ?: false,
                call.request.queryParameters["max_results"]?.toIntOrNull() ?: 50,
            )
            call.respondJson(
                buildJsonObject {
                    put("query", r.query)
                    put("count", r.count)
                    put("truncated", r.truncated)
                    put("matches", buildJsonArray {
                        r.matches.forEach {
                            add(buildJsonObject {
                                put("file", it.file)
                                put("line", it.line)
                                put("text", it.text)
                            })
                        }
                    })
                }.toString(),
            )
        }

        get("/write") {
            val path = call.request.queryParameters["path"]
            val content = call.request.queryParameters["content"]
            if (path.isNullOrBlank() || content == null) {
                call.respondError("'path' and 'content' are required")
                return@get
            }
            call.respondJson(changeJson(fileService.writeFile(path, content)).toString())
        }

        get("/edit") {
            val path = call.request.queryParameters["path"]
            val old = call.request.queryParameters["old"]
            val new = call.request.queryParameters["new"]
            if (path.isNullOrBlank() || old == null || new == null) {
                call.respondError("'path', 'old' and 'new' are required")
                return@get
            }
            call.respondJson(
                changeJson(
                    fileService.editFile(
                        path, old, new,
                        call.request.queryParameters["replace_all"]?.toBooleanStrictOrNull() ?: false,
                    ),
                ).toString(),
            )
        }

        get("/diff") {
            call.respondJson(buildJsonObject { put("diff", fileService.diff()) }.toString())
        }

        get("/apply") {
            val r = fileService.applyChanges()
            call.respondJson(
                buildJsonObject {
                    put("applied", buildJsonArray { r.applied.forEach { add(it) } })
                    put("diff", r.diff)
                }.toString(),
            )
        }

        get("/discard") {
            call.respondJson(buildJsonObject { put("discarded", fileService.discardChanges()) }.toString())
        }
    }
}

private fun changeJson(r: FileService.ChangeResult) = buildJsonObject {
    put("path", r.path)
    put("status", r.status)
    r.occurrences?.let { put("occurrences", it) }
    put("diff", r.diff)
}

private suspend fun ApplicationCall.respondJson(json: String) {
    respondText(json, ContentType.Application.Json)
}

private suspend fun ApplicationCall.respondError(message: String) {
    respondText(
        buildJsonObject { put("error", message) }.toString(),
        ContentType.Application.Json,
        HttpStatusCode.BadRequest,
    )
}

private fun indexHtml(): String = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>📁 Files MCP</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2rem 1rem; font-family: system-ui, -apple-system, sans-serif;
         background: #151517; color: #f9fafb; line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { text-align: center; }
  .sub { text-align: center; color: #979da6; margin-bottom: 2rem; }
  .card { background: #232324; border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
          padding: 1.5rem; margin-bottom: 1.5rem; }
  button { margin-top: .8rem; padding: .6rem 1.1rem; font-size: 1rem; cursor: pointer;
           border: none; border-radius: 8px; background: #4176e6; color: white; font-weight: 600; }
  .result { margin-top: 1rem; padding: .9rem 1rem; border-radius: 10px;
            background: #151517; border: 1px solid rgba(255,255,255,.12); white-space: pre-wrap;
            overflow-wrap: anywhere; font-family: ui-monospace, monospace; font-size: .85rem; }
  code { background: #1b1b1c; padding: .1em .4em; border-radius: 5px; }
  a { color: #adb2b8; }
</style>
</head>
<body>
<div class="wrap">
  <h1>📁 Files MCP</h1>
  <p class="sub">REST API for the files MCP server · <a href="/api/methods">/api/methods</a></p>

  <div class="card">
    <h2>List</h2>
    <button onclick="get('/api/list?path=.&max_files=100', 'list')">List files</button>
    <div class="result" id="list"></div>
  </div>

  <div class="card">
    <h2>Read</h2>
    <button onclick="get('/api/read?path=README.md', 'read')">Read README.md</button>
    <div class="result" id="read"></div>
  </div>

  <div class="card">
    <h2>Search</h2>
    <button onclick="get('/api/search?query=FileService&path=.&max_results=20', 'search')">Search "FileService"</button>
    <div class="result" id="search"></div>
  </div>

  <div class="card">
    <h2>Diff / Apply</h2>
    <button onclick="get('/api/diff', 'diff')">Show pending diff</button>
    <button onclick="get('/api/apply', 'apply')">Apply changes</button>
    <button onclick="get('/api/discard', 'discard')">Discard changes</button>
    <div class="result" id="diff"></div>
    <div class="result" id="apply"></div>
    <div class="result" id="discard"></div>
  </div>

  <p class="sub">MCP endpoint: <code>POST /mcp</code> (Streamable HTTP). Root: file system scope, dry-run by default.</p>
</div>
<script>
function get(url, id) {
  document.getElementById(id).textContent = 'Loading...';
  fetch(url).then(r => r.json())
    .then(d => document.getElementById(id).textContent = JSON.stringify(d, null, 2))
    .catch(e => document.getElementById(id).textContent = 'Error: ' + e);
}
</script>
</body>
</html>
""".trimIndent()
