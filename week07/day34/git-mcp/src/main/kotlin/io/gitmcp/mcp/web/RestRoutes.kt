package io.gitmcp.mcp.web

import io.gitmcp.mcp.git.GitService
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
 */
fun Route.gitRoutes(gitService: GitService) {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "git-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(buildJsonObject {
                                put("name", "git_branch")
                                put("endpoint", "/api/branch")
                            })
                            add(buildJsonObject {
                                put("name", "list_files")
                                put("endpoint", "/api/files?path=.&max_files=200")
                            })
                            add(buildJsonObject {
                                put("name", "git_diff")
                                put("endpoint", "/api/diff?staged=false&stat=false")
                            })
                        },
                    )
                }.toString(),
            )
        }

        get("/branch") {
            call.respondJson(
                buildJsonObject {
                    put("current_branch", gitService.currentBranch())
                    put("branches", buildJsonArray { gitService.branches().forEach { add(it) } })
                }.toString(),
            )
        }

        get("/files") {
            val path = call.request.queryParameters["path"]?.ifBlank { "." } ?: "."
            val maxFiles = call.request.queryParameters["max_files"]?.toIntOrNull() ?: 200
            val (count, files) = gitService.listFiles(path, maxFiles)
            call.respondJson(
                buildJsonObject {
                    put("count", count)
                    put("truncated", count > files.size)
                    put("files", buildJsonArray { files.forEach { add(it) } })
                }.toString(),
            )
        }

        get("/diff") {
            val staged = call.request.queryParameters["staged"]?.toBooleanStrictOrNull() ?: false
            val stat = call.request.queryParameters["stat"]?.toBooleanStrictOrNull() ?: false
            val path = call.request.queryParameters["path"]?.ifBlank { "." } ?: "."
            val diff = gitService.diff(staged, stat, path)
            call.respondJson(
                buildJsonObject {
                    put("staged", staged)
                    put("stat", stat)
                    put("diff", diff.ifBlank { "No changes (working tree clean)." })
                }.toString(),
            )
        }
    }
}

private suspend fun ApplicationCall.respondJson(json: String) {
    respondText(json, ContentType.Application.Json)
}

private fun indexHtml(): String = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🌿 Git MCP</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2rem 1rem; font-family: system-ui, -apple-system, sans-serif;
         background: #0f1419; color: #e6edf3; line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { text-align: center; }
  .sub { text-align: center; color: #8b98a5; margin-bottom: 2rem; }
  .card { background: #161c22; border: 1px solid #30363d; border-radius: 12px;
          padding: 1.5rem; margin-bottom: 1.5rem; }
  button { margin-top: .8rem; padding: .6rem 1.1rem; font-size: 1rem; cursor: pointer;
           border: none; border-radius: 8px; background: #238636; color: white; font-weight: 600; }
  .result { margin-top: 1rem; padding: .9rem 1rem; border-radius: 10px;
            background: #0d1117; border: 1px dashed #30363d; white-space: pre-wrap;
            overflow-wrap: anywhere; font-family: ui-monospace, monospace; font-size: .85rem; }
  code { background: #0d1117; padding: .1em .4em; border-radius: 5px; }
  a { color: #8b98a5; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🌿 Git MCP</h1>
  <p class="sub">REST API for the git MCP server · <a href="/api/methods">/api/methods</a></p>

  <div class="card">
    <h2>Branch</h2>
    <button onclick="get('/api/branch', 'branch')">Show branch</button>
    <div class="result" id="branch"></div>
  </div>

  <div class="card">
    <h2>Files</h2>
    <button onclick="get('/api/files?path=.&max_files=100', 'files')">List files</button>
    <div class="result" id="files"></div>
  </div>

  <div class="card">
    <h2>Diff</h2>
    <button onclick="get('/api/diff?staged=false&stat=true', 'diff')">Show diffstat</button>
    <div class="result" id="diff"></div>
  </div>

  <p class="sub">MCP endpoint: <code>POST /mcp</code> (Streamable HTTP).</p>
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
