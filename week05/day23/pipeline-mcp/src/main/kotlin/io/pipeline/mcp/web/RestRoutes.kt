package io.pipeline.mcp.web

import io.pipeline.mcp.save.SaveService
import io.pipeline.mcp.search.SearchService
import io.pipeline.mcp.summarize.SummarizeService
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.*
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * REST endpoints used for quick browser/curl testing of the three pipeline tools.
 */
fun Route.pipelineRoutes(
    searchService: SearchService,
    summarizeService: SummarizeService,
    saveService: SaveService,
) {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "pipeline-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(buildJsonObject { put("name", "search"); put("description", "Web search via Yandex") })
                            add(buildJsonObject { put("name", "summarize"); put("description", "LLM summarization via DeepSeek") })
                            add(buildJsonObject { put("name", "save_to_file"); put("description", "Save text to a file") })
                        },
                    )
                }.toString(),
            )
        }

        get("/search") {
            val query = call.request.queryParameters["query"]
            if (query.isNullOrBlank()) {
                call.respondError("Missing 'query' parameter.")
                return@get
            }
            val limit = call.request.queryParameters["limit"]?.toIntOrNull() ?: 5
            try {
                call.respondJson(searchService.search(query, limit))
            } catch (e: Exception) {
                call.respondError(e.message ?: "search failed")
            }
        }

        get("/summarize") {
            val text = call.request.queryParameters["text"]
            if (text.isNullOrBlank()) {
                call.respondError("Missing 'text' parameter.")
                return@get
            }
            try {
                call.respondJson(summarizeService.summarize(text))
            } catch (e: Exception) {
                call.respondError(e.message ?: "summarize failed")
            }
        }

        get("/save") {
            val filename = call.request.queryParameters["filename"]
            val content = call.request.queryParameters["content"]
            if (filename.isNullOrBlank() || content == null) {
                call.respondError("Missing 'filename' or 'content' parameter.")
                return@get
            }
            try {
                call.respondJson(saveService.saveToFile(filename, content))
            } catch (e: Exception) {
                call.respondError(e.message ?: "save failed")
            }
        }
    }
}

private suspend fun ApplicationCall.respondJson(json: String) {
    respondText(json, ContentType.Application.Json)
}

private suspend fun ApplicationCall.respondError(message: String) {
    respondText(
        buildJsonObject { put("error", message) }.toString(),
        ContentType.Application.Json,
        status = HttpStatusCode.BadRequest,
    )
}

private fun indexHtml(): String = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🔗 Pipeline MCP</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2rem 1rem; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0f1420; color: #e8edf6; line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { text-align: center; }
  .sub { text-align: center; color: #9aa7bd; margin-bottom: 2rem; }
  .card { background: #172033; border: 1px solid #2a3a58; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; }
  label { display: block; margin: .8rem 0 .3rem; font-weight: 600; }
  input, textarea { width: 100%; padding: .6rem .7rem; border-radius: 8px; border: 1px solid #3a4c70; background: #0f1420; color: #e8edf6; }
  textarea { min-height: 90px; font-family: monospace; }
  button { margin-top: 1rem; padding: .65rem 1.1rem; border: none; border-radius: 8px; background: #2f6fed; color: white; font-weight: 600; cursor: pointer; }
  .result { margin-top: 1rem; padding: .9rem 1rem; border-radius: 10px; background: #1c2940; border: 1px dashed #3a4c70;
            min-height: 1.2em; white-space: pre-wrap; overflow-wrap: anywhere; font-family: monospace; }
  .hint { color: #9aa7bd; font-size: .9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔗 Pipeline MCP</h1>
  <p class="sub">search → summarize → save_to_file</p>
  <div class="card">
    <h2>1. search</h2>
    <label for="q">Query</label>
    <input id="q" type="text" placeholder="deepseek">
    <button onclick="call('/api/search?query=' + enc('q'))">search</button>
    <div class="result" id="r-search"></div>
  </div>
  <div class="card">
    <h2>2. summarize</h2>
    <label for="t">Text</label>
    <textarea id="t"></textarea>
    <button onclick="call('/api/summarize?text=' + enc('t'))">summarize</button>
    <div class="result" id="r-summarize"></div>
  </div>
  <div class="card">
    <h2>3. save_to_file</h2>
    <label for="f">Filename</label>
    <input id="f" type="text" placeholder="summary.txt">
    <label for="c">Content</label>
    <textarea id="c"></textarea>
    <button onclick="call('/api/save?filename=' + enc('f') + '&content=' + enc('c'))">save_to_file</button>
    <div class="result" id="r-save"></div>
  </div>
  <p class="hint">MCP endpoint: <code>POST /mcp</code> (Streamable HTTP).</p>
</div>
<script>
function enc(id) { return encodeURIComponent(document.getElementById(id).value); }
function call(url) {
  fetch(url).then(r => r.json()).then(d => {
    var id = url.includes('/search') ? 'r-search' : url.includes('/summarize') ? 'r-summarize' : 'r-save';
    document.getElementById(id).textContent = JSON.stringify(d, null, 2);
  }).catch(e => console.error(e));
}
</script>
</body>
</html>
""".trimIndent()
