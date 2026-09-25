package io.currency.mcp.web

import io.currency.mcp.Config
import io.currency.mcp.db.RateDb
import io.currency.mcp.rates.SummaryService
import io.currency.mcp.server.summaryJson
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.*
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.time.LocalDate
import java.time.ZoneOffset

/**
 * REST endpoints used for quick browser/curl testing.
 */
fun Route.currencyRoutes(db: RateDb, summaries: SummaryService) {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "currency-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(buildJsonObject { put("name", "get_rate"); put("description", "Latest EUR/USD rate") })
                            add(buildJsonObject { put("name", "get_summary"); put("description", "Daily summary { count, avg, trend, last_n }") })
                        },
                    )
                }.toString(),
            )
        }

        get("/rate") {
            val latest = db.latestRate()
            if (latest == null) {
                call.respondError("No rate available yet")
                return@get
            }
            call.respondJson(
                buildJsonObject {
                    put("source", latest.source)
                    put("target", latest.target)
                    put("rate", latest.rate)
                    put("time", latest.time)
                }.toString(),
            )
        }

        get("/summary") {
            val date = call.request.queryParameters["date"]
                ?: LocalDate.now(ZoneOffset.UTC).toString()
            if (!isValidDate(date)) {
                call.respondError("Invalid 'date'. Expected format: YYYY-MM-DD.")
                return@get
            }
            call.respondJson(summaryJson(summaries.ensureSummary(date)))
        }

        get("/stats") {
            call.respondJson(
                buildJsonObject {
                    put("db_path", Config.dbPath)
                    put("source", Config.source)
                    put("target", Config.target)
                    put("poll_interval_ms", Config.pollIntervalMs)
                    put("summary_interval_ms", Config.summaryIntervalMs)
                    put("stored_rates", db.countRates())
                }.toString(),
            )
        }
    }
}

private fun isValidDate(date: String): Boolean = try {
    LocalDate.parse(date)
    true
} catch (e: Exception) {
    false
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
<title>💱 Currency MCP</title>
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
  input { width: 100%; padding: .6rem .7rem; border-radius: 8px; border: 1px solid #3a4c70; background: #0f1420; color: #e8edf6; }
  button { margin-top: 1rem; padding: .65rem 1.1rem; border: none; border-radius: 8px; background: #2f6fed; color: white; font-weight: 600; cursor: pointer; }
  .result { margin-top: 1rem; padding: .9rem 1rem; border-radius: 10px; background: #1c2940; border: 1px dashed #3a4c70;
            min-height: 1.2em; white-space: pre-wrap; overflow-wrap: anywhere; font-family: monospace; }
  .hint { color: #9aa7bd; font-size: .9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>💱 Currency MCP</h1>
  <p class="sub">EUR/USD rates · SQLite storage · periodic polling &amp; summarization</p>
  <div class="card">
    <h2>Current rate</h2>
    <button onclick="get('/api/rate', 'r-rate')">get_rate</button>
    <div class="result" id="r-rate"></div>
  </div>
  <div class="card">
    <h2>Daily summary</h2>
    <label for="date">Date (YYYY-MM-DD, empty = today)</label>
    <input id="date" type="text" placeholder="2026-09-25">
    <button onclick="summary()">get_summary</button>
    <div class="result" id="r-summary"></div>
  </div>
  <p class="hint">MCP endpoint: <code>POST /mcp</code> (Streamable HTTP).</p>
</div>
<script>
function show(id, text) { document.getElementById(id).textContent = text; }
function get(url, id) {
  fetch(url).then(r => r.json()).then(d => show(id, JSON.stringify(d, null, 2))).catch(e => show(id, 'Error: ' + e));
}
function summary() {
  var d = document.getElementById('date').value.trim();
  get('/api/summary' + (d ? '?date=' + encodeURIComponent(d) : ''), 'r-summary');
}
</script>
</body>
</html>
""".trimIndent()
