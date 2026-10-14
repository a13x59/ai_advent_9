package io.fortunetelling.mcp.web

import io.fortunetelling.mcp.fortune.Bibliomancy
import io.fortunetelling.mcp.fortune.Book
import io.fortunetelling.mcp.fortune.Line
import io.fortunetelling.mcp.fortune.Magic8Ball
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * REST endpoints used for testing in the browser.
 */
fun Route.fortuneRoutes() {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "fortune-telling-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(
                                buildJsonObject {
                                    put("name", "magic_8_ball")
                                    put("description", "Classic Magic 8 Ball with the standard set of 20 answers.")
                                    put("endpoint", "/api/magic8ball?question=...")
                                },
                            )
                            add(
                                buildJsonObject {
                                    put("name", "bibliomancy")
                                    put("description", "Divination by opening a random page of a chosen book.")
                                    put("endpoint", "/api/bibliomancy?question=...&book=<book-id>&line=top|bottom")
                                },
                            )
                        },
                    )
                }.toString(),
            )
        }

        get("/books") {
            call.respondJson(
                buildJsonObject {
                    put(
                        "books",
                        buildJsonArray {
                            Book.entries.forEach { book ->
                                add(
                                    buildJsonObject {
                                        put("id", book.id)
                                        put("title", book.title)
                                    },
                                )
                            }
                        },
                    )
                }.toString(),
            )
        }

        get("/lines") {
            call.respondJson(
                buildJsonObject {
                    put(
                        "lines",
                        buildJsonArray {
                            Line.entries.forEach { line ->
                                add(buildJsonObject { put("id", line.id) })
                            }
                        },
                    )
                }.toString(),
            )
        }

        get("/magic8ball") {
            val question = call.request.queryParameters["question"]
            if (question.isNullOrBlank()) {
                call.respondError("The 'question' query parameter is required.")
                return@get
            }
            val outcome = Magic8Ball.predict(question)
            call.respondJson(
                buildJsonObject {
                    put("method", "magic_8_ball")
                    put("question", question)
                    put("answer", outcome.answer)
                    put("result", outcome.text)
                }.toString(),
            )
        }

        get("/bibliomancy") {
            val question = call.request.queryParameters["question"]
            val book = Book.fromId(call.request.queryParameters["book"])
            val line = Line.fromId(call.request.queryParameters["line"])

            when {
                question.isNullOrBlank() ->
                    call.respondError("The 'question' query parameter is required.")
                book == null ->
                    call.respondError("Unknown 'book'. Valid values: ${Book.entries.joinToString(", ") { it.id }}.")
                line == null ->
                    call.respondError("Unknown 'line'. Valid values: ${Line.entries.joinToString(", ") { it.id }}.")
                else -> {
                    val prediction = Bibliomancy.predict(question, book, line)
                    call.respondJson(
                        buildJsonObject {
                            put("method", "bibliomancy")
                            put("question", question)
                            put(
                                "book",
                                buildJsonObject {
                                    put("id", book.id)
                                    put("title", book.title)
                                },
                            )
                            put("line", line.id)
                            put("page", prediction.page)
                            put("passage", prediction.passage)
                            put("result", prediction.text)
                        }.toString(),
                    )
                }
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
<title>🔮 Fortune Telling MCP</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #14121f;
    color: #ece9f5;
    line-height: 1.5;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { text-align: center; font-size: 2rem; }
  .sub { text-align: center; color: #a99fc4; margin-bottom: 2rem; }
  .card {
    background: #1f1b30; border: 1px solid #322b4d; border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .card h2 { margin-top: 0; }
  label { display: block; margin: .8rem 0 .3rem; font-weight: 600; }
  input, select {
    width: 100%; padding: .6rem .7rem; border-radius: 8px;
    border: 1px solid #4a3f6b; background: #14121f; color: #ece9f5; font-size: 1rem;
  }
  button {
    margin-top: 1rem; padding: .65rem 1.1rem; font-size: 1rem; cursor: pointer;
    border: none; border-radius: 8px; background: #7c5cff; color: white; font-weight: 600;
  }
  button:hover { background: #6a4bf0; }
  .result {
    margin-top: 1rem; padding: .9rem 1rem; border-radius: 10px;
    background: #2a2340; border: 1px dashed #4a3f6b; min-height: 1.2em;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .hint { color: #a99fc4; font-size: .9rem; }
  code { background: #2a2340; padding: .1em .4em; border-radius: 5px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔮 Fortune Telling MCP</h1>
  <p class="sub">REST API for the fortune-telling MCP server · <a href="/api/methods" style="color:#a99fc4">/api/methods</a></p>

  <div class="card">
    <h2>🎱 Magic 8 Ball</h2>
    <label for="q8">Question</label>
    <input id="q8" type="text" placeholder="Will it rain tomorrow?">
    <button onclick="askBall()">Ask the ball</button>
    <div class="result" id="r8"></div>
  </div>

  <div class="card">
    <h2>📚 Bibliomancy</h2>
    <label for="qbm">Question</label>
    <input id="qbm" type="text" placeholder="What does the future hold?">
    <label for="book">Book</label>
    <select id="book"></select>
    <label for="line">Line</label>
    <select id="line">
      <option value="top" selected>top</option>
      <option value="bottom">bottom</option>
    </select>
    <button onclick="askBook()">Open the book</button>
    <div class="result" id="rbm"></div>
  </div>

  <p class="hint">
    MCP endpoint: <code>POST /mcp</code> (Streamable HTTP). Open it with
    <code>npx -y @modelcontextprotocol/inspector</code> and connect to
    <code>http://localhost:8888/mcp</code>.
  </p>
</div>

<script>
function show(id, text) { document.getElementById(id).textContent = text; }

function askBall() {
  var q = document.getElementById('q8').value.trim();
  if (!q) { show('r8', 'Please enter a question.'); return; }
  show('r8', 'Shaking...');
  fetch('/api/magic8ball?question=' + encodeURIComponent(q))
    .then(function (r) { return r.json(); })
    .then(function (data) { show('r8', data.error ? data.error : data.result); })
    .catch(function (e) { show('r8', 'Error: ' + e); });
}

function askBook() {
  var q = document.getElementById('qbm').value.trim();
  var b = document.getElementById('book').value;
  var l = document.getElementById('line').value;
  if (!q) { show('rbm', 'Please enter a question.'); return; }
  show('rbm', 'Opening the book...');
  fetch('/api/bibliomancy?question=' + encodeURIComponent(q) + '&book=' + encodeURIComponent(b) + '&line=' + encodeURIComponent(l))
    .then(function (r) { return r.json(); })
    .then(function (data) { show('rbm', data.error ? data.error : data.result); })
    .catch(function (e) { show('rbm', 'Error: ' + e); });
}

fetch('/api/books')
  .then(function (r) { return r.json(); })
  .then(function (data) {
    var sel = document.getElementById('book');
    data.books.forEach(function (b) {
      var opt = document.createElement('option');
      opt.value = b.id;
      opt.textContent = b.title;
      sel.appendChild(opt);
    });
  })
  .catch(function () {});
</script>
</body>
</html>
""".trimIndent()
