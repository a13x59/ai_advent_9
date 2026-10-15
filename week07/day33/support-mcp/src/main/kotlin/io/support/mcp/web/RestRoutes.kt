package io.support.mcp.web

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.support.mcp.Config
import io.support.mcp.db.SupportDb
import io.support.mcp.db.Ticket
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/** REST-эндпоинты для быстрого теста в браузере / curl. */
fun Route.supportRoutes(db: SupportDb) {
    get("/") {
        call.respondText(indexHtml(), ContentType.Text.Html)
    }

    route("/api") {
        get("/methods") {
            call.respondJson(
                buildJsonObject {
                    put("service", "support-mcp")
                    put(
                        "methods",
                        buildJsonArray {
                            add(buildJsonObject { put("name", "create_ticket"); put("description", "Создать тикет") })
                            add(buildJsonObject { put("name", "get_ticket"); put("description", "Тикет по номеру/id") })
                            add(buildJsonObject { put("name", "find_user"); put("description", "Покупатель по e-mail") })
                            add(buildJsonObject { put("name", "list_tickets_for_user"); put("description", "Тикеты покупателя") })
                            add(buildJsonObject { put("name", "search_tickets"); put("description", "Поиск по тикетам") })
                        },
                    )
                }.toString(),
            )
        }

        get("/users") {
            call.respondJson(
                buildJsonObject {
                    put(
                        "users",
                        buildJsonArray {
                            db.listUsers().forEach { u ->
                                add(buildJsonObject {
                                    put("id", u.id); put("email", u.email); put("name", u.name)
                                    put("phone", u.phone); put("plan", u.plan)
                                })
                            }
                        },
                    )
                }.toString(),
            )
        }

        get("/tickets") {
            call.respondJson(ticketsPayload(db.listTickets()))
        }

        post("/tickets") {
            val body = runCatching { Json.parseToJsonElement(call.receiveText()).jsonObject }.getOrNull()
            val subject = body?.get("subject")?.jsonPrimitive?.contentOrNull
            val message = body?.get("message")?.jsonPrimitive?.contentOrNull
            val email = body?.get("customer_email")?.jsonPrimitive?.contentOrNull
            when {
                subject.isNullOrBlank() || message.isNullOrBlank() ->
                    call.respondError("Both 'subject' and 'message' are required.")
                else -> {
                    val t = db.createTicket(subject.trim(), message.trim(), email?.trim())
                    call.respondJson(ticketPayload(t))
                }
            }
        }

        get("/stats") {
            call.respondJson(
                buildJsonObject {
                    put("db_path", Config.dbPath)
                    put("users", db.countUsers())
                    put("tickets", db.countTickets())
                }.toString(),
            )
        }
    }
}

private fun ticketsPayload(tickets: List<Ticket>): String = buildJsonObject {
    put("count", tickets.size)
    put(
        "tickets",
        buildJsonArray {
            tickets.forEach { t ->
                add(buildJsonObject {
                    put("id", t.id); put("number", t.number); put("subject", t.subject)
                    put("status", t.status); put("priority", t.priority)
                    put("customer_id", t.customerId); put("created_at", t.createdAt)
                    put("first_message", t.firstMessage)
                })
            }
        },
    )
}.toString()

private fun ticketPayload(t: Ticket): String = buildJsonObject {
    put("id", t.id); put("number", t.number); put("subject", t.subject)
    put("status", t.status); put("priority", t.priority)
    put("customer_id", t.customerId); put("created_at", t.createdAt)
    put("first_message", t.firstMessage)
}.toString()

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
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🛠 Support MCP</title>
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
  <h1>🛠 Support MCP</h1>
  <p class="sub">Тикеты и покупатели · SQLite storage</p>
  <div class="card">
    <h2>Создать тикет</h2>
    <label for="subject">Тема</label>
    <input id="subject" type="text" placeholder="Не работает авторизация">
    <label for="message">Сообщение</label>
    <input id="message" type="text" placeholder="Не могу войти в личный кабинет">
    <label for="email">E-mail покупателя (необязательно)</label>
    <input id="email" type="text" placeholder="anna@velodom.example">
    <button onclick="createTicket()">create_ticket</button>
    <div class="result" id="r-create"></div>
  </div>
  <div class="card">
    <h2>Справочники</h2>
    <button onclick="get('/api/users', 'r-users')">Пользователи</button>
    <button onclick="get('/api/tickets', 'r-tickets')">Тикеты</button>
    <div class="result" id="r-users"></div>
    <div class="result" id="r-tickets"></div>
  </div>
  <p class="hint">MCP endpoint: <code>POST /mcp</code> (Streamable HTTP).</p>
</div>
<script>
function show(id, text) { document.getElementById(id).textContent = text; }
function get(url, id) {
  fetch(url).then(r => r.json()).then(d => show(id, JSON.stringify(d, null, 2))).catch(e => show(id, 'Error: ' + e));
}
function createTicket() {
  var body = JSON.stringify({
    subject: document.getElementById('subject').value.trim(),
    message: document.getElementById('message').value.trim(),
    customer_email: document.getElementById('email').value.trim() || null
  });
  fetch('/api/tickets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body })
    .then(r => r.json()).then(d => show('r-create', JSON.stringify(d, null, 2)))
    .catch(e => show('r-create', 'Error: ' + e));
}
</script>
</body>
</html>
""".trimIndent()
