package io.support.mcp.server

import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import io.support.mcp.db.SupportDb
import io.support.mcp.db.Ticket
import io.support.mcp.db.User
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory

private val log = LoggerFactory.getLogger("io.support.mcp.server.McpTools")

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

private fun toolText(text: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(text)))

/** Сериализует покупателя в JSON. */
fun userJson(u: User): String = buildJsonObject {
    put("id", u.id)
    put("email", u.email)
    put("name", u.name)
    put("phone", u.phone)
    put("plan", u.plan)
}.toString()

/** Сериализует тикет (с опциональным покупателем) в JSON. */
fun ticketJson(t: Ticket, customer: User?): String = buildJsonObject {
    put("id", t.id)
    put("number", t.number)
    put("subject", t.subject)
    put("status", t.status)
    put("priority", t.priority)
    put("created_at", t.createdAt)
    put("first_message", t.firstMessage)
    if (customer != null) {
        put(
            "customer",
            buildJsonObject {
                put("id", customer.id)
                put("email", customer.email)
                put("name", customer.name)
                put("plan", customer.plan)
            },
        )
    }
}.toString()

private fun ticketsJson(tickets: List<Ticket>): String = buildJsonObject {
    put("count", tickets.size)
    put(
        "tickets",
        buildJsonArray {
            tickets.forEach { t ->
                add(
                    buildJsonObject {
                        put("id", t.id)
                        put("number", t.number)
                        put("subject", t.subject)
                        put("status", t.status)
                        put("priority", t.priority)
                        put("created_at", t.createdAt)
                        put("first_message", t.firstMessage)
                    },
                )
            }
        },
    )
}.toString()

/**
 * Регистрирует инструменты поддержки на MCP [Server]:
 *  - create_ticket         — создать тикет (write);
 *  - get_ticket            — получить тикет по номеру/id;
 *  - find_user             — найти покупателя по e-mail;
 *  - list_tickets_for_user — тикеты покупателя;
 *  - search_tickets        — поиск по тикетам.
 */
fun Server.registerSupportTools(db: SupportDb) {

    addTool(
        name = "create_ticket",
        description = "Создаёт новый тикет в системе поддержки и возвращает его номер (например T-1002). " +
            "Используй, когда пользователь задаёт вопрос о продукте: зарегистрируй обращение, чтобы " +
            "последующие сообщения велись в контексте этого тикета.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("subject", buildJsonObject {
                    put("type", "string")
                    put("description", "Короткая тема обращения (например 'Не работает авторизация').")
                })
                put("message", buildJsonObject {
                    put("type", "string")
                    put("description", "Первое сообщение клиента — его вопрос или описание проблемы.")
                })
                put("customer_email", buildJsonObject {
                    put("type", "string")
                    put("description", "E-mail покупателя (необязательно); если указан, тикет привязывается к клиенту.")
                })
            },
            required = listOf("subject", "message"),
        ),
    ) { request ->
        val subject = request.arguments?.get("subject")?.jsonPrimitive?.contentOrNull
        val message = request.arguments?.get("message")?.jsonPrimitive?.contentOrNull
        val email = request.arguments?.get("customer_email")?.jsonPrimitive?.contentOrNull
        when {
            subject.isNullOrBlank() -> toolError("The 'subject' argument is required and must not be empty.")
            message.isNullOrBlank() -> toolError("The 'message' argument is required and must not be empty.")
            else -> {
                val ticket = db.createTicket(subject.trim(), message.trim(), email?.trim())
                val customer = ticket.customerId?.let { db.findUserById(it) }
                log.info("tool 'create_ticket' -> {}", ticket.number)
                toolText(ticketJson(ticket, customer))
            }
        }
    }

    addTool(
        name = "get_ticket",
        description = "Возвращает тикет поддержки по номеру (например 'T-1001') или числовому id вместе с " +
            "данными покупателя. Используй, когда пользователь указывает номер тикета, чтобы отвечать " +
            "с учётом контекста этого обращения.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("ticket_id", buildJsonObject {
                    put("type", "string")
                    put("description", "Номер тикета (T-1001) или числовой id.")
                })
            },
            required = listOf("ticket_id"),
        ),
    ) { request ->
        val ref = request.arguments?.get("ticket_id")?.jsonPrimitive?.contentOrNull
        if (ref.isNullOrBlank()) {
            toolError("The 'ticket_id' argument is required and must not be empty.")
        } else {
            val ticket = resolveTicket(db, ref.trim())
            if (ticket == null) {
                toolError("Тикет '$ref' не найден.")
            } else {
                val customer = ticket.customerId?.let { db.findUserById(it) }
                log.info("tool 'get_ticket' -> {}", ticket.number)
                toolText(ticketJson(ticket, customer))
            }
        }
    }

    addTool(
        name = "find_user",
        description = "Возвращает покупателя по e-mail: { id, email, name, phone, plan }. " +
            "Используй, чтобы получить контекст клиента перед ответом.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("email", buildJsonObject {
                    put("type", "string")
                    put("description", "E-mail покупателя.")
                })
            },
            required = listOf("email"),
        ),
    ) { request ->
        val email = request.arguments?.get("email")?.jsonPrimitive?.contentOrNull
        if (email.isNullOrBlank()) {
            toolError("The 'email' argument is required and must not be empty.")
        } else {
            val user = db.findUserByEmail(email.trim())
            if (user == null) toolError("Покупатель '$email' не найден.") else toolText(userJson(user))
        }
    }

    addTool(
        name = "list_tickets_for_user",
        description = "Возвращает список тикетов покупателя по e-mail. Используй для истории обращений клиента.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("email", buildJsonObject {
                    put("type", "string")
                    put("description", "E-mail покупателя.")
                })
            },
            required = listOf("email"),
        ),
    ) { request ->
        val email = request.arguments?.get("email")?.jsonPrimitive?.contentOrNull
        if (email.isNullOrBlank()) {
            toolError("The 'email' argument is required and must not be empty.")
        } else {
            val tickets = db.listTicketsForUser(email.trim())
            toolText(ticketsJson(tickets))
        }
    }

    addTool(
        name = "search_tickets",
        description = "Ищет тикеты по номеру, теме или первому сообщению (подстрока). " +
            "Используй, чтобы найти обращение, если пользователь не помнит точный номер.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("query", buildJsonObject {
                    put("type", "string")
                    put("description", "Поисковый запрос (подстрока).")
                })
            },
            required = listOf("query"),
        ),
    ) { request ->
        val query = request.arguments?.get("query")?.jsonPrimitive?.contentOrNull
        if (query.isNullOrBlank()) {
            toolError("The 'query' argument is required and must not be empty.")
        } else {
            toolText(ticketsJson(db.searchTickets(query.trim())))
        }
    }
}

private fun resolveTicket(db: SupportDb, ref: String): Ticket? =
    db.getTicketByNumber(ref) ?: ref.toLongOrNull()?.let { db.getTicketById(it) }
