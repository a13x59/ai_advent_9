package io.currency.mcp.server

import io.currency.mcp.db.RateDb
import io.currency.mcp.rates.Summary
import io.currency.mcp.rates.SummaryService
import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory
import java.time.LocalDate
import java.time.format.DateTimeParseException

private val log = LoggerFactory.getLogger("io.currency.mcp.server.McpTools")

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

private fun toolText(text: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(text)))

/** Serializes a [Summary] into the `{ count, avg, trend, last_n }` JSON object. */
fun summaryJson(summary: Summary): String = buildJsonObject {
    put("date", summary.date)
    put("count", summary.count)
    put("avg", summary.avgRounded)
    put("trend", summary.trend)
    put(
        "last_n",
        buildJsonArray {
            summary.lastN.forEach { item ->
                add(buildJsonObject { put("rate", item.rate); put("time", item.time) })
            }
        },
    )
}.toString()

/**
 * Registers the two currency tools on the MCP [Server]:
 *  - get_rate    — current EUR/USD rate;
 *  - get_summary — daily summary { count, avg, trend, last_n } for a given date.
 *
 * Each invocation is logged so it is visible in the service output when the agent
 * (or MCP Inspector) calls a tool.
 */
fun Server.registerCurrencyTools(db: RateDb, summaries: SummaryService) {

    addTool(
        name = "get_rate",
        description = "Возвращает текущий курс EUR/USD (евро к доллару США) на настоящий момент. " +
            "Используй этот инструмент, когда пользователь спрашивает про курс валют, например: " +
            "'какой курс евро к доллару', 'выведи USD/EUR', 'сколько стоит евро в долларах', " +
            "'текущий курс доллара'. Поддерживаются только валюты EUR и USD.",
        inputSchema = ToolSchema(
            properties = buildJsonObject { },
            required = emptyList(),
        ),
    ) { request ->
        log.info("tool 'get_rate' invoked, args: {}", request.arguments?.toString() ?: "{}")
        val latest = db.latestRate()
        if (latest == null) {
            log.info("tool 'get_rate' -> no data available yet")
            toolError("No EUR/USD rate available yet — the service has not stored any sample.")
        } else {
            log.info(
                "tool 'get_rate' -> source={}, target={}, rate={}, time={}",
                latest.source, latest.target, latest.rate, latest.time,
            )
            toolText(
                buildJsonObject {
                    put("source", latest.source)
                    put("target", latest.target)
                    put("rate", latest.rate)
                    put("time", latest.time)
                }.toString(),
            )
        }
    }

    addTool(
        name = "get_summary",
        description = "Возвращает дневную сводку по курсу EUR/USD за указанную дату в формате " +
            "{ count, avg, trend, last_n }: число котировок, средний курс, тренд (up/down/flat) " +
            "и последние N значений. Используй, когда нужна статистика или динамика курса за день.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "date",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "Дата сводки в формате YYYY-MM-DD.")
                    },
                )
            },
            required = listOf("date"),
        ),
    ) { request ->
        val date = request.arguments?.get("date")?.jsonPrimitive?.contentOrNull
        log.info("tool 'get_summary' invoked, args: {}", request.arguments?.toString() ?: "{}")
        when {
            date.isNullOrBlank() -> {
                log.info("tool 'get_summary' -> missing 'date' argument")
                toolError("The 'date' argument is required and must not be empty (YYYY-MM-DD).")
            }
            !isValidDate(date) -> {
                log.info("tool 'get_summary' -> invalid date '{}'", date)
                toolError("Invalid 'date' '$date'. Expected format: YYYY-MM-DD.")
            }
            else -> {
                val summary = summaries.ensureSummary(date)
                log.info(
                    "tool 'get_summary' -> date={}, count={}, avg={}, trend={}",
                    date, summary.count, summary.avgRounded, summary.trend,
                )
                toolText(summaryJson(summary))
            }
        }
    }
}

private fun isValidDate(date: String): Boolean = try {
    LocalDate.parse(date)
    true
} catch (e: DateTimeParseException) {
    false
}
