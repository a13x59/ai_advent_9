package io.currency.mcp.server

import io.currency.mcp.Config
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
import java.time.LocalDate
import java.time.format.DateTimeParseException

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
 */
fun Server.registerCurrencyTools(db: RateDb, summaries: SummaryService) {

    addTool(
        name = "get_rate",
        description = "Returns the latest EUR/USD exchange rate at the current time " +
            "(supported currencies: EUR and USD only).",
        inputSchema = ToolSchema(
            properties = buildJsonObject { },
            required = emptyList(),
        ),
    ) {
        val latest = db.latestRate()
        if (latest == null) {
            toolError("No EUR/USD rate available yet — the service has not stored any sample.")
        } else {
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
        description = "Returns the daily EUR/USD summary { count, avg, trend, last_n } for a given " +
            "date (YYYY-MM-DD).",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "date",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The date to summarize, in YYYY-MM-DD format.")
                    },
                )
            },
            required = listOf("date"),
        ),
    ) { request ->
        val date = request.arguments?.get("date")?.jsonPrimitive?.contentOrNull
        when {
            date.isNullOrBlank() ->
                toolError("The 'date' argument is required and must not be empty (YYYY-MM-DD).")
            !isValidDate(date) ->
                toolError("Invalid 'date' '$date'. Expected format: YYYY-MM-DD.")
            else ->
                toolText(summaryJson(summaries.ensureSummary(date)))
        }
    }
}

private fun isValidDate(date: String): Boolean = try {
    LocalDate.parse(date)
    true
} catch (e: DateTimeParseException) {
    false
}
