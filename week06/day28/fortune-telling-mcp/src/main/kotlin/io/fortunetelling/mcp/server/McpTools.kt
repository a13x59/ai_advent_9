package io.fortunetelling.mcp.server

import io.fortunetelling.mcp.fortune.Bibliomancy
import io.fortunetelling.mcp.fortune.Book
import io.fortunetelling.mcp.fortune.Line
import io.fortunetelling.mcp.fortune.Magic8Ball
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

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

/**
 * Registers the two "fortune telling" tools on the MCP [Server].
 */
fun Server.registerFortuneTools() {
    addTool(
        name = "magic_8_ball",
        description = "Answers a yes/no question by consulting the classic Magic 8 Ball. " +
            "Returns the prediction in the format: ❓<question> 🔮 <random answer>.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "question",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The yes/no question you want the Magic 8 Ball to answer.")
                    },
                )
            },
            required = listOf("question"),
        ),
    ) { request ->
        val question = request.arguments?.get("question")?.jsonPrimitive?.contentOrNull
        if (question.isNullOrBlank()) {
            CallToolResult(
                content = listOf(TextContent("The 'question' argument is required and must not be empty.")),
                isError = true,
            )
        } else {
            CallToolResult(content = listOf(TextContent(Magic8Ball.predict(question).text)))
        }
    }

    addTool(
        name = "bibliomancy",
        description = "Divines an answer by opening a random page of a chosen book and reading a passage " +
            "from the top or the bottom of the page. Returns the prediction in the format: " +
            "❓Question: <question> 📚 From book: <title> 📄 Page: <page> 📖 Answer: <passage>.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "question",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The question you want answered.")
                    },
                )
                put(
                    "book",
                    buildJsonObject {
                        put("type", "string")
                        put(
                            "enum",
                            buildJsonArray {
                                Book.entries.forEach { add(it.id) }
                            },
                        )
                        put("description", "The book to divine from. One of: ${Book.entries.joinToString(", ") { it.id }}.")
                    },
                )
                put(
                    "line",
                    buildJsonObject {
                        put("type", "string")
                        put(
                            "enum",
                            buildJsonArray {
                                Line.entries.forEach { add(it.id) }
                            },
                        )
                        put(
                            "description",
                            "Whether to read the passage from the top or the bottom of the page. " +
                                "One of: ${Line.entries.joinToString(", ") { it.id }}.",
                        )
                    },
                )
            },
            required = listOf("question", "book", "line"),
        ),
    ) { request ->
        val question = request.arguments?.get("question")?.jsonPrimitive?.contentOrNull
        val bookId = request.arguments?.get("book")?.jsonPrimitive?.contentOrNull
        val lineId = request.arguments?.get("line")?.jsonPrimitive?.contentOrNull
        val book = Book.fromId(bookId)
        val line = Line.fromId(lineId)

        when {
            question.isNullOrBlank() ->
                toolError("The 'question' argument is required and must not be empty.")
            book == null ->
                toolError("Unknown book '$bookId'. Valid values: ${Book.entries.joinToString(", ") { it.id }}.")
            line == null ->
                toolError("Unknown line '$lineId'. Valid values: ${Line.entries.joinToString(", ") { it.id }}.")
            else ->
                CallToolResult(content = listOf(TextContent(Bibliomancy.predict(question, book, line).text)))
        }
    }
}
