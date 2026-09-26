package io.pipeline.mcp.server

import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.ToolSchema
import io.pipeline.mcp.Config
import io.pipeline.mcp.save.SaveService
import io.pipeline.mcp.search.SearchService
import io.pipeline.mcp.summarize.SummarizeService
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory

private val log = LoggerFactory.getLogger("io.pipeline.mcp.server.McpTools")

private fun toolText(text: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(text)))

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

/**
 * Registers the three "pipeline" tools on the MCP [Server]:
 *
 *  - `search`       — FETCH data (web search via Yandex Search API);
 *  - `summarize`    — PROCESS data (LLM summarization via DeepSeek);
 *  - `save_to_file` — STORE the result (write a file to the output directory).
 *
 * Together they form the pipeline: search → summarize → save_to_file.
 */
fun Server.registerPipelineTools(
    searchService: SearchService,
    summarizeService: SummarizeService,
    saveService: SaveService,
) {
    addTool(
        name = "search",
        description = "Searches the web via the Yandex Search API and returns the top results " +
            "(title, url, passage) as a JSON object { query, count, results }. " +
            "Use this tool as the FIRST step of a pipeline to fetch data for a topic.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "query",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The search query.")
                    },
                )
                put(
                    "limit",
                    buildJsonObject {
                        put("type", "integer")
                        put("description", "Maximum number of results (default ${Config.searchMaxResults}).")
                    },
                )
            },
            required = listOf("query"),
        ),
    ) { request ->
        val query = request.arguments?.get("query")?.jsonPrimitive?.contentOrNull
        val limit = request.arguments?.get("limit")?.jsonPrimitive?.contentOrNull?.toIntOrNull()
            ?: Config.searchMaxResults
        if (query.isNullOrBlank()) {
            toolError("The 'query' argument is required and must not be empty.")
        } else {
            try {
                toolText(searchService.search(query.trim(), limit))
            } catch (e: Exception) {
                log.warn("search failed: {}", e.message)
                toolError("search failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "summarize",
        description = "Summarizes the given text using the DeepSeek LLM and returns a JSON object " +
            "{ summary, model, input_chars, usage }. Use this tool as the SECOND step of a pipeline " +
            "to process text produced by another tool (e.g. search).",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "text",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The text to summarize.")
                    },
                )
            },
            required = listOf("text"),
        ),
    ) { request ->
        val text = request.arguments?.get("text")?.jsonPrimitive?.contentOrNull
        if (text.isNullOrBlank()) {
            toolError("The 'text' argument is required and must not be empty.")
        } else {
            try {
                toolText(summarizeService.summarize(text))
            } catch (e: Exception) {
                log.warn("summarize failed: {}", e.message)
                toolError("summarize failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "save_to_file",
        description = "Writes the given text content to a file under the service output directory " +
            "and returns a JSON object { path, filename, bytes, preview }. " +
            "Use this tool as the FINAL step of a pipeline to store the result.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "filename",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "Output file name, e.g. summary.txt")
                    },
                )
                put(
                    "content",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "The text content to write.")
                    },
                )
            },
            required = listOf("filename", "content"),
        ),
    ) { request ->
        val filename = request.arguments?.get("filename")?.jsonPrimitive?.contentOrNull
        val content = request.arguments?.get("content")?.jsonPrimitive?.contentOrNull
        when {
            filename.isNullOrBlank() -> toolError("The 'filename' argument is required.")
            content == null -> toolError("The 'content' argument is required.")
            else -> try {
                toolText(saveService.saveToFile(filename, content))
            } catch (e: Exception) {
                log.warn("save_to_file failed: {}", e.message)
                toolError("save_to_file failed: ${e.message}")
            }
        }
    }
}
