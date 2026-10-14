package io.gitmcp.mcp.server

import io.gitmcp.mcp.git.GitService
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

private val log = LoggerFactory.getLogger("io.gitmcp.mcp.server.McpTools")

private fun toolText(text: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(text)))

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

/**
 * Регистрирует git-инструменты ассистента разработчика:
 *
 *  - `git_branch` — текущая ветка и список веток;
 *  - `list_files` — список отслеживаемых файлов проекта;
 *  - `git_diff`   — изменения рабочей копии (опционально staged / --stat).
 */
fun Server.registerGitTools(gitService: GitService) {
    addTool(
        name = "git_branch",
        description = "Returns the current git branch and the list of local branches " +
            "of the project repository. Useful to know which branch the project is on.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {},
            required = emptyList(),
        ),
    ) {
        try {
            val current = gitService.currentBranch()
            val branches = gitService.branches()
            val json = buildJsonObject {
                put("current_branch", current)
                put(
                    "branches",
                    buildJsonArray { branches.forEach { add(it) } },
                )
            }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("git_branch failed: {}", e.message)
            toolError("git_branch failed: ${e.message}")
        }
    }

    addTool(
        name = "list_files",
        description = "Lists the files of the project (git-tracked) under the given path. " +
            "Returns JSON { count, truncated, files }.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "path",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "Path inside the project to list (default \".\").")
                    },
                )
                put(
                    "max_files",
                    buildJsonObject {
                        put("type", "integer")
                        put("description", "Maximum number of files to return (default 200).")
                    },
                )
            },
            required = emptyList(),
        ),
    ) { request ->
        val path = request.arguments?.get("path")?.jsonPrimitive?.contentOrNull?.ifBlank { "." } ?: "."
        val maxFiles = request.arguments?.get("max_files")?.jsonPrimitive?.contentOrNull?.toIntOrNull()
            ?: 200
        try {
            val (count, files) = gitService.listFiles(path, maxFiles)
            val json = buildJsonObject {
                put("count", count)
                put("truncated", count > files.size)
                put("files", buildJsonArray { files.forEach { add(it) } })
            }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("list_files failed: {}", e.message)
            toolError("list_files failed: ${e.message}")
        }
    }

    addTool(
        name = "git_diff",
        description = "Returns the working-tree diff of the project (or the staged diff when " +
            "staged=true). Use stat=true for a summary. Returns plain diff text.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put(
                    "staged",
                    buildJsonObject {
                        put("type", "boolean")
                        put("description", "Show staged changes (git diff --cached) instead of working tree.")
                    },
                )
                put(
                    "stat",
                    buildJsonObject {
                        put("type", "boolean")
                        put("description", "Return a diffstat summary instead of the full diff.")
                    },
                )
                put(
                    "path",
                    buildJsonObject {
                        put("type", "string")
                        put("description", "Path inside the project to diff (default \".\").")
                    },
                )
            },
            required = emptyList(),
        ),
    ) { request ->
        val staged = request.arguments?.get("staged")?.jsonPrimitive?.contentOrNull?.toBooleanStrictOrNull()
            ?: false
        val stat = request.arguments?.get("stat")?.jsonPrimitive?.contentOrNull?.toBooleanStrictOrNull()
            ?: false
        val path = request.arguments?.get("path")?.jsonPrimitive?.contentOrNull?.ifBlank { "." } ?: "."
        try {
            val diff = gitService.diff(staged, stat, path)
            toolText(diff.ifBlank { "No changes (working tree clean)." })
        } catch (e: Exception) {
            log.warn("git_diff failed: {}", e.message)
            toolError("git_diff failed: ${e.message}")
        }
    }
}
