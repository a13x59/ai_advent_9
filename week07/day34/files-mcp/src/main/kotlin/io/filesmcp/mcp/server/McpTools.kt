package io.filesmcp.mcp.server

import io.filesmcp.mcp.files.FileService
import io.modelcontextprotocol.kotlin.sdk.server.Server
import io.modelcontextprotocol.kotlin.sdk.types.CallToolRequest
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

private val log = LoggerFactory.getLogger("io.filesmcp.mcp.server.McpTools")

private fun toolText(text: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(text)))

private fun toolError(message: String): CallToolResult =
    CallToolResult(content = listOf(TextContent(message)), isError = true)

private fun strArg(request: CallToolRequest, name: String): String? =
    request.arguments?.get(name)?.jsonPrimitive?.contentOrNull

private fun intArg(request: CallToolRequest, name: String): Int? =
    request.arguments?.get(name)?.jsonPrimitive?.contentOrNull?.toIntOrNull()

private fun boolArg(request: CallToolRequest, name: String): Boolean =
    request.arguments?.get(name)?.jsonPrimitive?.contentOrNull?.toBooleanStrictOrNull() ?: false

/**
 * Регистрирует файловые инструменты ассистента:
 *
 *  - `list_dir`        — список файлов подкаталога (файловая система);
 *  - `read_file`       — чтение файла (целиком или диапазон строк);
 *  - `search_files`    — grep по нескольким файлам;
 *  - `write_file`      — создать/перезаписать файл (в pending, не на диск);
 *  - `edit_file`       — точечная замена строки (в pending);
 *  - `diff`            — unified diff pending-изменений;
 *  - `apply_changes`   — сохранить pending-изменения на диск;
 *  - `discard_changes` — отбросить pending-изменения.
 */
fun Server.registerFileTools(fileService: FileService) {

    addTool(
        name = "list_dir",
        description = "Lists files under the given project path (filesystem walk, " +
            "includes untracked files). Returns JSON { count, truncated, files }.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("path", buildJsonObject {
                    put("type", "string")
                    put("description", "Path inside the project to list (default \".\").")
                })
                put("max_files", buildJsonObject {
                    put("type", "integer")
                    put("description", "Maximum number of files to return (default 200, 0 = unlimited).")
                })
            },
            required = emptyList(),
        ),
    ) { request ->
        val path = strArg(request, "path")?.ifBlank { "." } ?: "."
        val maxFiles = intArg(request, "max_files") ?: 200
        try {
            val (count, files) = fileService.listDir(path, maxFiles)
            val json = buildJsonObject {
                put("count", count)
                put("truncated", count > files.size)
                put("files", buildJsonArray { files.forEach { add(it) } })
            }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("list_dir failed: {}", e.message)
            toolError("list_dir failed: ${e.message}")
        }
    }

    addTool(
        name = "read_file",
        description = "Reads a project file (whole file or a line range). " +
            "Returns JSON { path, total_lines, start_line, end_line, truncated, content }.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("path", buildJsonObject {
                    put("type", "string")
                    put("description", "Project-relative path to the file.")
                })
                put("start_line", buildJsonObject {
                    put("type", "integer")
                    put("description", "First line to read (1-based, default 1).")
                })
                put("end_line", buildJsonObject {
                    put("type", "integer")
                    put("description", "Last line to read (1-based, default = last line).")
                })
            },
            required = listOf("path"),
        ),
    ) { request ->
        val path = strArg(request, "path")?.trim()
        if (path.isNullOrEmpty()) {
            toolError("read_file: 'path' is required")
        } else {
            try {
                val r = fileService.readFile(path, intArg(request, "start_line"), intArg(request, "end_line"))
                val json = buildJsonObject {
                    put("path", r.path)
                    put("total_lines", r.totalLines)
                    put("start_line", r.startLine)
                    put("end_line", r.endLine)
                    put("truncated", r.truncated)
                    put("content", r.content)
                }
                toolText(json.toString())
            } catch (e: Exception) {
                log.warn("read_file failed: {}", e.message)
                toolError("read_file failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "search_files",
        description = "Searches across project files (grep) for a query (substring, " +
            "case-insensitive by default; or regex when regex=true). Returns " +
            "{ query, count, truncated, matches: [{ file, line, text }] }.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("query", buildJsonObject {
                    put("type", "string")
                    put("description", "Text or regex to search for.")
                })
                put("path", buildJsonObject {
                    put("type", "string")
                    put("description", "Directory to search in (default \".\").")
                })
                put("glob", buildJsonObject {
                    put("type", "string")
                    put("description", "Filename glob filter, e.g. \"*.py\" (default \"*\" = all).")
                })
                put("regex", buildJsonObject {
                    put("type", "boolean")
                    put("description", "Treat query as a regex (default false).")
                })
                put("max_results", buildJsonObject {
                    put("type", "integer")
                    put("description", "Maximum number of matches (default 50).")
                })
            },
            required = listOf("query"),
        ),
    ) { request ->
        val query = strArg(request, "query")?.trim()
        if (query.isNullOrEmpty()) {
            toolError("search_files: 'query' is required")
        } else {
            try {
                val r = fileService.searchFiles(
                    query,
                    strArg(request, "path")?.ifBlank { "." } ?: ".",
                    strArg(request, "glob")?.ifBlank { "*" } ?: "*",
                    boolArg(request, "regex"),
                    intArg(request, "max_results") ?: 50,
                )
                val json = buildJsonObject {
                    put("query", r.query)
                    put("count", r.count)
                    put("truncated", r.truncated)
                    put("matches", buildJsonArray {
                        r.matches.forEach {
                            add(buildJsonObject {
                                put("file", it.file)
                                put("line", it.line)
                                put("text", it.text)
                            })
                        }
                    })
                }
                toolText(json.toString())
            } catch (e: Exception) {
                log.warn("search_files failed: {}", e.message)
                toolError("search_files failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "write_file",
        description = "Creates or overwrites a project file. Does NOT write to disk: " +
            "the change is staged and a unified diff is returned. Use apply_changes to save.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("path", buildJsonObject {
                    put("type", "string")
                    put("description", "Project-relative path to write.")
                })
                put("content", buildJsonObject {
                    put("type", "string")
                    put("description", "Full new content of the file.")
                })
            },
            required = listOf("path", "content"),
        ),
    ) { request ->
        val path = strArg(request, "path")?.trim()
        val content = strArg(request, "content")
        if (path.isNullOrEmpty() || content == null) {
            toolError("write_file: 'path' and 'content' are required")
        } else {
            try {
                val r = fileService.writeFile(path, content)
                toolText(changeJson(r).toString())
            } catch (e: Exception) {
                log.warn("write_file failed: {}", e.message)
                toolError("write_file failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "edit_file",
        description = "Performs a targeted edit: replaces old_string with new_string in a " +
            "project file. Does NOT write to disk: the change is staged and a unified diff " +
            "is returned. Use apply_changes to save. Fails if old_string occurs 0 or, when " +
            "replace_all=false, more than 1 time.",
        inputSchema = ToolSchema(
            properties = buildJsonObject {
                put("path", buildJsonObject {
                    put("type", "string")
                    put("description", "Project-relative path to edit.")
                })
                put("old_string", buildJsonObject {
                    put("type", "string")
                    put("description", "Exact text to replace.")
                })
                put("new_string", buildJsonObject {
                    put("type", "string")
                    put("description", "Replacement text.")
                })
                put("replace_all", buildJsonObject {
                    put("type", "boolean")
                    put("description", "Replace all occurrences (default false).")
                })
            },
            required = listOf("path", "old_string", "new_string"),
        ),
    ) { request ->
        val path = strArg(request, "path")?.trim()
        val old = strArg(request, "old_string")
        val new = strArg(request, "new_string")
        if (path.isNullOrEmpty() || old == null || new == null) {
            toolError("edit_file: 'path', 'old_string' and 'new_string' are required")
        } else {
            try {
                val r = fileService.editFile(path, old, new, boolArg(request, "replace_all"))
                toolText(changeJson(r).toString())
            } catch (e: Exception) {
                log.warn("edit_file failed: {}", e.message)
                toolError("edit_file failed: ${e.message}")
            }
        }
    }

    addTool(
        name = "diff",
        description = "Returns the unified diff of all staged (not yet applied) changes. " +
            "Nothing is written to disk.",
        inputSchema = ToolSchema(properties = buildJsonObject {}, required = emptyList()),
    ) {
        try {
            val json = buildJsonObject { put("diff", fileService.diff()) }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("diff failed: {}", e.message)
            toolError("diff failed: ${e.message}")
        }
    }

    addTool(
        name = "apply_changes",
        description = "Saves all staged changes to disk and clears the staging area. " +
            "Returns { applied: [paths], diff }.",
        inputSchema = ToolSchema(properties = buildJsonObject {}, required = emptyList()),
    ) {
        try {
            val r = fileService.applyChanges()
            val json = buildJsonObject {
                put("applied", buildJsonArray { r.applied.forEach { add(it) } })
                put("diff", r.diff)
            }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("apply_changes failed: {}", e.message)
            toolError("apply_changes failed: ${e.message}")
        }
    }

    addTool(
        name = "discard_changes",
        description = "Discards all staged changes without writing anything to disk. " +
            "Returns { discarded: N }.",
        inputSchema = ToolSchema(properties = buildJsonObject {}, required = emptyList()),
    ) {
        try {
            val json = buildJsonObject { put("discarded", fileService.discardChanges()) }
            toolText(json.toString())
        } catch (e: Exception) {
            log.warn("discard_changes failed: {}", e.message)
            toolError("discard_changes failed: ${e.message}")
        }
    }
}

private fun changeJson(r: FileService.ChangeResult) = buildJsonObject {
    put("path", r.path)
    put("status", r.status)
    r.occurrences?.let { put("occurrences", it) }
    put("diff", r.diff)
}
