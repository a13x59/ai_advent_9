package io.filesmcp.mcp.files

import io.filesmcp.mcp.Config
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/**
 * Ядро файлового ассистента: чтение, grep-поиск и «сухой» набор правок.
 *
 * Скоуп жёстко ограничен [rootPath]:
 *  - абсолютные пути и выход за корень (`..`) отвергаются на этапе resolve;
 *  - запись ничего не трогает на диске до явного [applyChanges] — так ассистент
 *    сначала показывает diff (dry-run), а сохраняет изменения только по команде.
 *
 * Модель правок:
 *  - [writeFile] / [editFile] кладут в [pending] «путь → новое содержимое»;
 *  - [diff] возвращает unified diff pending-изменений относительно диска;
 *  - [applyChanges] записывает pending на диск и очищает его;
 *  - [discardChanges] отбрасывает pending без записи.
 */
class FileService(
    private val rootPath: String = Config.rootPath,
) {

    class FileException(message: String) : RuntimeException(message)

    data class ReadResult(
        val path: String,
        val totalLines: Int,
        val startLine: Int,
        val endLine: Int,
        val content: String,
        val truncated: Boolean,
    )

    data class Match(val file: String, val line: Int, val text: String)

    data class SearchResult(
        val query: String,
        val count: Int,
        val truncated: Boolean,
        val matches: List<Match>,
    )

    data class ChangeResult(
        val path: String,
        val status: String,
        val occurrences: Int?,
        val diff: String,
    )

    data class ApplyResult(val applied: List<String>, val diff: String)

    private val root: Path = Paths.get(rootPath).toAbsolutePath().normalize()
    private val pending = LinkedHashMap<String, String>()

    /** Служебные каталоги, которые не относятся к исходникам проекта. */
    private val skipDirs = setOf(
        ".git", ".venv", "venv", "build", ".gradle", "node_modules", "__pycache__",
        ".idea", ".kotlin", "out", "dist", "target",
    )

    private fun shouldSkip(p: Path): Boolean =
        root.relativize(p).any { it.toString() in skipDirs }

    /** Резолвит относительный путь внутри корня и отвергает выход за его пределы. */
    fun resolve(relative: String): Path {
        val p = Paths.get(relative)
        if (p.isAbsolute) throw FileException("absolute paths are not allowed: $relative")
        val normalized = root.resolve(p).normalize()
        if (!normalized.startsWith(root)) throw FileException("path escapes project root: $relative")
        return normalized
    }

    private fun rel(p: Path): String = root.relativize(p).toString().replace('\\', '/')

    fun pendingCount(): Int = pending.size

    /** Список файлов под [path] (файловая система, включая untracked). */
    fun listDir(path: String, maxFiles: Int): Pair<Int, List<String>> {
        val base = resolve(path)
        if (!Files.isDirectory(base)) throw FileException("not a directory: $path")
        val files = mutableListOf<String>()
        var count = 0
        Files.walk(base).use { stream ->
            for (p in stream) {
                if (!Files.isRegularFile(p) || shouldSkip(p)) continue
                count++
                if (maxFiles <= 0 || files.size < maxFiles) files.add(rel(p))
            }
        }
        return count to files
    }

    /** Читает файл целиком или диапазон строк [startLine]..[endLine]. */
    fun readFile(path: String, startLine: Int?, endLine: Int?): ReadResult {
        val resolved = resolve(path)
        if (!Files.isRegularFile(resolved)) throw FileException("file not found: $path")
        val all = readTextLines(resolved)
        val start = (startLine ?: 1).coerceAtLeast(1)
        val end = (endLine ?: all.size).coerceAtMost(all.size)
        val selected = if (start > all.size || start > end) emptyList() else all.subList(start - 1, end)
        var text = selected.joinToString("\n")
        val truncated = text.length > Config.maxReadChars
        if (truncated) text = text.take(Config.maxReadChars)
        return ReadResult(
            path = path,
            totalLines = all.size,
            startLine = start,
            endLine = if (selected.isEmpty()) start else end,
            content = text,
            truncated = truncated,
        )
    }

    /**
     * Grep-поиск по файлам под [path]: подстрока (по умолчанию, регистронезависимо)
     * или regex при [regex]=true. [glob] — фильтр имени файла (например `*.py`).
     */
    fun searchFiles(query: String, path: String, glob: String, regex: Boolean, maxResults: Int): SearchResult {
        if (query.isEmpty()) throw FileException("query must not be empty")
        val base = resolve(path)
        if (!Files.isDirectory(base)) throw FileException("not a directory: $path")
        val pattern = if (regex) {
            try {
                Regex(query, RegexOption.IGNORE_CASE)
            } catch (e: Exception) {
                throw FileException("bad regex: $query")
            }
        } else {
            null
        }
        val matcher = if (glob.isNotBlank() && glob != "*") {
            java.nio.file.FileSystems.getDefault().getPathMatcher("glob:$glob")
        } else {
            null
        }
        val matches = mutableListOf<Match>()
        var truncated = false
        Files.walk(base).use { stream ->
            for (p in stream) {
                if (matches.size >= maxResults) {
                    truncated = true
                    break
                }
                if (!Files.isRegularFile(p) || shouldSkip(p)) continue
                if (matcher != null && !matcher.matches(Paths.get(p.fileName.toString()))) continue
                val lines = try {
                    Files.readAllLines(p)
                } catch (e: Exception) {
                    continue  // бинарный/нечитаемый файл — пропускаем
                }
                for ((i, line) in lines.withIndex()) {
                    val hit = pattern?.containsMatchIn(line) ?: line.contains(query, ignoreCase = true)
                    if (hit) {
                        val text = if (line.length > 500) line.take(500) + "..." else line
                        matches.add(Match(rel(p), i + 1, text))
                        if (matches.size >= maxResults) {
                            truncated = true
                            break
                        }
                    }
                }
            }
        }
        return SearchResult(query, matches.size, truncated, matches)
    }

    /** Создаёт или перезаписывает файл: кладёт в pending, на диск НЕ пишет. */
    fun writeFile(path: String, content: String): ChangeResult {
        val resolved = resolve(path)
        val rel = rel(resolved)
        val existed = Files.exists(resolved)
        pending[rel] = content
        val old = if (existed) Files.readString(resolved) else null
        return ChangeResult(rel, if (existed) "modified" else "created", null, diffText(rel, old, content))
    }

    /** Точечная правка: заменяет [oldString] на [newString] (в pending). */
    fun editFile(path: String, oldString: String, newString: String, replaceAll: Boolean): ChangeResult {
        val resolved = resolve(path)
        val rel = rel(resolved)
        val before = currentContent(rel) ?: throw FileException("file not found: $path")
        if (oldString.isEmpty()) throw FileException("old_string must not be empty")
        val occurrences = countOccurrences(before, oldString)
        if (occurrences == 0) throw FileException("old_string not found in $path")
        if (!replaceAll && occurrences > 1) {
            throw FileException(
                "old_string occurs $occurrences times in $path; pass replace_all=true or a more specific old_string",
            )
        }
        val after = if (replaceAll) before.replace(oldString, newString) else before.replaceFirst(oldString, newString)
        pending[rel] = after
        return ChangeResult(rel, "edited", occurrences, diffText(rel, before, after))
    }

    /** Unified diff всех pending-изменений относительно диска (без записи). */
    fun diff(): String {
        if (pending.isEmpty()) return "No pending changes."
        val parts = mutableListOf<String>()
        for ((rel, newContent) in pending) {
            val resolved = resolve(rel)
            val old = if (Files.exists(resolved)) Files.readString(resolved) else null
            parts.add(diffText(rel, old, newContent))
        }
        return parts.joinToString("\n")
    }

    /** Применяет pending-изменения: записывает файлы и очищает pending. */
    fun applyChanges(): ApplyResult {
        if (pending.isEmpty()) return ApplyResult(emptyList(), "No pending changes.")
        val applied = mutableListOf<String>()
        val parts = mutableListOf<String>()
        for ((rel, newContent) in pending) {
            val resolved = resolve(rel)
            val old = if (Files.exists(resolved)) Files.readString(resolved) else null
            resolved.parent?.let { Files.createDirectories(it) }
            Files.writeString(resolved, newContent)
            applied.add(rel)
            parts.add(diffText(rel, old, newContent))
        }
        pending.clear()
        return ApplyResult(applied, parts.joinToString("\n"))
    }

    /** Отбрасывает pending-изменения без записи; возвращает количество отброшенных. */
    fun discardChanges(): Int {
        val n = pending.size
        pending.clear()
        return n
    }

    private fun readTextLines(p: Path): List<String> {
        return try {
            Files.readAllLines(p)
        } catch (e: java.nio.charset.MalformedInputException) {
            throw FileException("file is not UTF-8 text: ${rel(p)}")
        }
    }

    private fun currentContent(rel: String): String? {
        if (pending.containsKey(rel)) return pending[rel]
        val resolved = resolve(rel)
        return if (Files.exists(resolved)) Files.readString(resolved) else null
    }

    private fun countOccurrences(text: String, needle: String): Int {
        var count = 0
        var idx = text.indexOf(needle)
        while (idx >= 0) {
            count++
            idx = text.indexOf(needle, idx + needle.length)
        }
        return count
    }

    /** Простой unified diff (общий префикс/суффикс + средний блок). */
    private fun diffText(rel: String, old: String?, new: String): String {
        val oldLines = old?.lines() ?: emptyList()
        val newLines = new.lines()
        var prefix = 0
        while (prefix < oldLines.size && prefix < newLines.size && oldLines[prefix] == newLines[prefix]) prefix++
        var suffix = 0
        while (
            suffix < oldLines.size - prefix && suffix < newLines.size - prefix &&
            oldLines[oldLines.size - 1 - suffix] == newLines[newLines.size - 1 - suffix]
        ) suffix++

        val sb = StringBuilder()
        sb.append("--- a/").append(rel).append('\n')
        sb.append("+++ b/").append(rel).append('\n')

        val oldRemoved = oldLines.size - prefix - suffix
        val newAdded = newLines.size - prefix - suffix
        if (oldRemoved == 0 && newAdded == 0) {
            sb.append("@@ no changes @@")
            return sb.toString()
        }
        sb.append("@@ -").append(prefix + 1).append(',').append(oldRemoved)
            .append(" +").append(prefix + 1).append(',').append(newAdded).append(" @@\n")

        val ctx = 3
        val ctxBefore = maxOf(0, prefix - ctx)
        for (i in ctxBefore until prefix) sb.append(' ').append(oldLines[i]).append('\n')
        for (i in prefix until oldLines.size - suffix) sb.append('-').append(oldLines[i]).append('\n')
        for (i in prefix until newLines.size - suffix) sb.append('+').append(newLines[i]).append('\n')
        val ctxAfterStart = oldLines.size - suffix
        val ctxAfterEnd = minOf(oldLines.size, ctxAfterStart + ctx)
        for (i in ctxAfterStart until ctxAfterEnd) sb.append(' ').append(oldLines[i]).append('\n')
        return sb.toString().trimEnd()
    }
}
