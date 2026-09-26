package io.pipeline.mcp.save

import io.pipeline.mcp.Config
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths

/**
 * Writes text content to files under [Config.outputDir] and returns
 * `{ path, filename, bytes, preview }`. The file name is sanitized and path
 * traversal outside the output directory is rejected.
 */
class SaveService {

    private val log = LoggerFactory.getLogger(SaveService::class.java)

    fun saveToFile(filename: String, content: String): String {
        val safeName = sanitize(filename)
        if (safeName.isBlank()) {
            throw IllegalArgumentException("Filename must not be empty.")
        }
        val dir = Paths.get(Config.outputDir).toAbsolutePath().normalize()
        Files.createDirectories(dir)
        val target = dir.resolve(safeName).normalize()
        if (!target.startsWith(dir)) {
            throw IllegalArgumentException("Filename must stay inside the output directory.")
        }

        val bytes = content.toByteArray(StandardCharsets.UTF_8)
        Files.write(target, bytes)
        log.info("saved {} bytes to {}", bytes.size, target)

        return buildJsonObject {
            put("path", target.toString())
            put("filename", safeName)
            put("bytes", bytes.size)
            put("preview", content.take(200))
        }.toString()
    }

    fun listFiles(): List<String> {
        val dir = Paths.get(Config.outputDir).toAbsolutePath().normalize()
        if (!Files.isDirectory(dir)) return emptyList()
        return Files.list(dir).use { stream ->
            stream.map { it.fileName.toString() }.sorted().toList()
        }
    }

    private fun sanitize(name: String): String =
        name.trim().replace(Regex("[^A-Za-z0-9._-]"), "_")
}
