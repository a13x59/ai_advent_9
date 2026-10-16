package io.filesmcp.mcp

/**
 * Конфигурация files-MCP, читается из переменных окружения (с дефолтами).
 *
 * - PORT                 — порт сервиса (по умолчанию 8894, не пересекается
 *                          с fortune 8888 / currency 8889 / pipeline 8890 /
 *                          git 8892 / support 8893).
 * - FILES_ROOT           — корень, к которому скоупятся файловые операции.
 *                          По умолчанию — РОДИТЕЛЬСКИЙ каталог сервиса, т.е.
 *                          при запуске из files-mcp/ это week07/day34 (корень
 *                          проекта с файлами текущего дня). Запись/чтение за
 *                          пределами корня и path traversal запрещены.
 * - FILES_MAX_READ_CHARS — лимит символов при чтении файла (default 20000).
 * - FILES_MAX_SEARCH_RESULTS — лимит результатов grep (default 100).
 */
object Config {

    val port: Int = env("PORT", "8894").toIntOrNull() ?: 8894
    val rootPath: String = env("FILES_ROOT", defaultRoot())
    val maxReadChars: Int = env("FILES_MAX_READ_CHARS", "20000").toIntOrNull() ?: 20000
    val maxSearchResults: Int = env("FILES_MAX_SEARCH_RESULTS", "100").toIntOrNull() ?: 100

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default

    /** По умолчанию скоуп = родительский каталог сервиса (week07/day34). */
    private fun defaultRoot(): String {
        val cwd = java.nio.file.Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize()
        return cwd.parent?.toString() ?: cwd.toString()
    }
}
