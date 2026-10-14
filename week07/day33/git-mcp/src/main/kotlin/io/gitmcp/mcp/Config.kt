package io.gitmcp.mcp

/**
 * Конфигурация git-MCP, читается из переменных окружения (с дефолтами).
 *
 * - PORT             — порт сервиса (по умолчанию 8892).
 * - GIT_PROJECT_PATH — каталог проекта, к которому скоупятся git-операции.
 *   По умолчанию — рабочий каталог сервиса (запуск из папки проекта).
 * - GIT_BIN          — путь к исполняемому файлу git (по умолчанию "git").
 */
object Config {

    val port: Int = env("PORT", "8892").toIntOrNull() ?: 8892
    val projectPath: String = env("GIT_PROJECT_PATH", System.getProperty("user.dir"))
    val gitBin: String = env("GIT_BIN", "git")

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default
}
