package io.support.mcp

/**
 * Конфигурация сервиса поддержки (MCP), читается из переменных окружения.
 *
 * - PORT        — HTTP-порт (default 8893, не пересекается с fortune 8888 /
 *                 currency 8889 / pipeline 8890 / git 8892).
 * - DB_PATH     — путь к SQLite-файлу (default: support.db).
 * - SEED_DATA   — заполнять ли БД демо-данными при пустой таблице (default: true).
 */
object Config {

    val port: Int = env("PORT", "8893").toIntOrNull() ?: 8893
    val dbPath: String = env("DB_PATH", "support.db")

    val seedData: Boolean = env("SEED_DATA", "true").lowercase() in setOf("1", "true", "yes", "on")

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default
}
