package io.currency.mcp

/**
 * Configuration, read from environment variables (with sensible defaults).
 *
 * - PORT                 — HTTP/SSE port (default 8889, distinct from fortune-telling-mcp's 8888).
 * - DB_PATH              — SQLite file path (default: currency.db).
 * - POLL_INTERVAL_MS     — how often to fetch the live rate (default 120000 = 2 minutes).
 * - SUMMARY_INTERVAL_MS  — how often to pre-aggregate a daily summary (default 300000 = 5 minutes).
 * - RATES_API_URL        — base URL of the rates API.
 * - RATES_API_KEY        — bearer token for the rates API.
 * - SOURCE / TARGET      — supported currencies (USD / EUR only).
 * - LAST_N               — number of last rates kept in a summary's `last_n`.
 */
object Config {

    val port: Int = env("PORT", "8889").toIntOrNull() ?: 8889
    val dbPath: String = env("DB_PATH", "currency.db")

    val pollIntervalMs: Long = env("POLL_INTERVAL_MS", "120000").toLongOrNull() ?: 120000L
    val summaryIntervalMs: Long = env("SUMMARY_INTERVAL_MS", "300000").toLongOrNull() ?: 300000L

    val apiBaseUrl: String = env("RATES_API_URL", "https://allratestoday.com/api/v1/rates")
    val apiKey: String = env("RATES_API_KEY", "XXX")

    val source: String = env("SOURCE", "USD")
    val target: String = env("TARGET", "EUR")

    val lastN: Int = env("LAST_N", "5").toIntOrNull() ?: 5

    /** Supported currencies (requirement 3): EUR and USD. */
    val supportedCurrencies: Set<String> = setOf("EUR", "USD")

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default
}
