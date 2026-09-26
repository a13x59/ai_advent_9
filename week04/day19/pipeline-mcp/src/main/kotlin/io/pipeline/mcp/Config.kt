package io.pipeline.mcp

/**
 * Configuration, read from environment variables (with sensible defaults).
 *
 * - PORT                 — HTTP/SSE port (default 8890, distinct from fortune 8888 and currency 8889).
 * - YANDEX_SEARCH_URL    — Yandex Search API base URL.
 * - YANDEX_API_KEY       — Yandex Search API key (sent as `Authorization: Api-Key <key>`).
 * - YANDEX_FOLDER_ID     — Yandex Cloud folder id (required by the Search API).
 * - SEARCH_MAX_RESULTS   — default number of results returned by `search`.
 * - DEEPSEEK_API_URL     — DeepSeek chat completions endpoint.
 * - DEEPSEEK_API_KEY     — DeepSeek API key for LLM summarization.
 * - DEEPSEEK_MODEL       — model used by `summarize`.
 * - SUMMARIZE_MAX_TOKENS — max tokens for the summarization completion.
 * - SUMMARIZE_TEMPERATURE— sampling temperature for the summarization completion.
 * - OUTPUT_DIR           — directory where `save_to_file` writes files.
 */
object Config {

    val port: Int = env("PORT", "8890").toIntOrNull() ?: 8890

    // Yandex Search API.
    val yandexSearchUrl: String = env("YANDEX_SEARCH_URL", "https://searchapi.yandex.ru/v1/web/search")
    val yandexApiKey: String = env("YANDEX_API_KEY", "")
    val yandexFolderId: String = env("YANDEX_FOLDER_ID", "")
    val searchMaxResults: Int = env("SEARCH_MAX_RESULTS", "5").toIntOrNull() ?: 5

    // DeepSeek LLM (summarize).
    val deepseekApiUrl: String = env("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    val deepseekApiKey: String = env("DEEPSEEK_API_KEY", "")
    val deepseekModel: String = env("DEEPSEEK_MODEL", "deepseek-chat")
    val summarizeMaxTokens: Int = env("SUMMARIZE_MAX_TOKENS", "512").toIntOrNull() ?: 512
    val summarizeTemperature: Double = env("SUMMARIZE_TEMPERATURE", "0.3").toDoubleOrNull() ?: 0.3

    // File output.
    val outputDir: String = env("OUTPUT_DIR", "output")

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default
}
