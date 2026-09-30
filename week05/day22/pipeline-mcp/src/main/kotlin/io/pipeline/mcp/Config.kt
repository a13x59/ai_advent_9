package io.pipeline.mcp

/**
 * Configuration, read from environment variables (with defaults).
 *
 * Search (Yandex Search API v2, asynchronous: submit → poll → Base64-XML):
 *  - YANDEX_SEARCH_URL     — submit endpoint (web/searchAsync).
 *  - YANDEX_OPERATION_URL  — Cloud Operation endpoint used to poll the async result.
 *  - YANDEX_API_KEY        — service-account API key (`Authorization: Api-Key <key>`).
 *  - YANDEX_FOLDER_ID      — Yandex Cloud folder id.
 *  - YANDEX_SEARCH_TYPE    — SEARCH_TYPE_RU | SEARCH_TYPE_TR | SEARCH_TYPE_COM.
 *  - YANDEX_FAMILY_MODE    — FAMILY_MODE_MODERATE | FAMILY_MODE_NONE | FAMILY_MODE_STRICT.
 *  - YANDEX_L10N           — LOCALIZATION_RU | LOCALIZATION_EN | ...
 *  - YANDEX_POLL_INTERVAL_MS — delay between operation polls.
 *  - YANDEX_POLL_TIMEOUT_MS  — max time to wait for the async search result.
 *  - SEARCH_MAX_RESULTS    — default number of results returned by `search`.
 *
 * Summarize (DeepSeek LLM):
 *  - DEEPSEEK_API_URL / DEEPSEEK_API_KEY / DEEPSEEK_MODEL /
 *    SUMMARIZE_MAX_TOKENS / SUMMARIZE_TEMPERATURE.
 *
 * Files: OUTPUT_DIR.
 *
 * NOTE: for real projects put secrets into environment variables, not into code.
 */
object Config {

    val port: Int = env("PORT", "8890").toIntOrNull() ?: 8890

    // Yandex Search API v2.
    val yandexSearchUrl: String = env("YANDEX_SEARCH_URL", "https://searchapi.api.cloud.yandex.net/v2/web/searchAsync")
    val yandexOperationUrl: String = env("YANDEX_OPERATION_URL", "https://operation.api.cloud.yandex.net/operations")
    val yandexApiKey: String = env("YANDEX_API_KEY", "")
    val yandexFolderId: String = env("YANDEX_FOLDER_ID", "")
    val yandexSearchType: String = env("YANDEX_SEARCH_TYPE", "SEARCH_TYPE_RU")
    val yandexFamilyMode: String = env("YANDEX_FAMILY_MODE", "FAMILY_MODE_MODERATE")
    val yandexL10n: String = env("YANDEX_L10N", "LOCALIZATION_RU")
    val yandexPollIntervalMs: Long = env("YANDEX_POLL_INTERVAL_MS", "1000").toLongOrNull() ?: 1000L
    val yandexPollTimeoutMs: Long = env("YANDEX_POLL_TIMEOUT_MS", "60000").toLongOrNull() ?: 60000L
    val searchMaxResults: Int = env("SEARCH_MAX_RESULTS", "5").toIntOrNull() ?: 5

    // DeepSeek LLM (summarize).
    val deepseekApiUrl: String = env("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    val deepseekApiKey: String = env("DEEPSEEK_API_KEY", "sk-...")
    val deepseekModel: String = env("DEEPSEEK_MODEL", "deepseek-chat")
    val summarizeMaxTokens: Int = env("SUMMARIZE_MAX_TOKENS", "512").toIntOrNull() ?: 512
    val summarizeTemperature: Double = env("SUMMARIZE_TEMPERATURE", "0.3").toDoubleOrNull() ?: 0.3

    // File output.
    val outputDir: String = env("OUTPUT_DIR", "output")

    private fun env(name: String, default: String): String =
        System.getenv(name) ?: System.getProperty(name) ?: default
}
