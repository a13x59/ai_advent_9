package com.guitartab

object Config {
    val port: Int = env("PORT", "8080")?.toIntOrNull() ?: 8080
    val workerUrl: String = env("WORKER_URL", "http://localhost:8000") ?: "http://localhost:8000"
    val deepSeekApiKey: String = env("DEEPSEEK_API_KEY", "") ?: ""
    val deepSeekModel: String = env("DEEPSEEK_MODEL", "deepseek-chat") ?: "deepseek-chat"
    val enableLlmGate: Boolean = (env("ENABLE_LLM_GATE", "true") ?: "true").toBoolean()
    val ffmpegPath: String = env("FFMPEG_PATH", "ffmpeg") ?: "ffmpeg"

    private fun env(name: String, default: String?): String? = System.getenv(name) ?: default
}
