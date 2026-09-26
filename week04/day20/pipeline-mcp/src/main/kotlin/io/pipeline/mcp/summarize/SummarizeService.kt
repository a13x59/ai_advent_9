package io.pipeline.mcp.summarize

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.pipeline.mcp.Config
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory

/**
 * Summarizes text by calling the DeepSeek chat-completions API and returns
 * `{ summary, model, input_chars, usage }`.
 */
class SummarizeService {

    private val log = LoggerFactory.getLogger(SummarizeService::class.java)
    private val json = Json { ignoreUnknownKeys = true }
    private val client = HttpClient(CIO) { expectSuccess = false }

    suspend fun summarize(text: String): String {
        if (Config.deepseekApiKey.isBlank()) {
            throw IllegalStateException("DEEPSEEK_API_KEY is not set — cannot summarize.")
        }

        val requestBody = buildJsonObject {
            put("model", Config.deepseekModel)
            put("temperature", Config.summarizeTemperature)
            put("max_tokens", Config.summarizeMaxTokens)
            put("stream", false)
            put(
                "messages",
                buildJsonArray {
                    add(
                        buildJsonObject {
                            put("role", "system")
                            put(
                                "content",
                                "Ты — ассистент, который сжимает текст в короткую информативную сводку. " +
                                    "Выдели главное, сохрани факты и ключевые детали. " +
                                    "Отвечай на языке исходного текста.",
                            )
                        },
                    )
                    add(
                        buildJsonObject {
                            put("role", "user")
                            put("content", text)
                        },
                    )
                },
            )
        }.toString()

        val response = client.post(Config.deepseekApiUrl) {
            contentType(ContentType.Application.Json)
            header(HttpHeaders.Authorization, "Bearer ${Config.deepseekApiKey}")
            setBody(requestBody)
        }
        val respText = response.bodyAsText()
        if (response.status != HttpStatusCode.OK) {
            throw IllegalStateException("DeepSeek API returned ${response.status.value}: $respText")
        }

        val root = json.parseToJsonElement(respText).jsonObject
        val summary = root["choices"]?.jsonArray?.firstOrNull()?.jsonObject
            ?.get("message")?.jsonObject?.get("content")?.jsonPrimitive?.contentOrNull
            ?: throw IllegalStateException("Unexpected DeepSeek response: $respText")

        val usage = root["usage"]?.jsonObject
        log.info("summarized {} chars -> {} chars", text.length, summary.length)
        return buildJsonObject {
            put("summary", summary)
            put("model", Config.deepseekModel)
            put("input_chars", text.length)
            put(
                "usage",
                buildJsonObject {
                    put("prompt_tokens", usage?.get("prompt_tokens")?.jsonPrimitive?.contentOrNull ?: "")
                    put("completion_tokens", usage?.get("completion_tokens")?.jsonPrimitive?.contentOrNull ?: "")
                },
            )
        }.toString()
    }

    fun close() {
        client.close()
    }
}
