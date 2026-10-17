package com.guitartab.quality

import com.guitartab.model.Meta
import io.ktor.client.HttpClient
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
private data class ChatMessage(val role: String, val content: String)

@Serializable
private data class ResponseFormat(val type: String = "json_object")

@Serializable
private data class ChatRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val temperature: Double = 0.0,
    @SerialName("response_format") val responseFormat: ResponseFormat = ResponseFormat(),
)

@Serializable
private data class ChatChoice(val message: ChatMessage)

@Serializable
private data class ChatResponse(val choices: List<ChatChoice>)

@Serializable
data class Verdict(val ok: Boolean, val reason: String)

/** Клиент к OpenAI-совместимому API DeepSeek. */
class DeepSeekClient(
    private val client: HttpClient,
    private val apiKey: String,
    private val model: String,
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun evaluate(tab: String, meta: Meta): Verdict {
        val system = "Ты — эксперт по гитарной табулатуре. Оцени, выглядит ли табулатура " +
            "осмысленной и играбельной в стандартном строе EADGBE. Отвечай строго JSON-объектом " +
            "{\"ok\": true|false, \"reason\": \"краткое обоснование на русском\"}. \"ok\": false — если " +
            "табулатура явно бессмысленна (пустая, лады за пределами грифа, хаотичная раскладка)."
        val user = "Табулатура:\n$tab\n\nМетаданные: нот=${meta.noteCount}, " +
            "длительность=${meta.durationSec}с, строй=${meta.tuning}"

        val request = ChatRequest(model, listOf(ChatMessage("system", system), ChatMessage("user", user)))
        val resp = client.post("https://api.deepseek.com/chat/completions") {
            header(HttpHeaders.Authorization, "Bearer $apiKey")
            contentType(ContentType.Application.Json)
            setBody(json.encodeToString(ChatRequest.serializer(), request))
        }
        if (resp.status != HttpStatusCode.OK) {
            throw IllegalStateException("DeepSeek API error ${resp.status.value}: ${resp.bodyAsText().take(300)}")
        }
        val chat = json.decodeFromString(ChatResponse.serializer(), resp.bodyAsText())
        val content = chat.choices.firstOrNull()?.message?.content
            ?: throw IllegalStateException("Пустой ответ DeepSeek")
        return json.decodeFromString(Verdict.serializer(), content)
    }
}
