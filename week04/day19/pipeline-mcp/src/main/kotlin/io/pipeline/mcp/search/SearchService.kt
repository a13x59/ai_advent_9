package io.pipeline.mcp.search

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.parameter
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
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

/** A single normalized web-search hit. */
data class SearchResult(
    val title: String,
    val url: String,
    val passage: String,
)

/**
 * Fetches web-search results from the Yandex Search API and normalizes them into
 * `{ query, count, results: [{ title, url, passage }] }`.
 */
class SearchService {

    private val log = LoggerFactory.getLogger(SearchService::class.java)
    private val json = Json { ignoreUnknownKeys = true }
    private val client = HttpClient(CIO) { expectSuccess = false }

    suspend fun search(query: String, limit: Int = Config.searchMaxResults): String {
        if (Config.yandexApiKey.isBlank()) {
            throw IllegalStateException("YANDEX_API_KEY is not set — cannot perform web search.")
        }

        val response = client.get(Config.yandexSearchUrl) {
            parameter("query", query)
            parameter("page", "0")
            if (Config.yandexFolderId.isNotBlank()) {
                parameter("folderid", Config.yandexFolderId)
            }
            header(HttpHeaders.Authorization, "Api-Key ${Config.yandexApiKey}")
        }
        val body = response.bodyAsText()
        if (response.status != HttpStatusCode.OK) {
            throw IllegalStateException("Yandex Search API returned ${response.status.value}: $body")
        }

        val root = json.parseToJsonElement(body).jsonObject
        root["error"]?.jsonObject?.let { error ->
            val message = error["message"]?.jsonPrimitive?.contentOrNull ?: error.toString()
            throw IllegalStateException("Yandex Search API error: $message")
        }

        val results = root["results"]?.jsonArray
            ?: throw IllegalStateException("No 'results' field in Yandex response: $body")

        val items = results.mapNotNull { element ->
            val obj = element.jsonObject
            val title = obj["title"]?.jsonPrimitive?.contentOrNull ?: ""
            val url = obj["url"]?.jsonPrimitive?.contentOrNull ?: ""
            val passage = obj["passage"]?.jsonPrimitive?.contentOrNull
                ?: obj["snippet"]?.jsonPrimitive?.contentOrNull
                ?: obj["text"]?.jsonPrimitive?.contentOrNull
                ?: ""
            if (title.isBlank() && url.isBlank() && passage.isBlank()) null
            else SearchResult(title, url, passage)
        }.take(limit)

        log.info("search '{}' -> {} result(s)", query, items.size)
        return buildJsonObject {
            put("query", query)
            put("count", items.size)
            put(
                "results",
                buildJsonArray {
                    items.forEach { item ->
                        add(
                            buildJsonObject {
                                put("title", item.title)
                                put("url", item.url)
                                put("passage", item.passage)
                            },
                        )
                    }
                },
            )
        }.toString()
    }

    fun close() {
        client.close()
    }
}
