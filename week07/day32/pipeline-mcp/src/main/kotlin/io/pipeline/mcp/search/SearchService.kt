package io.pipeline.mcp.search

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.pipeline.mcp.Config
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory
import org.w3c.dom.Element
import java.io.StringReader
import java.util.Base64
import javax.xml.parsers.DocumentBuilderFactory

/** A single normalized web-search hit. */
data class SearchResult(
    val title: String,
    val url: String,
    val passage: String,
)

/**
 * Web search via the Yandex Search API **v2** (asynchronous flow):
 *
 *  1. `POST /v2/web/searchAsync` → returns an operation id;
 *  2. poll `GET /operations/{id}` until `done: true`;
 *  3. the result is a Base64-encoded XML body in `response.rawData` → decode and parse.
 *
 * Returns normalized `{ query, count, results: [{ title, url, passage }] }`.
 */
class SearchService {

    private val log = LoggerFactory.getLogger(SearchService::class.java)
    private val json = Json { ignoreUnknownKeys = true }
    private val client = HttpClient(CIO) { expectSuccess = false }

    suspend fun search(query: String, limit: Int = Config.searchMaxResults): String {
        if (Config.yandexApiKey.isBlank()) {
            throw IllegalStateException("YANDEX_API_KEY is not set — cannot perform web search.")
        }
        if (Config.yandexFolderId.isBlank()) {
            throw IllegalStateException("YANDEX_FOLDER_ID is not set — cannot perform web search.")
        }

        val operationId = submit(query, limit)
        val rawData = poll(operationId)
        val xml = String(Base64.getMimeDecoder().decode(rawData.trim()), Charsets.UTF_8)
        val items = parseXmlResults(xml).take(limit)

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

    /** Submits the async search and returns the operation id. */
    private suspend fun submit(query: String, limit: Int): String {
        val body = buildJsonObject {
            put(
                "query",
                buildJsonObject {
                    put("searchType", Config.yandexSearchType)
                    put("queryText", query)
                    put("familyMode", Config.yandexFamilyMode)
                    put("page", "0")
                },
            )
            put(
                "groupSpec",
                buildJsonObject {
                    put("groupMode", "GROUP_MODE_DEEP")
                    put("groupsOnPage", limit)
                    put("docsInGroup", 1)
                },
            )
            put("maxPassages", 2)
            if (Config.yandexL10n.isNotBlank()) {
                put("l10N", Config.yandexL10n)
            }
            put("folderId", Config.yandexFolderId)
        }.toString()

        val response = client.post(Config.yandexSearchUrl) {
            contentType(ContentType.Application.Json)
            header(HttpHeaders.Authorization, "Api-Key ${Config.yandexApiKey}")
            setBody(body)
        }
        val respText = response.bodyAsText()
        if (response.status != HttpStatusCode.OK) {
            throw IllegalStateException("Yandex Search API returned ${response.status.value}: ${respText.take(500)}")
        }
        val root = json.parseToJsonElement(respText).jsonObject
        val id = root["id"]?.jsonPrimitive?.contentOrNull
            ?: throw IllegalStateException("No operation id in Yandex response: $respText")
        log.info("search async submitted, operation={}", id)
        return id
    }

    /** Polls the operation until done (or timeout), returns the Base64 `rawData`. */
    private suspend fun poll(operationId: String): String {
        val deadline = System.currentTimeMillis() + Config.yandexPollTimeoutMs
        var last = ""
        while (System.currentTimeMillis() < deadline) {
            val response = client.get("${Config.yandexOperationUrl}/$operationId") {
                header(HttpHeaders.Authorization, "Api-Key ${Config.yandexApiKey}")
            }
            val respText = response.bodyAsText()
            last = respText
            if (response.status == HttpStatusCode.OK) {
                val root = json.parseToJsonElement(respText).jsonObject
                root["error"]?.jsonObject?.let { err ->
                    throw IllegalStateException("Yandex search operation error: ${err.toString()}")
                }
                if (root["done"]?.jsonPrimitive?.contentOrNull == "true") {
                    return root["response"]?.jsonObject?.get("rawData")?.jsonPrimitive?.contentOrNull
                        ?: throw IllegalStateException("No rawData in completed operation: $respText")
                }
            }
            delay(Config.yandexPollIntervalMs)
        }
        throw IllegalStateException(
            "Yandex search timed out after ${Config.yandexPollTimeoutMs} ms. Last response: ${last.take(300)}",
        )
    }

    /** Parses the Yandex XML response into a list of normalized results. */
    private fun parseXmlResults(xml: String): List<SearchResult> {
        val document = DocumentBuilderFactory.newInstance().newDocumentBuilder()
            .parse(org.xml.sax.InputSource(StringReader(xml)))

        val errorNodes = document.getElementsByTagName("error")
        if (errorNodes.length > 0) {
            val code = (errorNodes.item(0) as? Element)?.getAttribute("code") ?: ""
            val message = errorNodes.item(0).textContent?.trim() ?: ""
            throw IllegalStateException("Yandex search XML error (code $code): $message")
        }

        val results = mutableListOf<SearchResult>()
        val docNodes = document.getElementsByTagName("doc")
        for (i in 0 until docNodes.length) {
            val el = docNodes.item(i) as? Element ?: continue
            val url = text(el, "url")
            val title = text(el, "title")
            val snippets = passages(el)
            val passage = snippets.ifBlank { text(el, "headline") }
            if (url.isBlank() && title.isBlank()) continue
            results.add(SearchResult(title, url, passage))
        }
        return results
    }

    private fun text(el: Element, tag: String): String {
        val nodes = el.getElementsByTagName(tag)
        if (nodes.length == 0) return ""
        return nodes.item(0).textContent?.trim() ?: ""
    }

    private fun passages(el: Element): String {
        val nodes = el.getElementsByTagName("passage")
        return (0 until nodes.length).joinToString(" ") { nodes.item(it).textContent?.trim() ?: "" }.trim()
    }

    fun close() {
        client.close()
    }
}
