package io.currency.mcp.rates

import io.currency.mcp.Config
import io.currency.mcp.db.RateDb
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.parameter
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.slf4j.LoggerFactory
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter

/**
 * A single element of the rates API response, e.g.
 * `{"rate":1.13895,"source":"EUR","target":"USD","time":"2026-09-25T08:44:11+0000"}`.
 */
@Serializable
data class ApiRate(
    val rate: Double,
    val source: String,
    val target: String,
    val time: String,
)

/**
 * Fetches EUR/USD rates over HTTP and stores them in SQLite.
 */
class RateService(private val db: RateDb) {

    private val log = LoggerFactory.getLogger(RateService::class.java)
    private val json = Json { ignoreUnknownKeys = true }

    private val client = HttpClient(CIO) {
        expectSuccess = false
    }

    private val timeFormat: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssZ")

    /** Fetches the current rates from the API. */
    suspend fun fetch(): List<ApiRate> {
        val response = client.get(Config.apiBaseUrl) {
            parameter("source", Config.source)
            parameter("target", Config.target)
            if (Config.apiKey.isNotBlank()) {
                header(HttpHeaders.Authorization, "Bearer ${Config.apiKey}")
            }
        }
        val body = response.bodyAsText()
        if (response.status != HttpStatusCode.OK) {
            throw IllegalStateException("Rates API returned ${response.status.value}: $body")
        }
        return json.decodeFromString<List<ApiRate>>(body)
    }

    /** One polling tick: fetch and persist. Never throws. */
    suspend fun pollOnce() {
        try {
            val rates = fetch()
            for (r in rates) {
                db.insertRate(r.source, r.target, r.rate, parseEpoch(r.time), r.time)
            }
            log.info("polled {} rate sample(s)", rates.size)
        } catch (e: Exception) {
            log.warn("rate poll failed: {}", e.message)
        }
    }

    private fun parseEpoch(time: String): Long = try {
        OffsetDateTime.parse(time, timeFormat).toInstant().toEpochMilli()
    } catch (e: Exception) {
        System.currentTimeMillis()
    }

    fun close() {
        client.close()
    }
}
