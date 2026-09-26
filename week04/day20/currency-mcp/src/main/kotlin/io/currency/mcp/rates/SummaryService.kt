package io.currency.mcp.rates

import io.currency.mcp.Config
import io.currency.mcp.db.RateDb
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.slf4j.LoggerFactory
import java.time.LocalDate
import java.time.ZoneOffset

/** One entry of a summary's `last_n` array. */
data class LastNItem(val rate: Double, val time: String)

/**
 * A daily summary: `{ count, avg, trend, last_n }`.
 */
data class Summary(
    val date: String,
    val count: Int,
    val avg: Double,
    val trend: String,
    val lastN: List<LastNItem>,
) {
    /** Rounded to 6 decimals to match the API's precision. */
    val avgRounded: Double get() = round6(avg)

    companion object {
        private fun round6(v: Double): Double = Math.round(v * 1_000_000) / 1_000_000.0
    }
}

/**
 * Aggregates stored rates into daily summaries (requirement: "summarization every 5 minutes").
 *
 * [summarize] computes the summary for a date; [ensureSummary] returns a cached value when
 * available, otherwise computes and persists it. Both are used by the periodic job and by the
 * `get_summary` tool.
 */
class SummaryService(private val db: RateDb) {

    private val log = LoggerFactory.getLogger(SummaryService::class.java)

    /** Summarizes [date] from raw rates (no cache). */
    fun summarize(date: String, lastN: Int = Config.lastN): Summary {
        val rates = db.ratesForDate(date)
        if (rates.isEmpty()) {
            return Summary(date, 0, 0.0, TREND_FLAT, emptyList())
        }
        val count = rates.size
        val avg = rates.map { it.rate }.average()
        val previous = db.summaryBefore(date)
        val trend = when {
            previous == null -> TREND_FLAT
            avg > previous.avg -> TREND_UP
            avg < previous.avg -> TREND_DOWN
            else -> TREND_FLAT
        }
        val last = rates.takeLast(lastN).map { LastNItem(it.rate, it.time) }
        return Summary(date, count, avg, trend, last)
    }

    /** Returns a summary for [date], computing and caching it when necessary. */
    fun ensureSummary(date: String, lastN: Int = Config.lastN): Summary {
        db.summaryForDate(date)?.let { cached ->
            // `last_n` is recomputed from the table so it always reflects fresh data.
            return Summary(date, cached.count, cached.avg, cached.trend, lastNItems(date, lastN))
        }
        val summary = summarize(date, lastN)
        if (summary.count > 0) {
            persist(summary)
        }
        return summary
    }

    /** Periodic tick: summarize "today" (UTC) and persist. */
    fun summarizeToday() {
        val today = LocalDate.now(ZoneOffset.UTC).toString()
        try {
            val summary = summarize(today)
            if (summary.count > 0) {
                persist(summary)
                log.info(
                    "summarized {}: count={}, avg={}, trend={}",
                    today, summary.count, summary.avgRounded, summary.trend,
                )
            }
        } catch (e: Exception) {
            log.warn("summarization failed for {}: {}", today, e.message)
        }
    }

    private fun persist(summary: Summary) {
        val lastNJson = buildJsonArray {
            summary.lastN.forEach { item ->
                add(buildJsonObject { put("rate", item.rate); put("time", item.time) })
            }
        }.toString()
        db.upsertSummary(summary.date, summary.count, summary.avg, summary.trend, lastNJson)
    }

    private fun lastNItems(date: String, lastN: Int): List<LastNItem> =
        db.ratesForDate(date).takeLast(lastN).map { LastNItem(it.rate, it.time) }

    companion object {
        const val TREND_UP = "up"
        const val TREND_DOWN = "down"
        const val TREND_FLAT = "flat"
    }
}
