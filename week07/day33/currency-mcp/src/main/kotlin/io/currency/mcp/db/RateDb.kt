package io.currency.mcp.db

import java.sql.Connection
import java.sql.DriverManager

/**
 * A single rate sample, as returned by the rates API and stored in `rates`.
 */
data class RateRow(
    val source: String,
    val target: String,
    val rate: Double,
    val ts: Long,
    val time: String,
)

/**
 * A pre-aggregated daily summary, stored in `summaries`.
 */
data class SummaryRow(
    val date: String,
    val count: Int,
    val avg: Double,
    val trend: String,
    val lastN: String,
    val updatedAt: Long,
)

/**
 * SQLite storage for the currency MCP service (requirement 2: `.db` files).
 *
 * Tables:
 *  - rates     — every fetched rate sample (EUR/USD);
 *  - summaries — daily aggregated summaries (count, avg, trend, last_n).
 *
 * A single JDBC connection is shared, so all access is synchronized.
 */
class RateDb(private val path: String) {

    private val conn: Connection = DriverManager.getConnection("jdbc:sqlite:$path")

    init {
        synchronized(conn) {
            conn.createStatement().use { st ->
                st.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rates (
                        id     INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        target TEXT NOT NULL,
                        rate   REAL NOT NULL,
                        ts     INTEGER NOT NULL,
                        time   TEXT NOT NULL UNIQUE
                    )
                    """.trimIndent(),
                )
                st.execute(
                    """
                    CREATE TABLE IF NOT EXISTS summaries (
                        date       TEXT PRIMARY KEY,
                        count      INTEGER NOT NULL,
                        avg        REAL NOT NULL,
                        trend      TEXT NOT NULL,
                        last_n     TEXT NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                    """.trimIndent(),
                )
            }
        }
    }

    /** Inserts a rate sample; ignores duplicates keyed by the API `time` string. */
    fun insertRate(source: String, target: String, rate: Double, ts: Long, time: String) {
        synchronized(conn) {
            conn.prepareStatement(
                "INSERT OR IGNORE INTO rates (source, target, rate, ts, time) VALUES (?, ?, ?, ?, ?)",
            ).use { ps ->
                ps.setString(1, source)
                ps.setString(2, target)
                ps.setDouble(3, rate)
                ps.setLong(4, ts)
                ps.setString(5, time)
                ps.executeUpdate()
            }
        }
    }

    /** Latest stored rate, or null if the table is empty. */
    fun latestRate(): RateRow? = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery("SELECT source, target, rate, ts, time FROM rates ORDER BY id DESC LIMIT 1").use { rs ->
                if (rs.next()) {
                    RateRow(rs.getString(1), rs.getString(2), rs.getDouble(3), rs.getLong(4), rs.getString(5))
                } else {
                    null
                }
            }
        }
    }

    /** All rates for the given UTC date (`YYYY-MM-DD`), ordered oldest-first. */
    fun ratesForDate(date: String): List<RateRow> = synchronized(conn) {
        conn.prepareStatement(
            "SELECT source, target, rate, ts, time FROM rates WHERE time LIKE ? ORDER BY id ASC",
        ).use { ps ->
            ps.setString(1, "$date%")
            ps.executeQuery().use { rs ->
                buildList {
                    while (rs.next()) {
                        add(RateRow(rs.getString(1), rs.getString(2), rs.getDouble(3), rs.getLong(4), rs.getString(5)))
                    }
                }
            }
        }
    }

    /** Most recent summary strictly before [date] (used to compute `trend`). */
    fun summaryBefore(date: String): SummaryRow? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT date, count, avg, trend, last_n, updated_at FROM summaries WHERE date < ? ORDER BY date DESC LIMIT 1",
        ).use { ps ->
            ps.setString(1, date)
            ps.executeQuery().use { rs ->
                if (rs.next()) {
                    SummaryRow(rs.getString(1), rs.getInt(2), rs.getDouble(3), rs.getString(4), rs.getString(5), rs.getLong(6))
                } else {
                    null
                }
            }
        }
    }

    /** Cached summary for [date], or null. */
    fun summaryForDate(date: String): SummaryRow? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT date, count, avg, trend, last_n, updated_at FROM summaries WHERE date = ?",
        ).use { ps ->
            ps.setString(1, date)
            ps.executeQuery().use { rs ->
                if (rs.next()) {
                    SummaryRow(rs.getString(1), rs.getInt(2), rs.getDouble(3), rs.getString(4), rs.getString(5), rs.getLong(6))
                } else {
                    null
                }
            }
        }
    }

    /** Upserts a summary for [date]. */
    fun upsertSummary(date: String, count: Int, avg: Double, trend: String, lastNJson: String) {
        synchronized(conn) {
            conn.prepareStatement(
                """
                INSERT INTO summaries (date, count, avg, trend, last_n, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    count = excluded.count,
                    avg = excluded.avg,
                    trend = excluded.trend,
                    last_n = excluded.last_n,
                    updated_at = excluded.updated_at
                """.trimIndent(),
            ).use { ps ->
                ps.setString(1, date)
                ps.setInt(2, count)
                ps.setDouble(3, avg)
                ps.setString(4, trend)
                ps.setString(5, lastNJson)
                ps.setLong(6, System.currentTimeMillis())
                ps.executeUpdate()
            }
        }
    }

    /** Total number of stored rate samples (used by debug endpoints). */
    fun countRates(): Int = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery("SELECT COUNT(*) FROM rates").use { rs ->
                if (rs.next()) rs.getInt(1) else 0
            }
        }
    }

    fun close() {
        synchronized(conn) {
            conn.close()
        }
    }
}
