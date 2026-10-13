package io.currency.mcp

import io.currency.mcp.db.RateDb
import io.currency.mcp.rates.SummaryService
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class SummaryServiceTest {

    private lateinit var db: RateDb
    private lateinit var summaries: SummaryService
    private lateinit var dbFile: File

    @BeforeTest
    fun setUp() {
        dbFile = File.createTempFile("currency-test", ".db")
        dbFile.deleteOnExit()
        db = RateDb(dbFile.absolutePath)
        summaries = SummaryService(db)
    }

    @AfterTest
    fun tearDown() {
        db.close()
        dbFile.delete()
    }

    @Test
    fun `insert and read latest rate`() {
        db.insertRate("EUR", "USD", 1.13895, 1_000, "2026-09-25T08:44:11+0000")
        db.insertRate("EUR", "USD", 1.14, 2_000, "2026-09-25T08:46:11+0000")

        val latest = db.latestRate()
        assertEquals("EUR", latest?.source)
        assertEquals("USD", latest?.target)
        assertEquals(1.14, latest?.rate)
        assertEquals("2026-09-25T08:46:11+0000", latest?.time)
    }

    @Test
    fun `duplicate time is ignored`() {
        db.insertRate("EUR", "USD", 1.1, 1_000, "2026-09-25T08:44:11+0000")
        db.insertRate("EUR", "USD", 9.9, 1_000, "2026-09-25T08:44:11+0000")

        assertEquals(1, db.countRates())
        assertEquals(1.1, db.latestRate()?.rate)
    }

    @Test
    fun `summary computes count avg trend and last_n`() {
        db.insertRate("EUR", "USD", 1.10, 1_000, "2026-09-25T08:40:00+0000")
        db.insertRate("EUR", "USD", 1.20, 2_000, "2026-09-25T08:42:00+0000")
        db.insertRate("EUR", "USD", 1.30, 3_000, "2026-09-25T08:44:00+0000")

        val s = summaries.summarize("2026-09-25", lastN = 2)
        assertEquals(3, s.count)
        assertEquals(1.20, s.avg, 1e-9)
        assertEquals("flat", s.trend) // no previous day
        assertEquals(2, s.lastN.size)
        assertEquals(1.20, s.lastN[0].rate, 1e-9)
        assertEquals(1.30, s.lastN[1].rate, 1e-9)
    }

    @Test
    fun `trend compares against previous day`() {
        // day 1: avg 1.00
        db.insertRate("EUR", "USD", 1.00, 1_000, "2026-09-24T08:40:00+0000")
        db.insertRate("EUR", "USD", 1.00, 2_000, "2026-09-24T08:42:00+0000")
        summaries.ensureSummary("2026-09-24")

        // day 2: avg 1.20 -> up
        db.insertRate("EUR", "USD", 1.20, 3_000, "2026-09-25T08:40:00+0000")
        db.insertRate("EUR", "USD", 1.20, 4_000, "2026-09-25T08:42:00+0000")
        val s = summaries.ensureSummary("2026-09-25")
        assertEquals("up", s.trend)
    }

    @Test
    fun `ensureSummary caches the result`() {
        db.insertRate("EUR", "USD", 1.10, 1_000, "2026-09-25T08:40:00+0000")
        summaries.ensureSummary("2026-09-25")

        val cached = requireNotNull(db.summaryForDate("2026-09-25"))
        assertEquals(1, cached.count)
        assertEquals(1.10, cached.avg, 1e-9)
    }
}
