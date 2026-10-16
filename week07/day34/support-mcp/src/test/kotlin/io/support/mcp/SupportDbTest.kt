package io.support.mcp

import io.support.mcp.db.SupportDb
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class SupportDbTest {

    private lateinit var db: SupportDb
    private lateinit var dbFile: File

    @BeforeTest
    fun setUp() {
        dbFile = File.createTempFile("support-test", ".db")
        dbFile.deleteOnExit()
        db = SupportDb(dbFile.absolutePath)
    }

    @AfterTest
    fun tearDown() {
        db.close()
        dbFile.delete()
    }

    @Test
    fun `seed creates users and one example ticket`() {
        db.seedIfEmpty()
        assertEquals(3, db.countUsers())
        assertEquals(1, db.countTickets())
        assertNotNull(db.findUserByEmail("anna@velodom.example"))
        assertNotNull(db.getTicketByNumber("T-1001"))
    }

    @Test
    fun `create ticket generates readable number and links customer`() {
        db.seedIfEmpty()
        val anna = db.findUserByEmail("anna@velodom.example")!!
        val t = db.createTicket("Не работает авторизация", "Не могу войти", "anna@velodom.example")
        assertEquals("T-1002", t.number)
        assertEquals("open", t.status)
        assertEquals(anna.id, t.customerId)

        val loaded = db.getTicketByNumber("T-1002")
        assertNotNull(loaded)
        assertEquals("Не работает авторизация", loaded.subject)
        assertEquals("Не могу войти", loaded.firstMessage)
    }

    @Test
    fun `anonymous ticket has no customer`() {
        val t = db.createTicket("Вопрос о доставке", "Сколько идёт доставка?", null)
        assertNull(t.customerId)
        assertEquals("T-1001", t.number)
    }

    @Test
    fun `get ticket by numeric id works`() {
        db.seedIfEmpty()
        val byId = db.getTicketById(1)
        assertEquals("T-1001", byId?.number)
    }

    @Test
    fun `search tickets finds by subject substring`() {
        db.seedIfEmpty()
        db.createTicket("Проблема с авторизацией", "Не могу войти", null)
        val found = db.searchTickets("авторизац")
        assertEquals(1, found.size)
        assertEquals("Проблема с авторизацией", found[0].subject)
    }

    @Test
    fun `list tickets for user filters by email`() {
        db.seedIfEmpty()
        db.createTicket("Возврат", "Хочу вернуть шлем", "anna@velodom.example")
        db.createTicket("Оплата", "Не проходит оплата", null)
        val annaTickets = db.listTicketsForUser("anna@velodom.example")
        assertEquals(2, annaTickets.size) // сид + новый
    }
}
