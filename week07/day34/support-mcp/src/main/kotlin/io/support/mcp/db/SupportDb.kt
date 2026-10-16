package io.support.mcp.db

import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant

/** Покупатель интернет-магазина (мини-CRM). */
data class User(
    val id: Long,
    val email: String,
    val name: String,
    val phone: String,
    val plan: String,
)

/** Тикет поддержки. */
data class Ticket(
    val id: Long,
    val number: String,
    val subject: String,
    val status: String,
    val priority: String,
    val customerId: Long?,
    val firstMessage: String,
    val createdAt: String,
)

/**
 * SQLite-хранилище сервиса поддержки.
 *
 * Таблицы:
 *  - users   — покупатели (email, имя, телефон, тариф);
 *  - tickets — обращения (читаемый номер, тема, статус, приоритет, покупатель,
 *              первое сообщение).
 *
 * Одно JDBC-соединение, весь доступ синхронизирован (как в currency-mcp).
 */
class SupportDb(private val path: String) {

    private val conn: Connection = DriverManager.getConnection("jdbc:sqlite:$path")

    init {
        synchronized(conn) {
            conn.createStatement().use { st ->
                st.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        email      TEXT NOT NULL UNIQUE,
                        name       TEXT NOT NULL DEFAULT '',
                        phone      TEXT NOT NULL DEFAULT '',
                        plan       TEXT NOT NULL DEFAULT 'standard',
                        created_at TEXT NOT NULL
                    )
                    """.trimIndent(),
                )
                st.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tickets (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        number        TEXT NOT NULL UNIQUE,
                        subject       TEXT NOT NULL DEFAULT '',
                        status        TEXT NOT NULL DEFAULT 'open',
                        priority      TEXT NOT NULL DEFAULT 'normal',
                        customer_id   INTEGER,
                        first_message TEXT NOT NULL DEFAULT '',
                        created_at    TEXT NOT NULL,
                        FOREIGN KEY (customer_id) REFERENCES users(id)
                    )
                    """.trimIndent(),
                )
            }
        }
    }

    // ------------------------------------------------------------------ //
    // Пользователи                                                        //
    // ------------------------------------------------------------------ //

    fun findUserByEmail(email: String): User? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT id, email, name, phone, plan FROM users WHERE email = ?",
        ).use { ps ->
            ps.setString(1, email)
            ps.executeQuery().use { rs -> if (rs.next()) userFrom(rs) else null }
        }
    }

    fun findUserById(id: Long): User? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT id, email, name, phone, plan FROM users WHERE id = ?",
        ).use { ps ->
            ps.setLong(1, id)
            ps.executeQuery().use { rs -> if (rs.next()) userFrom(rs) else null }
        }
    }

    fun listUsers(): List<User> = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery("SELECT id, email, name, phone, plan FROM users ORDER BY id").use { rs ->
                buildList { while (rs.next()) add(userFrom(rs)) }
            }
        }
    }

    private fun insertUser(email: String, name: String, phone: String, plan: String) {
        conn.prepareStatement(
            "INSERT INTO users (email, name, phone, plan, created_at) VALUES (?, ?, ?, ?, ?)",
        ).use { ps ->
            ps.setString(1, email)
            ps.setString(2, name)
            ps.setString(3, phone)
            ps.setString(4, plan)
            ps.setString(5, nowIso())
            ps.executeUpdate()
        }
    }

    // ------------------------------------------------------------------ //
    // Тикеты                                                              //
    // ------------------------------------------------------------------ //

    /** Создаёт тикет с читаемым номером вида T-1001 (1000 + id). */
    fun createTicket(subject: String, firstMessage: String, customerEmail: String?): Ticket =
        synchronized(conn) {
            val now = nowIso()
            val customer = customerEmail?.takeIf { it.isNotBlank() }?.let { findUserByEmail(it) }
            conn.prepareStatement(
                "INSERT INTO tickets (number, subject, status, priority, customer_id, first_message, created_at) " +
                    "VALUES (?, ?, 'open', 'normal', ?, ?, ?)",
            ).use { ps ->
                ps.setString(1, "") // временный номер — обновим после получения id
                ps.setString(2, subject)
                if (customer == null) ps.setNull(3, java.sql.Types.INTEGER) else ps.setLong(3, customer.id)
                ps.setString(4, firstMessage)
                ps.setString(5, now)
                ps.executeUpdate()
            }
            val id = conn.createStatement().use { st ->
                st.executeQuery("SELECT last_insert_rowid()").use { rs ->
                    rs.next()
                    rs.getLong(1)
                }
            }
            val number = "T-" + (1000 + id)
            conn.prepareStatement("UPDATE tickets SET number = ? WHERE id = ?").use { u ->
                u.setString(1, number)
                u.setLong(2, id)
                u.executeUpdate()
            }
            Ticket(id, number, subject, "open", "normal", customer?.id, firstMessage, now)
        }

    fun getTicketByNumber(number: String): Ticket? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT id, number, subject, status, priority, customer_id, first_message, created_at " +
                "FROM tickets WHERE number = ?",
        ).use { ps ->
            ps.setString(1, number)
            ps.executeQuery().use { rs -> if (rs.next()) ticketFrom(rs) else null }
        }
    }

    fun getTicketById(id: Long): Ticket? = synchronized(conn) {
        conn.prepareStatement(
            "SELECT id, number, subject, status, priority, customer_id, first_message, created_at " +
                "FROM tickets WHERE id = ?",
        ).use { ps ->
            ps.setLong(1, id)
            ps.executeQuery().use { rs -> if (rs.next()) ticketFrom(rs) else null }
        }
    }

    fun listTickets(): List<Ticket> = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery(
                "SELECT id, number, subject, status, priority, customer_id, first_message, created_at " +
                    "FROM tickets ORDER BY id DESC",
            ).use { rs -> buildList { while (rs.next()) add(ticketFrom(rs)) } }
        }
    }

    fun listTicketsForUser(email: String): List<Ticket> = synchronized(conn) {
        conn.prepareStatement(
            "SELECT t.id, t.number, t.subject, t.status, t.priority, t.customer_id, t.first_message, t.created_at " +
                "FROM tickets t JOIN users u ON t.customer_id = u.id WHERE u.email = ? ORDER BY t.id DESC",
        ).use { ps ->
            ps.setString(1, email)
            ps.executeQuery().use { rs -> buildList { while (rs.next()) add(ticketFrom(rs)) } }
        }
    }

    fun searchTickets(query: String): List<Ticket> = synchronized(conn) {
        val like = "%${query.trim()}%"
        conn.prepareStatement(
            "SELECT id, number, subject, status, priority, customer_id, first_message, created_at " +
                "FROM tickets WHERE number LIKE ? OR subject LIKE ? OR first_message LIKE ? ORDER BY id DESC",
        ).use { ps ->
            ps.setString(1, like)
            ps.setString(2, like)
            ps.setString(3, like)
            ps.executeQuery().use { rs -> buildList { while (rs.next()) add(ticketFrom(rs)) } }
        }
    }

    // ------------------------------------------------------------------ //
    // Статистика / сиды                                                   //
    // ------------------------------------------------------------------ //

    fun countUsers(): Int = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery("SELECT COUNT(*) FROM users").use { rs -> if (rs.next()) rs.getInt(1) else 0 }
        }
    }

    fun countTickets(): Int = synchronized(conn) {
        conn.createStatement().use { st ->
            st.executeQuery("SELECT COUNT(*) FROM tickets").use { rs -> if (rs.next()) rs.getInt(1) else 0 }
        }
    }

    /** Заполняет БД демо-данными, если таблица пользователей пуста. */
    fun seedIfEmpty() {
        synchronized(conn) {
            if (countUsers() > 0) return
            insertUser("anna@velodom.example", "Анна Смирнова", "+7 900 111-22-33", "premium")
            insertUser("ivan@velodom.example", "Иван Петров", "+7 900 222-33-44", "standard")
            insertUser("maria@velodom.example", "Мария Кузнецова", "+7 900 333-44-55", "standard")
            createTicket(
                "Не приходит письмо с подтверждением",
                "Зарегистрировалась, но письмо для подтверждения аккаунта так и не пришло.",
                "anna@velodom.example",
            )
        }
    }

    fun close() {
        synchronized(conn) {
            conn.close()
        }
    }

    // ------------------------------------------------------------------ //
    // Helpers                                                             //
    // ------------------------------------------------------------------ //

    private fun userFrom(rs: java.sql.ResultSet): User = User(
        id = rs.getLong("id"),
        email = rs.getString("email"),
        name = rs.getString("name"),
        phone = rs.getString("phone"),
        plan = rs.getString("plan"),
    )

    private fun ticketFrom(rs: java.sql.ResultSet): Ticket = Ticket(
        id = rs.getLong("id"),
        number = rs.getString("number"),
        subject = rs.getString("subject"),
        status = rs.getString("status"),
        priority = rs.getString("priority"),
        customerId = rs.getLong("customer_id").takeIf { !rs.wasNull() },
        firstMessage = rs.getString("first_message"),
        createdAt = rs.getString("created_at"),
    )

    private fun nowIso(): String = Instant.now().toString()
}
