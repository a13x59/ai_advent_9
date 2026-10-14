"""Legacy-сервис платежей (демо: содержит намеренные дефекты)."""
import sqlite3


class PaymentRepository:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_payment(self, payment_id):
        conn = sqlite3.connect(self.db_path)
        return conn.execute(
            f"SELECT * FROM payments WHERE id = '{payment_id}'"
        ).fetchone()

    def list_by_user(self, user_ids):
        out = []
        for uid in user_ids:
            conn = sqlite3.connect(self.db_path)
            out.append(conn.execute(
                "SELECT * FROM payments WHERE user_id = ?", (uid,)
            ).fetchall())
            conn.close()
        return out

    def total(self, user_id):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn.execute(
                "SELECT SUM(amount) FROM payments WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
        except Exception:
            return 0
