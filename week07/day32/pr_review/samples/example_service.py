"""Пример сервиса с намеренными дефектами — демо для AI-ревью (День 32).

Это «результат PR»: файл добавляется в примерном PR (example_pr.diff) и
содержит несколько типовых проблем, которые ревью должно находить:
  - SQL-инъекция (подстановка имени в строку запроса);
  - N+1 запросов (коннект на каждую итерацию);
  - проглатывание исключений (return 0 при любой ошибке);
  - отсутствие обработки None (fetchone()[0]).
"""
import sqlite3


class UserRepository:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_user(self, username):
        # BUG: SQL-инъекция — имя пользователя подставляется в строку запроса.
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(f"SELECT * FROM users WHERE username = '{username}'")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row

    def list_orders(self, user_ids):
        # BUG: N+1 запрос — отдельный коннект и запрос на каждого пользователя.
        orders = []
        for uid in user_ids:
            conn = sqlite3.connect(self.db_path)
            orders.append(
                conn.execute("SELECT * FROM orders WHERE user_id = ?", (uid,)).fetchall()
            )
            conn.close()
        return orders

    def total_spent(self, user_id):
        # BUG: молчаливое исключение и нет проверки None.
        try:
            conn = sqlite3.connect(self.db_path)
            return conn.execute(
                "SELECT SUM(amount) FROM orders WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
        except Exception:
            return 0
