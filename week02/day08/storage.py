# storage.py
"""Персистентное хранилище истории диалогов агента (SQLite).

История хранится в файле agent_history.db рядом с сервером.
При старте агент загружает всю сохранённую историю обратно в память,
поэтому диалог можно продолжить так, как будто агент не выключался.
"""
import json
import logging
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("agent.storage")

DB_PATH = os.environ.get(
    "AGENT_HISTORY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_history.db"),
)


class HistoryStorage:
    """Хранит историю сообщений (messages) по сессиям (session_id)."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: Dict[str, List[dict]] = {}
        self._init_db()
        self.load_all_history()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id TEXT PRIMARY KEY,
                        messages    TEXT NOT NULL,
                        created_at  TEXT NOT NULL,
                        updated_at  TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def load_all_history(self) -> int:
        """Загружает всю сохранённую историю из БД в память.

        Вызывается при старте агента (при перезапуске), чтобы восстановить
        контекст всех предыдущих диалогов.
        """
        with self._lock:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT session_id, messages FROM conversations"
                ).fetchall()
            self._cache = {
                row["session_id"]: json.loads(row["messages"]) for row in rows
            }
        logger.info("История загружена: восстановлено сессий — %d", len(self._cache))
        return len(self._cache)

    def load(self, session_id: str) -> Optional[List[dict]]:
        """Возвращает историю сообщений сессии или None, если сессия новая."""
        with self._lock:
            messages = self._cache.get(session_id)
            # Копия, чтобы вызывающий код не менял кэш напрямую.
            return list(messages) if messages is not None else None

    def save(self, session_id: str, messages: List[dict]) -> None:
        """Сохраняет (создаёт или обновляет) историю сообщений сессии."""
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(messages, ensure_ascii=False)
        with self._lock:
            self._cache[session_id] = list(messages)
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (session_id, messages, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        messages   = excluded.messages,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, payload, now, now),
                )
                conn.commit()

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию; возвращает True, если сессия существовала."""
        with self._lock:
            if session_id not in self._cache:
                return False
            del self._cache[session_id]
            with closing(self._connect()) as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
                conn.commit()
            return True

    def list_sessions(self) -> List[dict]:
        """Возвращает список сохранённых сессий (для отладки/обзора)."""
        with self._lock:
            return [
                {"session_id": sid, "message_count": len(messages)}
                for sid, messages in self._cache.items()
            ]


# Синглтон хранилища: история загружается из БД сразу при импорте модуля,
# то есть при каждом запуске/перезапуске агента.
storage = HistoryStorage()
