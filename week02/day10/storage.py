# storage.py
"""Персистентное хранилище истории диалогов агента (SQLite).

История хранится в файле agent_history.db рядом с сервером.
При старте агент загружает всю сохранённую историю обратно в память,
поэтому диалог можно продолжить так, как будто агент не выключался.

Помимо полной истории сообщений хранится отдельно «резюме» (summary)
сжатой части диалога, а также счётчик сжатых сообщений и накопленная
экономия токенов. Всё это восстанавливается между перезапусками агента.
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
    """Хранит историю сообщений (messages) и резюме (summary) по сессиям."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: Dict[str, List[dict]] = {}
        self._summary_cache: Dict[str, dict] = {}
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
                # Отдельная таблица для резюме сжатой части диалога.
                # summary хранится независимо от полной истории и подставляется
                # в запрос вместо старых сообщений, экономя токены.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id          TEXT PRIMARY KEY,
                        summary             TEXT NOT NULL,
                        summarized_count    INTEGER NOT NULL DEFAULT 0,
                        total_saved_tokens  INTEGER NOT NULL DEFAULT 0,
                        total_summary_tokens INTEGER NOT NULL DEFAULT 0,
                        created_at          TEXT NOT NULL,
                        updated_at          TEXT NOT NULL
                    )
                    """
                )
                # Миграция для уже существующих БД: добавляем колонку учёта
                # затрат на саммаризацию, если таблица была создана раньше.
                cols = [
                    r[1]
                    for r in conn.execute("PRAGMA table_info(session_summaries)").fetchall()
                ]
                if "total_summary_tokens" not in cols:
                    conn.execute(
                        "ALTER TABLE session_summaries "
                        "ADD COLUMN total_summary_tokens INTEGER NOT NULL DEFAULT 0"
                    )
                conn.commit()

    def load_all_history(self) -> int:
        """Загружает всю сохранённую историю и резюме из БД в память.

        Вызывается при старте агента (при перезапуске), чтобы восстановить
        контекст всех предыдущих диалогов.
        """
        with self._lock:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT session_id, messages FROM conversations"
                ).fetchall()
                summary_rows = conn.execute(
                    "SELECT session_id, summary, summarized_count, "
                    "total_saved_tokens, total_summary_tokens "
                    "FROM session_summaries"
                ).fetchall()
            self._cache = {
                row["session_id"]: json.loads(row["messages"]) for row in rows
            }
            self._summary_cache = {
                row["session_id"]: {
                    "summary": row["summary"],
                    "summarized_count": row["summarized_count"],
                    "total_saved_tokens": row["total_saved_tokens"],
                    "total_summary_tokens": row["total_summary_tokens"],
                }
                for row in summary_rows
            }
        logger.info(
            "История загружена: восстановлено сессий — %d, резюме — %d",
            len(self._cache),
            len(self._summary_cache),
        )
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

    # ------------------------------------------------------------
    # Резюме (summary) — хранится отдельно от полной истории
    # ------------------------------------------------------------
    def load_summary(self, session_id: str) -> Optional[dict]:
        """Возвращает резюме сессии:
        {"summary", "summarized_count", "total_saved_tokens", "total_summary_tokens"}
        или None.
        """
        with self._lock:
            state = self._summary_cache.get(session_id)
            return dict(state) if state is not None else None

    def save_summary(
        self,
        session_id: str,
        summary: str,
        summarized_count: int,
        total_saved_tokens: int,
        total_summary_tokens: int = 0,
    ) -> None:
        """Сохраняет (создаёт или обновляет) резюме сжатой части диалога."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._summary_cache[session_id] = {
                "summary": summary,
                "summarized_count": summarized_count,
                "total_saved_tokens": total_saved_tokens,
                "total_summary_tokens": total_summary_tokens,
            }
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO session_summaries
                        (session_id, summary, summarized_count, total_saved_tokens,
                         total_summary_tokens, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        summary              = excluded.summary,
                        summarized_count     = excluded.summarized_count,
                        total_saved_tokens   = excluded.total_saved_tokens,
                        total_summary_tokens = excluded.total_summary_tokens,
                        updated_at           = excluded.updated_at
                    """,
                    (session_id, summary, summarized_count, total_saved_tokens,
                     total_summary_tokens, now, now),
                )
                conn.commit()

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию (историю и резюме); True, если сессия существовала."""
        with self._lock:
            if session_id not in self._cache:
                return False
            del self._cache[session_id]
            self._summary_cache.pop(session_id, None)
            with closing(self._connect()) as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
                conn.execute(
                    "DELETE FROM session_summaries WHERE session_id = ?", (session_id,)
                )
                conn.commit()
            return True

    def list_sessions(self) -> List[dict]:
        """Возвращает список сохранённых сессий (для отладки/обзора)."""
        with self._lock:
            result = []
            for sid, messages in self._cache.items():
                summary = self._summary_cache.get(sid)
                result.append(
                    {
                        "session_id": sid,
                        "message_count": len(messages),
                        "summarized_count": summary["summarized_count"] if summary else 0,
                        "total_saved_tokens": summary["total_saved_tokens"] if summary else 0,
                    }
                )
            return result


# Синглтон хранилища: история загружается из БД сразу при импорте модуля,
# то есть при каждом запуске/перезапуске агента.
storage = HistoryStorage()
