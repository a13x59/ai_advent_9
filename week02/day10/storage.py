# storage.py
"""Персистентное хранилище истории диалогов агента (SQLite).

История хранится в файле agent_history.db рядом с сервером.
При старте агент загружает всю сохранённую историю обратно в память,
поэтому диалог можно продолжить так, как будто агент не выключался.

Полная история ВСЕГДА хранится в БД и восстанавливается между перезапусками.
Поверх полной истории работают 3 стратегии управления контекстом, которые
выбираются параметром запроса:

  • sliding_window — в запрос уходит только окно последних N сообщений;
  • sticky_facts    — отдельный блок фактов (ключ-значение) + окно последних N;
  • branching       — диалог ветвится от контрольной точки (checkpoint): ветки
                      хранятся и продолжаются независимо, между ними можно
                      переключаться.

Для каждой сессии хранится:
  • полная история сообщений по каждой ветке (таблица branches);
  • указатель на активную ветку и выбранная стратегия (таблица conversations);
  • блок фактов ключ-значение для стратегии sticky_facts (таблица session_facts).
"""
import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("agent.storage")

DB_PATH = os.environ.get(
    "AGENT_HISTORY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_history.db"),
)


class HistoryStorage:
    """Хранит сессии, ветки диалога и факты (ключ-значение)."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        # session_id -> {
        #     "current_branch": str,
        #     "strategy": str,
        #     "window_size": int,
        #     "branches": {branch_id -> {"name": str, "messages": list, "checkpoint": int}},
        # }
        self._sessions: Dict[str, dict] = {}
        self._branch_index: Dict[str, str] = {}  # branch_id -> session_id
        self._facts: Dict[str, dict] = {}        # session_id -> facts dict
        self._init_db()
        self.load_all_history()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with closing(self._connect()) as conn:
                # Метаданные сессии: активная ветка, выбранная стратегия, окно.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id     TEXT PRIMARY KEY,
                        current_branch TEXT NOT NULL,
                        strategy       TEXT NOT NULL DEFAULT 'sliding_window',
                        window_size    INTEGER NOT NULL DEFAULT 10,
                        created_at     TEXT NOT NULL,
                        updated_at     TEXT NOT NULL
                    )
                    """
                )
                # Ветки диалога. Каждая ветка хранит СВОЮ полную историю,
                # поэтому даже после ветвления ничего не теряется.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS branches (
                        branch_id  TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        name       TEXT NOT NULL,
                        messages   TEXT NOT NULL,
                        checkpoint INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Блок фактов (ключ-значение) для стратегии sticky_facts.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_facts (
                        session_id TEXT PRIMARY KEY,
                        facts      TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Миграция: старый механизм саммаризации больше не используется,
                # его таблицу можно безопасно удалить.
                conn.execute("DROP TABLE IF EXISTS session_summaries")
                # Совместимость со старыми БД, где не было новых колонок.
                cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
                if "strategy" not in cols:
                    conn.execute(
                        "ALTER TABLE conversations ADD COLUMN strategy TEXT NOT NULL DEFAULT 'sliding_window'"
                    )
                if "window_size" not in cols:
                    conn.execute(
                        "ALTER TABLE conversations ADD COLUMN window_size INTEGER NOT NULL DEFAULT 10"
                    )
                conn.commit()

    def load_all_history(self) -> int:
        """Загружает все сессии, ветки и факты из БД в память при старте."""
        with self._lock:
            with closing(self._connect()) as conn:
                session_rows = conn.execute(
                    "SELECT session_id, current_branch, strategy, window_size FROM conversations"
                ).fetchall()
                branch_rows = conn.execute(
                    "SELECT branch_id, session_id, name, messages, checkpoint FROM branches"
                ).fetchall()
                facts_rows = conn.execute(
                    "SELECT session_id, facts FROM session_facts"
                ).fetchall()

            self._sessions = {}
            self._branch_index = {}
            for row in session_rows:
                self._sessions[row["session_id"]] = {
                    "current_branch": row["current_branch"],
                    "strategy": row["strategy"],
                    "window_size": row["window_size"],
                    "branches": {},
                }
            for row in branch_rows:
                session = self._sessions.get(row["session_id"])
                if session is None:
                    continue
                session["branches"][row["branch_id"]] = {
                    "name": row["name"],
                    "messages": json.loads(row["messages"]),
                    "checkpoint": row["checkpoint"],
                }
                self._branch_index[row["branch_id"]] = row["session_id"]
            self._facts = {
                row["session_id"]: json.loads(row["facts"]) for row in facts_rows
            }

        logger.info(
            "История загружена: сессий — %d, веток — %d, блоков фактов — %d",
            len(self._sessions),
            len(self._branch_index),
            len(self._facts),
        )
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Сессии и метаданные
    # ------------------------------------------------------------------
    def get_session_meta(self, session_id: str) -> Optional[dict]:
        """Возвращает {"current_branch", "strategy", "window_size"} или None."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "current_branch": session["current_branch"],
                "strategy": session["strategy"],
                "window_size": session["window_size"],
            }

    def create_session(
        self,
        session_id: str,
        strategy: str = "sliding_window",
        window_size: int = 10,
        messages: Optional[List[dict]] = None,
        branch_name: str = "main",
    ) -> None:
        """Создаёт новую сессию с одной стартовой веткой."""
        now = datetime.now(timezone.utc).isoformat()
        branch_id = uuid.uuid4().hex
        messages = list(messages or [])
        with self._lock:
            self._sessions[session_id] = {
                "current_branch": branch_id,
                "strategy": strategy,
                "window_size": window_size,
                "branches": {
                    branch_id: {
                        "name": branch_name,
                        "messages": messages,
                        "checkpoint": 0,
                    }
                },
            }
            self._branch_index[branch_id] = session_id
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                        (session_id, current_branch, strategy, window_size, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, branch_id, strategy, window_size, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO branches
                        (branch_id, session_id, name, messages, checkpoint, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        branch_id, session_id, branch_name,
                        json.dumps(messages, ensure_ascii=False), 0, now, now,
                    ),
                )
                conn.commit()

    def set_session_meta(self, session_id: str, strategy: str, window_size: int) -> bool:
        """Обновляет стратегию и размер окна сессии."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session["strategy"] = strategy
            session["window_size"] = window_size
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    UPDATE conversations
                    SET strategy = ?, window_size = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (strategy, window_size, now, session_id),
                )
                conn.commit()
            return True

    def get_current_branch_id(self, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
            return session["current_branch"] if session else None

    # ------------------------------------------------------------------
    # Сообщения (работа с активной веткой)
    # ------------------------------------------------------------------
    def load(self, session_id: str) -> Optional[List[dict]]:
        """Возвращает историю активной ветки сессии или None."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            branch = session["branches"].get(session["current_branch"])
            return list(branch["messages"]) if branch else None

    def load_branch(self, branch_id: str) -> Optional[List[dict]]:
        """Возвращает историю конкретной ветки по её id или None."""
        with self._lock:
            session_id = self._branch_index.get(branch_id)
            if not session_id:
                return None
            branch = self._sessions[session_id]["branches"].get(branch_id)
            return list(branch["messages"]) if branch else None

    def save(self, session_id: str, messages: List[dict]) -> None:
        """Сохраняет историю сообщений в активную ветку сессии."""
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(messages, ensure_ascii=False)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            branch_id = session["current_branch"]
            session["branches"][branch_id]["messages"] = list(messages)
            with closing(self._connect()) as conn:
                conn.execute(
                    "UPDATE branches SET messages = ?, updated_at = ? WHERE branch_id = ?",
                    (payload, now, branch_id),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Ветки (стратегия branching)
    # ------------------------------------------------------------------
    def create_branch(
        self,
        session_id: str,
        name: str,
        messages: List[dict],
        checkpoint: int,
    ) -> Optional[str]:
        """Создаёт новую ветку и возвращает её id (или None, если сессии нет)."""
        now = datetime.now(timezone.utc).isoformat()
        branch_id = uuid.uuid4().hex
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session["branches"][branch_id] = {
                "name": name,
                "messages": list(messages),
                "checkpoint": checkpoint,
            }
            self._branch_index[branch_id] = session_id
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO branches
                        (branch_id, session_id, name, messages, checkpoint, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        branch_id, session_id, name,
                        json.dumps(messages, ensure_ascii=False), checkpoint, now, now,
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()
            return branch_id

    def switch_branch(self, session_id: str, branch_id: str) -> bool:
        """Переключает активную ветку сессии; False, если ветки нет."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or branch_id not in session["branches"]:
                return False
            session["current_branch"] = branch_id
            with closing(self._connect()) as conn:
                conn.execute(
                    "UPDATE conversations SET current_branch = ?, updated_at = ? WHERE session_id = ?",
                    (branch_id, now, session_id),
                )
                conn.commit()
            return True

    def list_branches(self, session_id: str) -> List[dict]:
        """Список веток сессии: [{branch_id, name, checkpoint, message_count}]."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            result = []
            for bid, branch in session["branches"].items():
                result.append(
                    {
                        "branch_id": bid,
                        "name": branch["name"],
                        "checkpoint": branch["checkpoint"],
                        "message_count": len(branch["messages"]),
                    }
                )
            return result

    # ------------------------------------------------------------------
    # Факты (стратегия sticky_facts)
    # ------------------------------------------------------------------
    def load_facts(self, session_id: str) -> Optional[dict]:
        with self._lock:
            facts = self._facts.get(session_id)
            return dict(facts) if facts is not None else None

    def save_facts(self, session_id: str, facts: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._facts[session_id] = dict(facts)
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO session_facts (session_id, facts, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        facts      = excluded.facts,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, json.dumps(facts, ensure_ascii=False), now, now),
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Управление сессиями
    # ------------------------------------------------------------------
    def delete(self, session_id: str) -> bool:
        """Удаляет сессию целиком: ветки, факты и метаданные."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            for branch_id in session["branches"]:
                self._branch_index.pop(branch_id, None)
            self._facts.pop(session_id, None)
            with closing(self._connect()) as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )
                conn.execute(
                    "DELETE FROM branches WHERE session_id = ?", (session_id,)
                )
                conn.execute(
                    "DELETE FROM session_facts WHERE session_id = ?", (session_id,)
                )
                conn.commit()
            return True

    def list_sessions(self) -> List[dict]:
        """Список сохранённых сессий (для обзора/отладки)."""
        with self._lock:
            result = []
            for sid, session in self._sessions.items():
                branch = session["branches"].get(session["current_branch"])
                result.append(
                    {
                        "session_id": sid,
                        "strategy": session["strategy"],
                        "message_count": len(branch["messages"]) if branch else 0,
                        "branch_count": len(session["branches"]),
                        "facts_count": len(self._facts.get(sid, {})),
                    }
                )
            return result


# Синглтон хранилища: история загружается из БД сразу при импорте модуля,
# то есть при каждом запуске/перезапуске агента.
storage = HistoryStorage()
