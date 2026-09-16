# storage.py
"""Персистентное хранилище агента (SQLite) с явной моделью памяти.

Агент хранит информацию в трёх независимых слоях памяти:

  • краткосрочная (short_term) — текущий диалог, таблица branches;
  • рабочая (working)          — данные текущей задачи, таблица working_memory;
  • долговременная (long_term) — профиль, решения, знания, таблица long_term_memory.

Каждый слой хранится ОТДЕЛЬНО (своя таблица), а явные действия пользователя
(что и куда сохранить / переместить / удалить) логируются в таблицу memory_log.

В отличие от прежней версии факты больше не хранятся в session_facts — факты
стали долговременной памятью с kind='fact' (long_term_memory).
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

# Допустимые значения kind/state (для валидации и подсказок в UI).
WORKING_KINDS = {"goal", "constraint", "todo", "result", "context", "note"}
WORKING_STATES = {"pending", "in_progress", "done", "blocked"}
LONG_TERM_KINDS = {"profile", "decision", "knowledge", "preference", "agreement", "fact"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_kind(kind, allowed, default):
    kind = (kind or "").strip().lower()
    return kind if kind in allowed else default


def _normalize_state(state):
    state = (state or "").strip().lower()
    return state if state in WORKING_STATES else None


class HistoryStorage:
    """Хранит сессии, ветки диалога и два слоя памяти (working / long_term)."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        # session_id -> {
        #     "current_branch": str,
        #     "strategy": str,
        #     "window_size": int,
        #     "branches": {branch_id -> {"name", "messages", "checkpoint"}},
        # }
        self._sessions: Dict[str, dict] = {}
        self._branch_index: Dict[str, str] = {}  # branch_id -> session_id
        # Рабочая память: session_id -> {key -> entry}
        self._working: Dict[str, Dict[str, dict]] = {}
        # Долговременная память (глобальная): key -> entry
        self._long_term: Dict[str, dict] = {}
        # Профили пользователя (персонализация): profile_id -> entry
        self._profiles: Dict[str, dict] = {}
        self._active_profile_id: Optional[str] = None
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
                        session_id     TEXT PRIMARY KEY,
                        current_branch TEXT NOT NULL,
                        strategy       TEXT NOT NULL DEFAULT 'sliding_window',
                        window_size    INTEGER NOT NULL DEFAULT 10,
                        profile_id     TEXT,
                        created_at     TEXT NOT NULL,
                        updated_at     TEXT NOT NULL
                    )
                    """
                )
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
                # Рабочая память — данные текущей задачи. Привязана к сессии.
                # state — состояние выполнения задачи (pending/in_progress/done/blocked).
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS working_memory (
                        entry_id   TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        key        TEXT NOT NULL,
                        value      TEXT NOT NULL,
                        kind       TEXT NOT NULL DEFAULT 'note',
                        state      TEXT,
                        importance INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(session_id, key)
                    )
                    """
                )
                # Долговременная память — профиль, решения, знания. Глобальная
                # (общая для всех сессий), ключи уникальны.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memory (
                        entry_id   TEXT PRIMARY KEY,
                        key        TEXT NOT NULL UNIQUE,
                        profile_id TEXT,
                        value      TEXT NOT NULL,
                        kind       TEXT NOT NULL DEFAULT 'fact',
                        tags       TEXT NOT NULL DEFAULT '[]',
                        importance INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Миграция для уже существующих БД (добавляем столбец profile_id,
                # если таблицы были созданы прежней версией схемы).
                if "profile_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(conversations)")}:
                    conn.execute("ALTER TABLE conversations ADD COLUMN profile_id TEXT")
                if "profile_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(long_term_memory)")}:
                    conn.execute("ALTER TABLE long_term_memory ADD COLUMN profile_id TEXT")
                # Профиль пользователя — персонализация поверх памяти:
                # явное описание предпочтений (стиль, формат, ограничения),
                # которое подключается к каждому запросу к модели.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        profile_id   TEXT PRIMARY KEY,
                        name         TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL DEFAULT '',
                        style        TEXT NOT NULL DEFAULT '',
                        format       TEXT NOT NULL DEFAULT '',
                        constraints  TEXT NOT NULL DEFAULT '',
                        notes        TEXT NOT NULL DEFAULT '',
                        is_active    INTEGER NOT NULL DEFAULT 0,
                        created_at   TEXT NOT NULL,
                        updated_at   TEXT NOT NULL
                    )
                    """
                )
                # Журнал явных действий над памятью.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_log (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        layer      TEXT NOT NULL,
                        action     TEXT NOT NULL,
                        key        TEXT,
                        source     TEXT NOT NULL DEFAULT 'manual',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                # Прежние механизмы больше не нужны.
                conn.execute("DROP TABLE IF EXISTS session_facts")
                conn.execute("DROP TABLE IF EXISTS session_summaries")
                conn.commit()

    def load_all_history(self) -> int:
        """Загружает все сессии, ветки и оба слоя памяти из БД в память."""
        with self._lock:
            with closing(self._connect()) as conn:
                session_rows = conn.execute(
                    "SELECT session_id, current_branch, strategy, window_size, profile_id FROM conversations"
                ).fetchall()
                branch_rows = conn.execute(
                    "SELECT branch_id, session_id, name, messages, checkpoint FROM branches"
                ).fetchall()
                working_rows = conn.execute(
                    "SELECT entry_id, session_id, key, value, kind, state, importance, "
                    "created_at, updated_at FROM working_memory"
                ).fetchall()
                long_term_rows = conn.execute(
                    "SELECT entry_id, key, profile_id, value, kind, tags, importance, "
                    "created_at, updated_at FROM long_term_memory"
                ).fetchall()
                profile_rows = conn.execute(
                    "SELECT profile_id, name, display_name, style, format, constraints, notes, "
                    "is_active, created_at, updated_at FROM user_profiles"
                ).fetchall()

            self._sessions = {}
            self._branch_index = {}
            for row in session_rows:
                self._sessions[row["session_id"]] = {
                    "current_branch": row["current_branch"],
                    "strategy": row["strategy"],
                    "window_size": row["window_size"],
                    "profile_id": row["profile_id"],
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

            self._working = {}
            for row in working_rows:
                entry = {
                    "entry_id": row["entry_id"],
                    "key": row["key"],
                    "value": row["value"],
                    "kind": row["kind"],
                    "state": row["state"],
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                self._working.setdefault(row["session_id"], {})[row["key"]] = entry

            self._long_term = {}
            for row in long_term_rows:
                self._long_term[row["key"]] = {
                    "entry_id": row["entry_id"],
                    "key": row["key"],
                    "profile_id": row["profile_id"],
                    "value": row["value"],
                    "kind": row["kind"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

            self._profiles = {}
            self._active_profile_id = None
            for row in profile_rows:
                profile = {
                    "profile_id": row["profile_id"],
                    "name": row["name"],
                    "display_name": row["display_name"],
                    "style": row["style"],
                    "format": row["format"],
                    "constraints": row["constraints"],
                    "notes": row["notes"],
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                self._profiles[row["profile_id"]] = profile
                if profile["is_active"]:
                    self._active_profile_id = row["profile_id"]

        logger.info(
            "Память загружена: сессий — %d, веток — %d, рабочая — %d записей, "
            "долговременная — %d записей, профилей — %d",
            len(self._sessions),
            len(self._branch_index),
            sum(len(v) for v in self._working.values()),
            len(self._long_term),
            len(self._profiles),
        )
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Сессии и метаданные
    # ------------------------------------------------------------------
    def get_session_meta(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "current_branch": session["current_branch"],
                "strategy": session["strategy"],
                "window_size": session["window_size"],
                "profile_id": session.get("profile_id"),
            }

    def create_session(
        self,
        session_id: str,
        strategy: str = "sliding_window",
        window_size: int = 10,
        messages: Optional[List[dict]] = None,
        branch_name: str = "main",
        profile_id: Optional[str] = None,
    ) -> None:
        now = _now()
        branch_id = uuid.uuid4().hex
        messages = list(messages or [])
        with self._lock:
            self._sessions[session_id] = {
                "current_branch": branch_id,
                "strategy": strategy,
                "window_size": window_size,
                "profile_id": profile_id,
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
                        (session_id, current_branch, strategy, window_size, profile_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, branch_id, strategy, window_size, profile_id, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO branches
                        (branch_id, session_id, name, messages, checkpoint, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (branch_id, session_id, branch_name,
                     json.dumps(messages, ensure_ascii=False), 0, now, now),
                )
                conn.commit()

    def set_session_meta(self, session_id: str, strategy: str, window_size: int) -> bool:
        now = _now()
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
    # Связь «профиль ↔ сессия»
    # ------------------------------------------------------------------
    def get_session_profile(self, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.get("profile_id") if session else None

    def set_session_profile(self, session_id: str, profile_id: Optional[str]) -> bool:
        now = _now()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session["profile_id"] = profile_id
            with closing(self._connect()) as conn:
                conn.execute(
                    "UPDATE conversations SET profile_id = ?, updated_at = ? WHERE session_id = ?",
                    (profile_id, now, session_id),
                )
                conn.commit()
            return True

    # ------------------------------------------------------------------
    # Сообщения (краткосрочная память — активная ветка)
    # ------------------------------------------------------------------
    def load(self, session_id: str) -> Optional[List[dict]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            branch = session["branches"].get(session["current_branch"])
            return list(branch["messages"]) if branch else None

    def load_branch(self, branch_id: str) -> Optional[List[dict]]:
        with self._lock:
            session_id = self._branch_index.get(branch_id)
            if not session_id:
                return None
            branch = self._sessions[session_id]["branches"].get(branch_id)
            return list(branch["messages"]) if branch else None

    def save(self, session_id: str, messages: List[dict]) -> None:
        now = _now()
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
        now = _now()
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
                    (branch_id, session_id, name,
                     json.dumps(messages, ensure_ascii=False), checkpoint, now, now),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()
            return branch_id

    def switch_branch(self, session_id: str, branch_id: str) -> bool:
        now = _now()
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
    # Рабочая память (working)
    # ------------------------------------------------------------------
    def load_working(self, session_id: str) -> List[dict]:
        with self._lock:
            entries = self._working.get(session_id, {})
            return [dict(e) for e in entries.values()]

    def save_working_entry(self, session_id: str, entry: dict) -> dict:
        now = _now()
        key = entry["key"]
        kind = _normalize_kind(entry.get("kind"), WORKING_KINDS, "note")
        state = _normalize_state(entry.get("state"))
        importance = int(entry.get("importance") or 0)
        with self._lock:
            existing = self._working.get(session_id, {}).get(key)
            entry_id = existing["entry_id"] if existing else uuid.uuid4().hex
            created_at = existing["created_at"] if existing else now
            record = {
                "entry_id": entry_id,
                "key": key,
                "value": entry.get("value", ""),
                "kind": kind,
                "state": state,
                "importance": importance,
                "created_at": created_at,
                "updated_at": now,
            }
            self._working.setdefault(session_id, {})[key] = record
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO working_memory
                        (entry_id, session_id, key, value, kind, state, importance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, key) DO UPDATE SET
                        value      = excluded.value,
                        kind       = excluded.kind,
                        state      = excluded.state,
                        importance = excluded.importance,
                        updated_at = excluded.updated_at
                    """,
                    (entry_id, session_id, key, record["value"], kind, state,
                     importance, created_at, now),
                )
                conn.commit()
            return dict(record)

    def delete_working_entry(self, session_id: str, key: str) -> bool:
        with self._lock:
            removed = self._working.get(session_id, {}).pop(key, None)
            if removed is None:
                return False
            with closing(self._connect()) as conn:
                conn.execute(
                    "DELETE FROM working_memory WHERE session_id = ? AND key = ?",
                    (session_id, key),
                )
                conn.commit()
            return True

    def delete_working(self, session_id: str) -> None:
        with self._lock:
            self._working.pop(session_id, None)
            with closing(self._connect()) as conn:
                conn.execute("DELETE FROM working_memory WHERE session_id = ?", (session_id,))
                conn.commit()

    # ------------------------------------------------------------------
    # Долговременная память (long_term)
    # ------------------------------------------------------------------
    def load_long_term(self) -> List[dict]:
        with self._lock:
            return [dict(e) for e in self._long_term.values()]

    def load_long_term_for_profile(self, profile_id: Optional[str]) -> List[dict]:
        """Записи долговременной памяти, видимые профилю.

        Видны общие записи (без привязки, profile_id = NULL) плюс записи,
        привязанные к этому профилю. Если profile_id is None — только общие.
        """
        with self._lock:
            result = []
            for e in self._long_term.values():
                owner = e.get("profile_id") or None
                if owner is None or (profile_id is not None and owner == profile_id):
                    result.append(dict(e))
            return result

    def save_long_term_entry(self, entry: dict) -> dict:
        now = _now()
        key = entry["key"]
        kind = _normalize_kind(entry.get("kind"), LONG_TERM_KINDS, "fact")
        tags = entry.get("tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags) if tags else []
            except Exception:
                tags = []
        tags = list(tags) if isinstance(tags, list) else []
        importance = int(entry.get("importance") or 0)
        with self._lock:
            existing = self._long_term.get(key)
            entry_id = existing["entry_id"] if existing else uuid.uuid4().hex
            created_at = existing["created_at"] if existing else now
            # Привязка к профилю: явная, иначе сохраняем прежнюю (не стираем).
            if "profile_id" in entry:
                profile_id = entry.get("profile_id") or None
            else:
                profile_id = existing.get("profile_id") if existing else None
            record = {
                "entry_id": entry_id,
                "key": key,
                "profile_id": profile_id,
                "value": entry.get("value", ""),
                "kind": kind,
                "tags": tags,
                "importance": importance,
                "created_at": created_at,
                "updated_at": now,
            }
            self._long_term[key] = record
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO long_term_memory
                        (entry_id, key, profile_id, value, kind, tags, importance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        profile_id = excluded.profile_id,
                        value      = excluded.value,
                        kind       = excluded.kind,
                        tags       = excluded.tags,
                        importance = excluded.importance,
                        updated_at = excluded.updated_at
                    """,
                    (entry_id, key, profile_id, record["value"], kind,
                     json.dumps(tags, ensure_ascii=False), importance, created_at, now),
                )
                conn.commit()
            return dict(record)

    def delete_long_term_entry(self, key: str) -> bool:
        with self._lock:
            removed = self._long_term.pop(key, None)
            if removed is None:
                return False
            with closing(self._connect()) as conn:
                conn.execute("DELETE FROM long_term_memory WHERE key = ?", (key,))
                conn.commit()
            return True

    # ------------------------------------------------------------------
    # Профиль пользователя (персонализация)
    # ------------------------------------------------------------------
    def list_profiles(self) -> List[dict]:
        with self._lock:
            return [dict(p) for p in self._profiles.values()]

    def get_profile(self, profile_id: str) -> Optional[dict]:
        with self._lock:
            profile = self._profiles.get(profile_id)
            return dict(profile) if profile else None

    def get_active_profile(self) -> Optional[dict]:
        with self._lock:
            if not self._active_profile_id:
                return None
            profile = self._profiles.get(self._active_profile_id)
            return dict(profile) if profile else None

    def save_profile(self, profile: dict) -> dict:
        """Создаёт или обновляет профиль пользователя.

        Поля: name (уникальный ключ), display_name, style, format, constraints, notes.
        Если профиль первый — он автоматически становится активным.
        """
        now = _now()
        name = (profile.get("name") or "").strip()
        if not name:
            raise ValueError("Для профиля нужен непустой 'name'")
        with self._lock:
            existing = None
            profile_id = (profile.get("profile_id") or "").strip()
            if profile_id and profile_id in self._profiles:
                existing = self._profiles[profile_id]
            else:
                for pid, p in self._profiles.items():
                    if p["name"] == name:
                        existing = p
                        profile_id = pid
                        break

            entry_id = existing["profile_id"] if existing else (profile_id or uuid.uuid4().hex)
            created_at = existing["created_at"] if existing else now
            # Имя должно быть уникальным среди ПРОЧИХ профилей.
            for pid, p in self._profiles.items():
                if p["name"] == name and pid != entry_id:
                    raise ValueError(f"Профиль с именем '{name}' уже существует")

            if existing is not None:
                is_active = existing["is_active"]
            else:
                # Первый профиль автоматически активен.
                is_active = not self._profiles
                if is_active:
                    self._active_profile_id = entry_id

            record = {
                "profile_id": entry_id,
                "name": name,
                "display_name": (profile.get("display_name") or "").strip(),
                "style": (profile.get("style") or "").strip(),
                "format": (profile.get("format") or "").strip(),
                "constraints": (profile.get("constraints") or "").strip(),
                "notes": (profile.get("notes") or "").strip(),
                "is_active": is_active,
                "created_at": created_at,
                "updated_at": now,
            }
            self._profiles[entry_id] = record
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO user_profiles
                        (profile_id, name, display_name, style, format, constraints, notes,
                         is_active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id) DO UPDATE SET
                        name         = excluded.name,
                        display_name = excluded.display_name,
                        style        = excluded.style,
                        format       = excluded.format,
                        constraints  = excluded.constraints,
                        notes        = excluded.notes,
                        is_active    = excluded.is_active,
                        updated_at   = excluded.updated_at
                    """,
                    (entry_id, name, record["display_name"], record["style"], record["format"],
                     record["constraints"], record["notes"], 1 if record["is_active"] else 0,
                     created_at, now),
                )
                conn.commit()
            return dict(record)

    def set_active_profile(self, profile_id: str) -> bool:
        now = _now()
        with self._lock:
            if profile_id not in self._profiles:
                return False
            self._active_profile_id = profile_id
            for pid, p in self._profiles.items():
                p["is_active"] = (pid == profile_id)
            with closing(self._connect()) as conn:
                conn.execute("UPDATE user_profiles SET is_active = 0")
                conn.execute(
                    "UPDATE user_profiles SET is_active = 1, updated_at = ? WHERE profile_id = ?",
                    (now, profile_id),
                )
                conn.commit()
            return True

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock:
            removed = self._profiles.pop(profile_id, None)
            if removed is None:
                return False
            if self._active_profile_id == profile_id:
                self._active_profile_id = None
            # Отвязываем сессии от удалённого профиля.
            for session in self._sessions.values():
                if session.get("profile_id") == profile_id:
                    session["profile_id"] = None
            # Удаляем записи долговременной памяти, привязанные к этому профилю.
            keys_to_delete = [
                k for k, e in self._long_term.items()
                if (e.get("profile_id") or None) == profile_id
            ]
            for k in keys_to_delete:
                self._long_term.pop(k, None)
            with closing(self._connect()) as conn:
                conn.execute("DELETE FROM user_profiles WHERE profile_id = ?", (profile_id,))
                conn.execute("UPDATE conversations SET profile_id = NULL WHERE profile_id = ?", (profile_id,))
                for k in keys_to_delete:
                    conn.execute("DELETE FROM long_term_memory WHERE key = ?", (k,))
                conn.commit()
            return True

    # ------------------------------------------------------------------
    # Операции над памятью (явный выбор «что и куда»)
    # ------------------------------------------------------------------
    def log_memory(self, session_id: str, layer: str, action: str, key: str, source: str) -> None:
        with self._lock:
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO memory_log (session_id, layer, action, key, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, layer, action, key, source or "manual", _now()),
                )
                conn.commit()

    def get_working_entry(self, session_id: str, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._working.get(session_id, {}).get(key)
            return dict(entry) if entry else None

    def get_long_term_entry(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._long_term.get(key)
            return dict(entry) if entry else None

    def apply_memory_op(self, session_id: str, op: dict, source: str = "manual") -> dict:
        """Применяет одну операцию памяти.

        Допустимые операции:
          • save   — сохранить/обновить запись: {"action":"save","layer":..,"key":..,...}
          • delete — удалить: {"action":"delete","layer":..,"key":..}
          • move   — перенести между слоями: {"action":"move","from":..,"to":..,"key":..}
        """
        action = (op.get("action") or "").strip().lower()
        layer = (op.get("layer") or "").strip().lower()
        key = (op.get("key") or "").strip()

        if action == "save":
            if not key:
                raise ValueError("Для save нужен непустой 'key'")
            if layer == "working":
                entry = self.save_working_entry(session_id, {
                    "key": key,
                    "value": op.get("value", ""),
                    "kind": op.get("kind"),
                    "state": op.get("state"),
                    "importance": op.get("importance"),
                })
                self.log_memory(session_id, "working", "save", key, source)
                return {"layer": "working", "entry": entry}
            if layer == "long_term":
                entry_data = {
                    "key": key,
                    "value": op.get("value", ""),
                    "kind": op.get("kind"),
                    "tags": op.get("tags"),
                    "importance": op.get("importance"),
                }
                # Привязка к профилю: явная из op, иначе — профиль сессии
                # (связь «профиль ↔ долговременная память»).
                if "profile_id" in op:
                    entry_data["profile_id"] = op.get("profile_id") or None
                elif session_id:
                    sp = self.get_session_profile(session_id)
                    if sp:
                        entry_data["profile_id"] = sp
                entry = self.save_long_term_entry(entry_data)
                self.log_memory(session_id, "long_term", "save", key, source)
                return {"layer": "long_term", "entry": entry}
            raise ValueError(f"Неизвестный слой памяти: {layer}")

        if action == "delete":
            if layer == "working":
                ok = self.delete_working_entry(session_id, key)
            elif layer == "long_term":
                ok = self.delete_long_term_entry(key)
            else:
                raise ValueError(f"Неизвестный слой памяти: {layer}")
            if not ok:
                raise ValueError(f"Запись не найдена: {layer}/{key}")
            self.log_memory(session_id, layer, "delete", key, source)
            return {"layer": layer, "key": key, "deleted": True}

        if action == "move":
            src = (op.get("from") or "").strip().lower()
            dst = (op.get("to") or "").strip().lower()
            if src not in ("working", "long_term") or dst not in ("working", "long_term"):
                raise ValueError("Для move нужны 'from' и 'to' из {working, long_term}")
            if src == dst:
                raise ValueError("from и to не должны совпадать")
            if src == "working":
                src_entry = self.get_working_entry(session_id, key)
                if src_entry is None:
                    raise ValueError(f"Запись не найдена: working/{key}")
                self.delete_working_entry(session_id, key)
                profile_id = op.get("profile_id")
                if profile_id is None and session_id:
                    profile_id = self.get_session_profile(session_id)
                target = self.save_long_term_entry({
                    "key": key,
                    "profile_id": profile_id,
                    "value": src_entry["value"],
                    "kind": op.get("kind") or ("fact" if src_entry["kind"] == "fact" else src_entry["kind"]),
                    "tags": op.get("tags") or [],
                    "importance": src_entry["importance"],
                })
            else:
                src_entry = self.get_long_term_entry(key)
                if src_entry is None:
                    raise ValueError(f"Запись не найдена: long_term/{key}")
                self.delete_long_term_entry(key)
                target = self.save_working_entry(session_id, {
                    "key": key,
                    "value": src_entry["value"],
                    "kind": op.get("kind") or ("note" if src_entry["kind"] == "fact" else src_entry["kind"]),
                    "state": op.get("state"),
                    "importance": src_entry["importance"],
                })
            self.log_memory(session_id, dst, "move", key, source)
            return {"layer": dst, "entry": target, "moved_from": src}

        raise ValueError(f"Неизвестное действие памяти: {action}")

    def list_memory(self, session_id: str) -> dict:
        """Снимок всех трёх слоёв памяти для UI.

        Долговременная память возвращается в разрезе профиля сессии:
        общие записи + привязанные к профилю этой сессии.
        """
        working = self.load_working(session_id)
        long_term = self.load_long_term_for_profile(self.get_session_profile(session_id))
        return {
            "short_term": {"message_count": len(self.load(session_id) or [])},
            "working": working,
            "long_term": long_term,
        }

    # ------------------------------------------------------------------
    # Управление сессиями
    # ------------------------------------------------------------------
    def delete(self, session_id: str) -> bool:
        """Удаляет сессию целиком: ветки, рабочую память и метаданные.

        Долговременная память НЕ удаляется — она глобальная и переживает сессии.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            for branch_id in session["branches"]:
                self._branch_index.pop(branch_id, None)
            self._working.pop(session_id, None)
            with closing(self._connect()) as conn:
                conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM branches WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM working_memory WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM memory_log WHERE session_id = ?", (session_id,))
                conn.commit()
            return True

    def list_sessions(self) -> List[dict]:
        with self._lock:
            result = []
            for sid, session in self._sessions.items():
                branch = session["branches"].get(session["current_branch"])
                result.append(
                    {
                        "session_id": sid,
                        "strategy": session["strategy"],
                        "profile_id": session.get("profile_id"),
                        "message_count": len(branch["messages"]) if branch else 0,
                        "branch_count": len(session["branches"]),
                        "working_count": len(self._working.get(sid, {})),
                        "long_term_count": len(self._long_term),
                    }
                )
            return result


# Синглтон хранилища: история и память загружаются из БД при импорте модуля.
storage = HistoryStorage()
