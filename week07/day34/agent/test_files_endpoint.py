# test_files_endpoint.py
"""Тесты эндпоинта /files: dry-run/apply, автономный вызов инструментов, 503/400.

Используют мок-провайдер, фейковый files-mcp клиент (стейджинг в памяти) и
скриптованный провайдер, который сам запрашивает инструмент — так проверяется,
что ассистент ИНИЦИИРУЕТ работу с файлами (function calling), а эндпоинт
детерминированно применяет/отбрасывает изменения.
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app
from storage import HistoryStorage
from mock_agent import MockProvider


class FakeFilesClient:
    """Эмуляция files-mcp: стейджинг + diff/apply/discard в памяти."""

    SERVER_NAME = "files-mcp"

    def __init__(self, files=None):
        self.disk = dict(files or {})
        self.staged = {}
        self.applied_log = []

    def list_tools(self):
        def tool(name, desc, props, required=None):
            return {
                "name": name,
                "description": desc,
                "inputSchema": {"type": "object", "properties": props, "required": required or []},
            }

        return [
            tool("list_dir", "list files", {"path": {"type": "string"}}),
            tool("read_file", "read file", {"path": {"type": "string"}}, ["path"]),
            tool("search_files", "search files", {"query": {"type": "string"}}, ["query"]),
            tool("write_file", "write file",
                 {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            tool("edit_file", "edit file",
                 {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                 ["path", "old_string", "new_string"]),
            tool("diff", "show diff", {}),
            tool("apply_changes", "apply changes", {}),
            tool("discard_changes", "discard changes", {}),
        ]

    def call_tool(self, name, arguments):
        arguments = arguments or {}
        if name == "list_dir":
            return self._ok({"count": len(self.disk), "truncated": False, "files": sorted(self.disk)})
        if name == "read_file":
            path = arguments["path"]
            if path not in self.disk:
                return self._err(f"file not found: {path}")
            return self._ok({"path": path, "total_lines": 1, "content": self.disk[path]})
        if name == "search_files":
            q = arguments.get("query", "")
            matches = [
                {"file": f, "line": 1, "text": c}
                for f, c in self.disk.items() if q.lower() in c.lower()
            ]
            return self._ok({"query": q, "count": len(matches), "truncated": False, "matches": matches})
        if name == "write_file":
            path, content = arguments["path"], arguments["content"]
            status = "created" if path not in self.disk else "modified"
            self.staged[path] = content
            return self._ok({
                "path": path, "status": status,
                "diff": f"--- a/{path}\n+++ b/{path}\n+{content}",
            })
        if name == "edit_file":
            path = arguments["path"]
            old, new = arguments["old_string"], arguments["new_string"]
            base = self.staged.get(path, self.disk.get(path))
            if base is None:
                return self._err(f"file not found: {path}")
            if old not in base:
                return self._err(f"old_string not found in {path}")
            self.staged[path] = base.replace(old, new)
            return self._ok({
                "path": path, "status": "edited", "occurrences": 1,
                "diff": f"--- a/{path}\n+++ b/{path}\n-{old}\n+{new}",
            })
        if name == "diff":
            diff = "\n".join(
                f"--- a/{p}\n+++ b/{p}\n-{self.disk.get(p, '')}\n+{c}"
                for p, c in self.staged.items()
            )
            return self._ok({"diff": diff})
        if name == "apply_changes":
            applied = list(self.staged)
            for p in applied:
                self.disk[p] = self.staged[p]
            diff = "\n".join(f"--- a/{p}\n+++ b/{p}\n+{self.disk[p]}" for p in applied)
            self.applied_log.append(applied)
            self.staged.clear()
            return self._ok({"applied": applied, "diff": diff})
        if name == "discard_changes":
            n = len(self.staged)
            self.staged.clear()
            return self._ok({"discarded": n})
        return self._err("unknown tool")

    def _ok(self, data):
        return {"ok": True, "text": json.dumps(data, ensure_ascii=False)}

    def _err(self, msg):
        return {"ok": False, "text": msg}


class ScriptedToolsProvider(MockProvider):
    """Провайдер, который САМ запрашивает инструмент на первом раунде, затем отвечает."""

    def __init__(self, tool_call, final="Готово."):
        super().__init__()
        self.tool_call = tool_call
        self.final = final
        self.turns = 0

    def complete(self, messages, request, task, user_text, tools=None):
        self.turns += 1
        if self.turns == 1 and tools:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": self.tool_call["name"],
                                "arguments": json.dumps(self.tool_call.get("arguments", {})),
                            },
                        }],
                    },
                }],
            }
        return self._response(self.final)


def _make_client(store_path, provider, files_client=None):
    clients = [files_client] if files_client is not None else []
    app = create_app(provider, HistoryStorage(store_path), mcp_clients=clients)
    client = TestClient(app)
    client.files = files_client
    return client


def _tmp_client(provider, files_client):
    fd, path = tempfile.mkstemp(prefix="files_", suffix=".db")
    os.close(fd)
    client = _make_client(path, provider, files_client)
    yield client
    client.close()
    os.remove(path)


@pytest.fixture()
def fc():
    return FakeFilesClient()


def test_missing_files_client_returns_503():
    fd, path = tempfile.mkstemp(prefix="files_nc_", suffix=".db")
    os.close(fd)
    try:
        app = create_app(MockProvider(), HistoryStorage(path), mcp_clients=[])
        c = TestClient(app)
        resp = c.post("/files", json={"message": "найди X"})
        assert resp.status_code == 503
        assert "files-mcp" in resp.json()["detail"]
    finally:
        c.close()
        os.remove(path)


def test_empty_goal_returns_400(fc):
    for client in _tmp_client(MockProvider(), fc):
        resp = client.post("/files", json={"message": "   "})
        assert resp.status_code == 400


def test_dry_run_returns_diff_without_applying(fc):
    provider = ScriptedToolsProvider(
        {"name": "write_file", "arguments": {"path": "README.md", "content": "# Hi"}}
    )
    for client in _tmp_client(provider, fc):
        r = client.post(
            "/files", json={"message": "создай README", "apply": False}
        ).json()

        assert r["applied"] is False
        assert r["changed_files"] == ["README.md"]
        assert "+# Hi" in r["diff"]
        # Изменения НЕ сохранились: диск пуст, стейджинг очищен.
        assert "README.md" not in fc.disk
        assert fc.staged == {}
        # Агент инициировал вызов инструмента (трейс заполнен).
        assert r["tool_calls"][0]["name"] == "write_file"
        assert r["tool_calls"][0]["ok"] is True


def test_apply_writes_files_and_returns_diff(fc):
    provider = ScriptedToolsProvider(
        {"name": "write_file", "arguments": {"path": "ADR-001.md", "content": "# ADR"}}
    )
    for client in _tmp_client(provider, fc):
        r = client.post(
            "/files", json={"message": "сгенерируй ADR", "apply": True}
        ).json()

        assert r["applied"] is True
        assert r["changed_files"] == ["ADR-001.md"]
        assert fc.disk.get("ADR-001.md") == "# ADR"
        assert "# ADR" in r["diff"]
        assert fc.staged == {}


def test_read_only_scenario_has_empty_changes(fc):
    provider = ScriptedToolsProvider(
        {"name": "search_files", "arguments": {"query": "api"}},
        final="Найдено 2 вхождения.",
    )
    for client in _tmp_client(provider, fc):
        r = client.post(
            "/files", json={"message": "найди все места использования api", "apply": False}
        ).json()

        assert r["applied"] is False
        assert r["changed_files"] == []
        assert r["tool_calls"][0]["name"] == "search_files"
        assert "Найдено 2 вхождения." in r["response"]
