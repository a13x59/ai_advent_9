# test_mcp_tools.py
"""Тесты подключения MCP-инструментов: клиент, маппинг схем и сквозной пайплайн.

Все тесты идут БЕЗ сети: клиент тестируется на фейковом транспорте, а пайплайн —
на детерминированном FakeMcpClient + ToolCallingProvider.
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import mcp_client
from agent_core import create_app, _deepseek_tools, _tools_enabled
from mock_agent import MockProvider
from storage import HistoryStorage
from mcp_client import McpUnavailable


# ----------------------------------------------------------------------
# Фейковый транспорт для юнит-теста FortuneMcpClient
# ----------------------------------------------------------------------
class FakeResp:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class FakePost:
    def __init__(self):
        self.calls = []
        self.by_method = {}

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        method = (json or {}).get("method")
        # Уведомление (нет id) — отвечаем 202 без тела.
        if "id" not in (json or {}):
            return FakeResp(status_code=202, json_data=None, headers={}, text="")
        results = self.by_method.get(method) or []
        result = results.pop(0) if results else None
        if isinstance(result, Exception):
            raise result
        return FakeResp(
            status_code=200,
            json_data={"jsonrpc": "2.0", "id": (json or {}).get("id"), "result": result},
            headers={"Mcp-Session-Id": "sess-test"},
            text="",
        )


def test_client_list_and_call_tools(monkeypatch):
    fake = FakePost()
    fake.by_method = {
        "initialize": [{
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fortune-telling-mcp", "version": "1.0.0"},
        }],
        "tools/list": [{"tools": [{
            "name": "magic_8_ball",
            "description": "Magic 8 Ball",
            "inputSchema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        }]}],
        "tools/call": [{
            "content": [{"type": "text", "text": "❓q 🔮 Yes."}],
            "isError": False,
        }],
    }
    monkeypatch.setattr(mcp_client.requests, "post", fake)

    client = mcp_client.FortuneMcpClient(base_url="http://localhost:9999")
    tools = client.list_tools()
    assert tools[0]["name"] == "magic_8_ball"

    res = client.call_tool("magic_8_ball", {"question": "q"})
    assert res == {"ok": True, "text": "❓q 🔮 Yes."}

    # Порядок JSON-RPC: initialize → notification → tools/list → tools/call.
    methods = [c["json"]["method"] for c in fake.calls]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    # Идентификатор сессии пробрасывается в заголовок после initialize.
    assert fake.calls[-1]["headers"].get("Mcp-Session-Id") == "sess-test"


def test_client_transport_error_raises_mcp_unavailable(monkeypatch):
    fake = FakePost()
    fake.by_method = {"initialize": [McpUnavailable("boom")]}
    monkeypatch.setattr(mcp_client.requests, "post", fake)

    client = mcp_client.FortuneMcpClient(base_url="http://localhost:9999")
    with pytest.raises(McpUnavailable):
        client.list_tools()


# ----------------------------------------------------------------------
# Маппинг схем и гейт инструментов
# ----------------------------------------------------------------------
def test_deepseek_tools_mapping():
    tools = _deepseek_tools([{
        "name": "magic_8_ball",
        "description": "Magic 8 Ball",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    }])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "magic_8_ball"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert tools[0]["function"]["parameters"]["required"] == ["question"]


def test_tools_enabled_gate():
    class Req:
        enable_tools = True
        model = "deepseek-chat"

    assert _tools_enabled(Req(), object()) is True
    Req.model = "deepseek-reasoner"
    assert _tools_enabled(Req(), object()) is False
    Req.model = "deepseek-chat"
    Req.enable_tools = False
    assert _tools_enabled(Req(), object()) is False
    assert _tools_enabled(Req(), None) is False


# ----------------------------------------------------------------------
# Сквозной пайплайн (без сети)
# ----------------------------------------------------------------------
class FakeMcpClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {"name": "magic_8_ball", "description": "Magic 8 Ball",
             "inputSchema": {"type": "object",
                             "properties": {"question": {"type": "string"}},
                             "required": ["question"]}},
            {"name": "bibliomancy", "description": "Bibliomancy",
             "inputSchema": {"type": "object",
                             "properties": {
                                 "question": {"type": "string"},
                                 "book": {"type": "string", "enum": ["the_art_of_war"]},
                                 "line": {"type": "string", "enum": ["top", "bottom"]},
                             },
                             "required": ["question", "book", "line"]}},
        ]

    def call_tool(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        if name == "magic_8_ball":
            return {"ok": True, "text": "❓Сдам ли я собеседование? 🔮 Signs point to yes."}
        return {"ok": True, "text": "❓... 📚 The Art of War ..."}


class ToolCallingProvider(MockProvider):
    """На 1-м вызове (с tools) отдаёт tool_calls, на 2-м — финальный ответ."""

    def __init__(self):
        super().__init__()
        self.rounds = 0

    def complete(self, messages, request, task, user_text, tools=None):
        if tools and self.rounds == 0:
            self.rounds += 1
            return {
                "id": "mock-tool",
                "choices": [{"message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "magic_8_ball",
                            "arguments": json.dumps({"question": "Сдам ли я собеседование?"}),
                        },
                    }],
                }}],
            }
        self.rounds += 1
        return {
            "id": "mock-final",
            "choices": [{"message": {
                "role": "assistant",
                "content": "Шар ответил: Signs point to yes. Похоже, всё получится!",
            }}],
        }


def _make_app(mcp_client=None, provider=None):
    fd, path = tempfile.mkstemp(prefix="tool_", suffix=".db")
    os.close(fd)
    app = create_app(provider or MockProvider(), HistoryStorage(path), mcp_client=mcp_client)
    return app, path


def _post(app, text, model="deepseek-chat", enable_tools=True, session_id="sess1"):
    with TestClient(app) as c:
        return c.post("/agent", json={
            "session_id": session_id,
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "enable_tools": enable_tools,
        })


def test_full_tool_call_flow():
    mcp = FakeMcpClient()
    app, path = _make_app(mcp_client=mcp, provider=ToolCallingProvider())
    try:
        resp = _post(app, "Сдам ли я собеседование?")
        assert resp.status_code == 200
        r = resp.json()

        # Модель использовала результат инструмента в финальном ответе.
        assert "Шар ответил" in r["response"]
        # Информация о вызове вернулась клиенту.
        assert r["tool_calls"]
        tc = r["tool_calls"][0]
        assert tc["name"] == "magic_8_ball"
        assert tc["ok"] is True
        assert "Signs point to yes" in tc["result"]
        # Инструмент реально вызван с аргументами.
        assert mcp.calls[0]["name"] == "magic_8_ball"
        assert mcp.calls[0]["arguments"]["question"] == "Сдам ли я собеседование?"
        # Результат сохранён в рабочую память.
        assert any(e["kind"] == "result" for e in r["memory"]["working"])
    finally:
        os.remove(path)


def test_mcp_unavailable_is_graceful():
    class FailingMcpClient(FakeMcpClient):
        def call_tool(self, name, arguments):
            raise McpUnavailable("сервис гадания лежит")

    app, path = _make_app(mcp_client=FailingMcpClient(), provider=ToolCallingProvider())
    try:
        resp = _post(app, "погадай")
        assert resp.status_code == 200  # не 500
        r = resp.json()
        assert r["tool_calls"][0]["ok"] is False
        assert r["tool_calls"][0]["error"]
    finally:
        os.remove(path)


def test_non_capable_model_skips_tools():
    mcp = FakeMcpClient()
    app, path = _make_app(mcp_client=mcp, provider=MockProvider())
    try:
        r = _post(app, "погадай", model="deepseek-reasoner").json()
        assert r["tool_calls"] == []
        assert mcp.calls == []
    finally:
        os.remove(path)


def test_tools_disabled_skips_mcp():
    mcp = FakeMcpClient()
    app, path = _make_app(mcp_client=mcp, provider=MockProvider())
    try:
        r = _post(app, "погадай", enable_tools=False).json()
        assert r["tool_calls"] == []
        assert mcp.calls == []
    finally:
        os.remove(path)
