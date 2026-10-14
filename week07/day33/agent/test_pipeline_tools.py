# test_pipeline_tools.py
"""Тесты цепочки MCP-инструментов search → summarize → save_to_file.

Без сети: детерминированный PipelineFakeMcpClient + многораундовый провайдер,
который эмулирует LLM, последовательно строящую цепочку и передающую результат
предыдущего инструмента в аргументы следующего.
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import mcp_client
from agent_core import create_app, _deepseek_tools, MAX_TOOL_CYCLES
from mock_agent import MockProvider
from mcp_client import McpUnavailable
from storage import HistoryStorage

SEARCH_RESULT = '{"query":"deepseek","count":1,"results":[{"title":"DeepSeek","url":"https://deepseek.com","passage":"DeepSeek is an AI model."}]}'
SUMMARIZE_RESULT = '{"summary":"DeepSeek — AI-модель.","input_chars":120}'
SAVE_RESULT = '{"path":"/out/summary.txt","filename":"summary.txt","bytes":42}'


class PipelineFakeMcpClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {"name": "search", "description": "web search",
             "inputSchema": {"type": "object",
                             "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                             "required": ["query"]}},
            {"name": "summarize", "description": "summarize text",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]}},
            {"name": "save_to_file", "description": "save to file",
             "inputSchema": {"type": "object",
                             "properties": {"filename": {"type": "string"}, "content": {"type": "string"}},
                             "required": ["filename", "content"]}},
        ]

    def call_tool(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        if name == "search":
            return {"ok": True, "text": SEARCH_RESULT}
        if name == "summarize":
            return {"ok": True, "text": SUMMARIZE_RESULT}
        if name == "save_to_file":
            return {"ok": True, "text": SAVE_RESULT}
        return {"ok": False, "text": "", "error": "unknown tool"}


class PipelineProvider(MockProvider):
    """Эмулирует модель, которая строит цепочку и передаёт данные между шагами.

    Раунд 1: search(query); раунд 2: summarize(text=<результат search>);
    раунд 3: save_to_file(content=<результат summarize>); раунд 4: финальный ответ.
    """

    def __init__(self):
        super().__init__()
        self.step = 0

    def complete(self, messages, request, task, user_text, tools=None):
        self.step += 1
        if self.step == 1:
            return self._tool_call("call_search", "search", {"query": "deepseek"})
        if self.step == 2:
            return self._tool_call("call_summarize", "summarize", {"text": self._last_tool_text(messages)})
        if self.step == 3:
            return self._tool_call(
                "call_save", "save_to_file",
                {"filename": "summary.txt", "content": self._last_tool_text(messages)},
            )
        return {"id": "mock-final", "choices": [{"message": {
            "role": "assistant", "content": "Сводка сохранена в summary.txt",
        }}]}

    @staticmethod
    def _tool_call(call_id, name, arguments):
        return {"id": "mock", "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
        }}]}

    @staticmethod
    def _last_tool_text(messages):
        """Последний результат инструмента в диалоге (то, что агент подставил модели)."""
        for m in reversed(messages):
            if m.get("role") == "tool":
                return m.get("content", "")
        return ""


class SingleCallProvider(MockProvider):
    """Один раунд вызова инструмента + финальный ответ (для проверки _run_tool_turn)."""

    def __init__(self):
        super().__init__()
        self.step = 0

    def complete(self, messages, request, task, user_text, tools=None):
        self.step += 1
        if self.step == 1:
            return {"id": "mock", "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c", "type": "function",
                                "function": {"name": "search", "arguments": json.dumps({"query": "deepseek"})}}],
            }}]}
        return {"id": "mock-final", "choices": [{"message": {
            "role": "assistant", "content": "Нашёл DeepSeek.",
        }}]}


class NeverEndingProvider(MockProvider):
    """Всегда возвращает tool_calls — для проверки лимита циклов _run_tool_loop."""

    def complete(self, messages, request, task, user_text, tools=None):
        return {"id": "mock", "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c", "type": "function",
                            "function": {"name": "search", "arguments": json.dumps({"query": "x"})}}],
        }}]}


def _make_app(mcp_client=None, provider=None):
    fd, path = tempfile.mkstemp(prefix="pipe_", suffix=".db")
    os.close(fd)
    app = create_app(provider or MockProvider(), HistoryStorage(path), mcp_client=mcp_client)
    return app, path


def _post(app, text, pipeline=True, enable_tools=True, model="deepseek-chat", session_id="sess1"):
    with TestClient(app) as c:
        return c.post("/agent", json={
            "session_id": session_id,
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "enable_tools": enable_tools,
            "pipeline": pipeline,
        })


def test_pipeline_schema_mapping():
    tools = _deepseek_tools([
        {"name": "search", "description": "d",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "summarize", "description": "d",
         "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        {"name": "save_to_file", "description": "d",
         "inputSchema": {"type": "object", "properties": {}, "required": []}},
    ])
    assert [t["function"]["name"] for t in tools] == ["search", "summarize", "save_to_file"]
    assert all(t["type"] == "function" for t in tools)
    assert tools[2]["function"]["parameters"]["type"] == "object"


def test_pipeline_runs_chain_and_passes_data():
    mcp = PipelineFakeMcpClient()
    provider = PipelineProvider()
    app, path = _make_app(mcp_client=mcp, provider=provider)
    try:
        resp = _post(app, "найди информацию про deepseek и сохрани сводку в файл")
        assert resp.status_code == 200
        r = resp.json()

        # 1. Цепочка выполнилась АВТОМАТИЧЕСКИ: 3 вызова + финальный ответ.
        assert [c["name"] for c in mcp.calls] == ["search", "summarize", "save_to_file"]
        assert provider.step == 4  # 3 раунда вызовов + финальный ответ

        # 2. Данные переданы КОРРЕКТНО: search → summarize, summarize → save_to_file.
        assert mcp.calls[0]["arguments"]["query"] == "deepseek"
        assert mcp.calls[1]["arguments"]["text"] == SEARCH_RESULT
        assert mcp.calls[2]["arguments"]["content"] == SUMMARIZE_RESULT
        assert mcp.calls[2]["arguments"]["filename"] == "summary.txt"

        # 3. Полный трейс цепочки вернулся клиенту в порядке выполнения.
        assert [tc["name"] for tc in r["tool_calls"]] == ["search", "summarize", "save_to_file"]
        assert all(tc["ok"] for tc in r["tool_calls"])

        # 4. Финальный ответ модели использует результат цепочки.
        assert "summary.txt" in r["response"]
    finally:
        os.remove(path)


def test_pipeline_loop_respects_cycle_limit():
    mcp = PipelineFakeMcpClient()
    provider = NeverEndingProvider()
    app, path = _make_app(mcp_client=mcp, provider=provider)
    try:
        resp = _post(app, "зацикли меня")
        assert resp.status_code == 200
        # Лимит циклов MAX_TOOL_CYCLES защищает от бесконечного выполнения.
        assert len(mcp.calls) == MAX_TOOL_CYCLES
    finally:
        os.remove(path)


def test_single_call_still_uses_single_turn():
    # pipeline=False использует _run_tool_turn: один раунд вызовов + финальный ответ.
    mcp = PipelineFakeMcpClient()
    provider = SingleCallProvider()
    app, path = _make_app(mcp_client=mcp, provider=provider)
    try:
        resp = _post(app, "поищи deepseek", pipeline=False)
        assert resp.status_code == 200
        r = resp.json()
        assert [c["name"] for c in mcp.calls] == ["search"]
        assert provider.step == 2  # один раунд вызовов + финальный ответ
        assert r["response"] == "Нашёл DeepSeek."
    finally:
        os.remove(path)


# ----------------------------------------------------------------------
# Восстановление клиента после перезапуска MCP-сервера (протухшая сессия).
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


class RestartingFakePost:
    """Транспорт, который эмулирует перезапуск MCP-сервера: после restart()
    ранее выданные session id перестают работать (HTTP 404 «Session not found»)."""

    def __init__(self):
        self.active_sessions = set()
        self.calls = []
        self._session_counter = 0

    def restart(self):
        self.active_sessions.clear()

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"json": json, "headers": headers})
        method = (json or {}).get("method")
        sid = (headers or {}).get("Mcp-Session-Id")
        req_id = (json or {}).get("id")

        # Уведомления (нет id) — 202 без тела.
        if "id" not in (json or {}):
            return FakeResp(status_code=202, json_data=None, headers={}, text="")

        # initialize всегда выдаёт новую валидную сессию.
        if method == "initialize":
            self._session_counter += 1
            new_sid = f"sess-{self._session_counter}"
            self.active_sessions.add(new_sid)
            return FakeResp(
                status_code=200,
                json_data={"jsonrpc": "2.0", "id": req_id, "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "pipeline-mcp", "version": "1.0.0"},
                }},
                headers={"Mcp-Session-Id": new_sid},
                text="",
            )

        # Остальные запросы требуют валидной сессии.
        if sid not in self.active_sessions:
            return FakeResp(
                status_code=404,
                json_data={"jsonrpc": "2.0", "id": req_id,
                           "error": {"code": -32000, "message": "Session not found"}},
                headers={},
                text="",
            )

        if method == "tools/list":
            return FakeResp(status_code=200, json_data={"jsonrpc": "2.0", "id": req_id, "result": {
                "tools": [{"name": "save_to_file", "description": "d",
                           "inputSchema": {"type": "object", "properties": {}, "required": []}}],
            }}, headers={}, text="")
        if method == "tools/call":
            return FakeResp(status_code=200, json_data={"jsonrpc": "2.0", "id": req_id, "result": {
                "content": [{"type": "text", "text": "ok"}],
            }}, headers={}, text="")
        return FakeResp(status_code=200, json_data={"jsonrpc": "2.0", "id": req_id, "result": {}}, headers={}, text="")


def test_client_recovers_from_stale_session(monkeypatch):
    fake = RestartingFakePost()
    monkeypatch.setattr(mcp_client.requests, "post", fake)

    client = mcp_client.PipelineMcpClient(base_url="http://localhost:9999")
    client.list_tools()
    assert client._session_id == "sess-1"

    # Имитируем перезапуск сервера: старая сессия больше не валидна.
    fake.restart()

    # tools/call со старой сессией → 404 → клиент должен переинициализироваться и повторить.
    res = client.call_tool("save_to_file", {"filename": "a.txt", "content": "x"})
    assert res == {"ok": True, "text": "ok"}
    assert client._session_id == "sess-2"

    # initialize вызывался дважды: до рестарта и при восстановлении.
    init_count = sum(1 for c in fake.calls if (c["json"] or {}).get("method") == "initialize")
    assert init_count == 2
