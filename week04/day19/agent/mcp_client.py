# mcp_client.py
"""Синхронный MCP-клиент для fortune-telling-mcp (Streamable HTTP, JSON-RPC 2.0).

Агент подключается к /mcp, узнаёт доступные инструменты через tools/list и
вызывает их через tools/call. Клиент намеренно синхронный (requests), чтобы
вписаться в синхронный пайплайн агента (провайдер и хранилище синхронные).

Протокол Streamable HTTP: клиент шлёт JSON-RPC POST-запросы на /mcp; для
initialize/tools/list/tools/call сервер отвечает обычным JSON, а идентификатор
сессии передаётся в заголовке Mcp-Session-Id. Для уведомлений
(notifications/initialized) ожидается 202 с пустым телом.
"""
import json
import os

import requests

MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "http://localhost:8888")
CURRENCY_MCP_BASE_URL = os.environ.get("CURRENCY_MCP_BASE_URL", "http://localhost:8889")
PIPELINE_MCP_BASE_URL = os.environ.get("PIPELINE_MCP_BASE_URL", "http://localhost:8890")
MCP_TIMEOUT = float(os.environ.get("MCP_TIMEOUT", "10"))

# Версия протокола, которую клиент заявляет при initialize.
PROTOCOL_VERSION = "2025-06-18"


class McpUnavailable(Exception):
    """MCP-сервис недоступен или вернул ошибку на транспортном/протокольном уровне."""


class McpClient:
    """Универсальный MCP-клиент (Streamable HTTP, JSON-RPC 2.0).

    Интерфейс намеренно похож на MCP-клиент: list_tools() / call_tool(name, args),
    чтобы транспорт можно было заменить, не трогая пайплайн агента.
    """

    CLIENT_NAME = "agent-mcp-client"

    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = (base_url or MCP_BASE_URL).rstrip("/")
        self.mcp_url = self.base_url + "/mcp"
        self.timeout = timeout if timeout is not None else MCP_TIMEOUT
        self._session_id = None
        self._request_id = 0
        self._tools = None

    # ------------------------------------------------------------------
    # Инфраструктура JSON-RPC / Streamable HTTP
    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _parse_sse(self, text: str):
        """Минимальный разбор SSE-потока: возвращает последний JSON-RPC-объект."""
        results = []
        for line in (text or "").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                results.append(json.loads(payload))
            except ValueError:
                continue
        if not results:
            raise McpUnavailable("MCP вернул пустой SSE-поток")
        return results[-1]

    def _post(self, payload: dict):
        try:
            resp = requests.post(
                self.mcp_url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise McpUnavailable(f"не удалось достучаться до MCP: {e}") from e

        if resp.status_code not in (200, 202):
            raise McpUnavailable(f"MCP вернул HTTP {resp.status_code}")

        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            return self._parse_sse(resp.text)
        try:
            return resp.json()
        except ValueError:
            # Уведомления (notifications/initialized) отвечают пустым телом.
            return {}

    def _rpc(self, method: str, params: dict = None):
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
        if params is not None:
            payload["params"] = params
        data = self._post(payload)
        if isinstance(data, dict) and data.get("jsonrpc") == "2.0":
            if "error" in data:
                err = data["error"] or {}
                raise McpUnavailable(
                    f"MCP error {err.get('code')}: {err.get('message')}"
                )
            return data.get("result")
        return data

    def _initialize(self) -> dict:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.CLIENT_NAME, "version": "1.0.0"},
            },
        )
        # Уведомление «инициализация завершена» — без id и без ожидания ответа.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result or {}

    def _ensure_session(self) -> None:
        if self._session_id is None:
            self._initialize()

    # ------------------------------------------------------------------
    # Публичный интерфейс инструментов
    # ------------------------------------------------------------------
    def list_tools(self) -> list:
        """Список доступных инструментов (кэшируется после первого запроса)."""
        if self._tools is None:
            self._ensure_session()
            result = self._rpc("tools/list")
            tools = result.get("tools") if isinstance(result, dict) else []
            self._tools = [t for t in tools if isinstance(t, dict)]
        return [dict(t) for t in self._tools]

    @staticmethod
    def _extract_text(result) -> str:
        """Вытаскивает текст из CallToolResult (content: [TextContent, ...])."""
        if not isinstance(result, dict):
            return str(result)
        content = result.get("content") or []
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        if parts:
            return "\n".join(parts)
        return str(result.get("content", ""))

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Вызывает инструмент и возвращает {"ok": bool, "text": str}.

        ok=False означает, что инструмент вернул isError=true; транспортные
        ошибки поднимаются как McpUnavailable.
        """
        self._ensure_session()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        is_error = bool(result.get("isError")) if isinstance(result, dict) else False
        return {"ok": not is_error, "text": self._extract_text(result)}


class FortuneMcpClient(McpClient):
    """Клиент инструментов гадания (magic_8_ball, bibliomancy)."""

    CLIENT_NAME = "agent-fortune-client"

    def __init__(self, base_url: str = None, timeout: float = None):
        super().__init__(base_url=base_url or MCP_BASE_URL, timeout=timeout)


class CurrencyMcpClient(McpClient):
    """Клиент инструментов валют (get_rate, get_summary)."""

    CLIENT_NAME = "agent-currency-client"

    def __init__(self, base_url: str = None, timeout: float = None):
        super().__init__(base_url=base_url or CURRENCY_MCP_BASE_URL, timeout=timeout)


class PipelineMcpClient(McpClient):
    """Клиент инструментов пайплайна (search, summarize, save_to_file)."""

    CLIENT_NAME = "agent-pipeline-client"

    def __init__(self, base_url: str = None, timeout: float = None):
        super().__init__(base_url=base_url or PIPELINE_MCP_BASE_URL, timeout=timeout)
