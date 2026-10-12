# test_service_limits.py
"""Тесты сервисных ограничений приватного AI-сервиса.

Проверяют HTTP-слой (не модель): /healthz, /limits, rate limit 429 и
защиту от переполнения контекста (413). Провайдер — детерминированный мок,
сеть и ключ не нужны.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app
from mock_agent import MockProvider
from storage import HistoryStorage


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("AGENT_MAX_PROMPT_TOKENS", "200")
    fd, path = tempfile.mkstemp(prefix="limits_", suffix=".db")
    os.close(fd)
    app = create_app(MockProvider(), HistoryStorage(path))
    with TestClient(app) as c:
        c.app = app
        yield c
    os.remove(path)


def _post(client, text="привет", session_id=None):
    body = {"model": "mock-model", "messages": [{"role": "user", "content": text}]}
    if session_id:
        body["session_id"] = session_id
    return client.post("/agent", json=body)


def test_healthz_and_limits(client):
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    limits = client.get("/limits")
    assert limits.status_code == 200
    body = limits.json()
    assert body["rate_limit"] == "3/minute"
    assert body["max_prompt_tokens"] == 200


def test_rate_limit_returns_429(client):
    for _ in range(3):
        assert _post(client, session_id="rl").status_code == 200
    r = _post(client, session_id="rl")
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert r.headers["X-RateLimit-Limit"] == "3"


def test_healthz_not_rate_limited(client):
    # healthz исключён из rate limit — даже после исчерпания лимита отдаёт 200.
    for _ in range(3):
        _post(client, session_id="rl2")
    assert client.get("/healthz").status_code == 200


def test_context_guard_returns_413(client):
    long_text = ("слово " * 500).strip()
    r = _post(client, text=long_text, session_id="ctx")
    assert r.status_code == 413
    assert "превышает лимит" in r.json()["detail"]
