# test_service_limits.py
"""Тесты сервисных ограничений приватного AI-сервиса.

Проверяют HTTP-слой (не модель): /healthz, /limits, rate limit 429 и
защиту от переполнения контекста (413 + обрезка старых сообщений).
Провайдер — детерминированный мок, сеть и ключ не нужны.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app, _trim_messages_to_budget
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
    # Один запрос, который сам по себе не влезает в бюджет, — 413.
    long_text = ("слово " * 500).strip()
    r = _post(client, text=long_text, session_id="ctx")
    assert r.status_code == 413
    assert "превышает лимит" in r.json()["detail"]


def test_context_guard_trims_old_messages(client):
    # История суммарно превышает бюджет (130+130 токенов), но последнее
    # сообщение влезает: старые обрезаются, запрос проходит (200), не 413.
    for text in ("a " * 100, "b " * 100):
        r = client.post("/agent", json={
            "session_id": "trim", "model": "mock-model",
            "messages": [{"role": "user", "content": text}],
        })
        assert r.status_code == 200
    assert "b b" in r.json()["response"]


def test_trim_messages_to_budget_keeps_system_and_last():
    # Системные блоки и последнее сообщение сохраняются, старые реплики уходят.
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a " * 100},      # ~130 токенов
        {"role": "assistant", "content": "b " * 100},  # ~130 токенов
        {"role": "user", "content": "c " * 50},        # ~65 токенов
    ]
    out = _trim_messages_to_budget(msgs, 150)
    assert out is not None
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "SYS"
    assert out[-1]["content"] == "c " * 50


def test_trim_messages_to_budget_returns_none_on_overflow():
    # Даже последнее сообщение + системный блок не влезают → None (сигнал 413).
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "c " * 50},  # ~65 токенов > бюджета 50
    ]
    assert _trim_messages_to_budget(msgs, 50) is None
