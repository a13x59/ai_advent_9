# test_support_endpoint.py
"""Тесты эндпоинта /support: создание/загрузка тикета, офтопик-гейт, привязка.

Используют мок-провайдер, фейковый support-mcp клиент и фейковый RAG-ретривер,
чтобы проверить логику пайплайна без сети и без запущенных сервисов.
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app, SUPPORT_ABSTAIN_REPLY
from storage import HistoryStorage
from mock_agent import MockProvider


class FakeSupportClient:
    """Эмуляция support-mcp: create_ticket / get_ticket в памяти."""

    SERVER_NAME = "support-mcp"

    def __init__(self):
        self.tickets = {}
        self.counter = 1000

    def call_tool(self, name, arguments):
        if name == "create_ticket":
            self.counter += 1
            number = f"T-{self.counter}"
            ticket = {
                "id": self.counter,
                "number": number,
                "subject": arguments.get("subject", ""),
                "status": "open",
                "first_message": arguments.get("message", ""),
                "customer": None,
            }
            self.tickets[number] = ticket
            return {"ok": True, "text": json.dumps(ticket, ensure_ascii=False)}
        if name == "get_ticket":
            ref = str(arguments.get("ticket_id", ""))
            ticket = self.tickets.get(ref)
            if ticket is None:
                return {"ok": False, "text": f"Тикет '{ref}' не найден."}
            return {"ok": True, "text": json.dumps(ticket, ensure_ascii=False)}
        return {"ok": False, "text": "unknown tool"}


class OnTopicRetriever:
    """RAG-ретривер: всегда релевантный чанк (по теме)."""

    def retrieve(self, query, strategy="structural", top_k=5, mode="baseline",
                 top_k_candidates=None, min_score=None, corpus="docs"):
        return {
            "chunks": [{
                "chunk_id": "c1",
                "source": "faq/avtorizaciya.md",
                "title": "avtorizaciya",
                "section": "Почему не работает авторизация?",
                "score": 0.9,
                "text": "Проверьте пароль и подтверждение аккаунта.",
            }],
            "below_relevance": False,
            "max_score": 0.9,
            "mode": mode,
        }


class OffTopicRetriever:
    """RAG-ретривер: всегда ниже порога (офтопик)."""

    def retrieve(self, query, **kwargs):
        return {"chunks": [], "below_relevance": True, "max_score": 0.1, "mode": "baseline"}


def _make_client(store_path, retriever):
    sc = FakeSupportClient()
    app = create_app(
        MockProvider(),
        HistoryStorage(store_path),
        mcp_clients=[sc],
        rag_retriever=retriever,
    )
    client = TestClient(app)
    client.support = sc
    return client


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(prefix="support_", suffix=".db")
    os.close(fd)
    c = _make_client(path, OnTopicRetriever())
    yield c
    c.close()
    os.remove(path)


@pytest.fixture()
def offtopic_client():
    fd, path = tempfile.mkstemp(prefix="support_off_", suffix=".db")
    os.close(fd)
    c = _make_client(path, OffTopicRetriever())
    yield c
    c.close()
    os.remove(path)


def _post(client, message, session_id="sess1"):
    return client.post("/support", json={"session_id": session_id, "message": message})


def test_create_ticket_on_question(client):
    r = _post(client, "/support Почему не работает авторизация?").json()

    assert r["ticket_created"] is True
    assert r["ticket_number"] == "T-1001"
    assert r["ticket"]["number"] == "T-1001"
    assert r["ticket"]["subject"] == "Почему не работает авторизация?"
    assert "Создан тикет №T-1001" in r["response"]
    assert r["abstained"] is False
    # RAG-источник попал в ответ.
    assert r["sources"][0]["source"] == "faq/avtorizaciya.md"


def test_plain_message_continues_bound_ticket(client):
    _post(client, "/support Почему не работает авторизация?")
    r = _post(client, "А если письмо с подтверждением не пришло?").json()

    assert r["ticket_created"] is False
    assert r["ticket_number"] == "T-1001"
    assert "Создан тикет" not in r["response"]


def test_explicit_ticket_id_loads_ticket(client):
    _post(client, "/support Вопрос про доставку", session_id="s2")
    r = _post(
        client,
        "/support --ticket_id='T-1001' А что с возвратом?",
        session_id="s3",
    ).json()

    assert r["ticket_created"] is False
    assert r["ticket_number"] == "T-1001"
    assert r["ticket"]["subject"] == "Вопрос про доставку"


def test_ticket_not_found_returns_404(client):
    resp = _post(client, "/support --ticket_id=T-9999 Вопрос")
    assert resp.status_code == 404
    assert "не найден" in resp.json()["detail"].lower()


def test_offtopic_abstains_and_does_not_create_ticket(offtopic_client):
    r = _post(offtopic_client, "/support Кто был 44 президентом США?").json()

    assert r["abstained"] is True
    assert r["response"] == SUPPORT_ABSTAIN_REPLY
    assert r["ticket"] is None
    assert r["ticket_number"] is None
    # Тикет не создан в «CRM».
    assert offtopic_client.support.tickets == {}


def test_empty_question_returns_400(client):
    resp = _post(client, "/support")
    assert resp.status_code == 400


def test_ticket_id_without_question_shows_summary(client):
    _post(client, "/support Вопрос про доставку", session_id="s2")
    r = _post(client, "/support --ticket_id=T-1001", session_id="s3").json()

    assert r["ticket_created"] is False
    assert r["ticket_number"] == "T-1001"
    assert r["ticket"]["subject"] == "Вопрос про доставку"
    assert "Тикет №T-1001" in r["response"]
    assert "Задайте вопрос" in r["response"]


def test_ticket_id_without_question_not_found_404(client):
    resp = _post(client, "/support --ticket_id=T-9999")
    assert resp.status_code == 404
