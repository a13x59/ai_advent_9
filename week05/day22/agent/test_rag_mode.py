# test_rag_mode.py
"""Тесты режима RAG: инъекция чанков в контекст и поле rag_context.

Используют мок-провайдер (без сети) и фейковый ретривер, чтобы проверить
логику пайплайна, а не реальный rag-сервис.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app
from storage import HistoryStorage
from mock_agent import MockProvider


class RecordingProvider(MockProvider):
    """Мок, который запоминает сообщения, переданные модели."""

    def __init__(self):
        self.last_messages = []

    def complete(self, messages, request, task, user_text, tools=None):
        self.last_messages = messages
        return self._response("Эхо (mock).")


class FakeRetriever:
    """Детерминированный ретривер вместо реального rag-сервиса."""

    def retrieve(self, query, strategy="structural", top_k=5):
        return [{
            "chunk_id": "c1",
            "source": "articles/word2vec.txt",
            "title": "word2vec",
            "section": "",
            "score": 0.9,
            "text": "word2vec — модель векторного представления слов.",
        }]


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(prefix="rag_", suffix=".db")
    os.close(fd)
    provider = RecordingProvider()
    app = create_app(provider, HistoryStorage(path), rag_retriever=FakeRetriever())
    with TestClient(app) as c:
        c.provider = provider
        yield c
    os.remove(path)


def _post(client, text, rag):
    return client.post("/agent", json={
        "session_id": "sess1",
        "model": "mock-model",
        "messages": [{"role": "user", "content": text}],
        "rag": rag,
    }).json()


def test_rag_mode_injects_context_and_reports_sources(client):
    r = _post(client, "Что такое word2vec?", rag=True)

    assert r["rag_context"]["sources"] == ["articles/word2vec.txt"]
    assert r["rag_context"]["chunks"][0]["title"] == "word2vec"

    system_text = "\n".join(
        m["content"] for m in client.provider.last_messages if m["role"] == "system"
    )
    assert "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (RAG)" in system_text
    assert "word2vec — модель векторного представления слов." in system_text


def test_rag_off_does_not_retrieve(client):
    r = _post(client, "Привет", rag=False)
    assert r["rag_context"] is None
    system_text = "\n".join(
        m["content"] for m in client.provider.last_messages if m["role"] == "system"
    )
    assert "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (RAG)" not in system_text
