# test_rag_sources_history.py
"""Тесты: источники RAG пишутся в историю + новые kind памяти (clarification/term).

Проверяют:
  • ответ /agent содержит структурированный список sources;
  • sources сохраняются в сообщении ассистента и отдаются в GET /agent/{id};
  • при выключенном RAG sources не добавляются;
  • kind clarification и term принимаются рабочей памятью без коэрции.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app
from mock_agent import MockProvider
from storage import HistoryStorage


class ChunkRetriever:
    """Детерминированный ретривер в формате реального rag-сервиса (dict с chunks)."""

    def retrieve(self, query, strategy="structural", top_k=5, mode="baseline",
                 top_k_candidates=None, min_score=None):
        return {
            "chunks": [{
                "chunk_id": "c1",
                "source": "articles/word2vec.txt",
                "title": "word2vec",
                "section": "Определение",
                "score": 0.9,
                "text": "word2vec — модель векторного представления слов.",
            }],
            "mode": "baseline",
        }


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(prefix="rag_hist_", suffix=".db")
    os.close(fd)
    app = create_app(MockProvider(), HistoryStorage(path), rag_retriever=ChunkRetriever())
    with TestClient(app) as c:
        yield c
    os.remove(path)


def _post(client, session_id, text, rag):
    return client.post("/agent", json={
        "session_id": session_id,
        "model": "mock-model",
        "messages": [{"role": "user", "content": text}],
        "rag": rag,
    }).json()


def test_sources_attached_to_response_and_history(client):
    r = _post(client, "sess1", "Что такое word2vec?", rag=True)

    assert r["sources"] == [{
        "n": 1,
        "source": "articles/word2vec.txt",
        "title": "word2vec",
        "section": "Определение",
        "chunk_id": "c1",
        "score": 0.9,
    }]

    hist = client.get("/agent/sess1").json()
    assistant = [m for m in hist["messages"] if m["role"] == "assistant"]
    assert assistant
    assert assistant[-1]["sources"][0]["source"] == "articles/word2vec.txt"
    assert assistant[-1]["sources"][0]["chunk_id"] == "c1"


def test_no_sources_when_rag_off(client):
    r = _post(client, "sess2", "Привет", rag=False)

    assert r["sources"] == []
    hist = client.get("/agent/sess2").json()
    assistant = [m for m in hist["messages"] if m["role"] == "assistant"]
    assert assistant
    assert "sources" not in assistant[-1]


def test_clarification_and_term_kinds_persist(tmp_path):
    store = HistoryStorage(str(tmp_path / "kinds.db"))
    store.create_session("s1")

    c = store.save_working_entry("s1", {
        "key": "что_уточнил", "value": "объяснять без кода", "kind": "clarification", "state": "done",
    })
    assert c["kind"] == "clarification"

    t = store.save_working_entry("s1", {
        "key": "эмбеддинг", "value": "вектор слова", "kind": "term",
    })
    assert t["kind"] == "term"

    kinds = {e["kind"] for e in store.load_working("s1")}
    assert {"clarification", "term"} <= kinds


def test_delete_working_memory_via_op_does_not_deadlock(tmp_path):
    """Регрессия: удаление рабочей памяти раньше зависало (вложенный захват Lock)."""
    store = HistoryStorage(str(tmp_path / "del.db"))
    store.create_session("s1")
    store.save_working_entry("s1", {
        "key": "цель", "value": "сравнить word2vec и bag-of-words",
        "kind": "goal", "state": "done",
    })

    res = store.apply_memory_op("s1", {"action": "delete", "layer": "working", "key": "цель"})

    assert res["deleted"] is True
    assert store.load_working("s1") == []


def test_move_working_to_long_term_does_not_deadlock(tmp_path):
    """Регрессия: перенос рабочая → долговременная тоже проходил через тот же лок."""
    store = HistoryStorage(str(tmp_path / "move.db"))
    store.create_session("s1")
    store.save_working_entry("s1", {"key": "факт", "value": "x", "kind": "note"})

    res = store.apply_memory_op("s1", {"action": "move", "from": "working", "to": "long_term", "key": "факт"})

    assert res["moved_from"] == "working"
    assert store.load_working("s1") == []
    assert store.get_long_term_entry("факт") is not None
