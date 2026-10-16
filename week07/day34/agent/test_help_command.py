# test_help_command.py
"""Тесты команды /help: распознавание, авто-RAG, git-контекст и поле help_context.

Используют мок-провайдер (без сети), фейковый RAG-ретривер и фейковый git-MCP-клиент,
чтобы проверить логику пайплайна, а не реальные сервисы.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app, _is_help_command, _collect_git_context, _is_project_doc
from mock_agent import MockProvider
from storage import HistoryStorage


class RecordingProvider(MockProvider):
    """Мок, который запоминает сообщения, переданные модели."""

    def __init__(self):
        self.last_messages = []

    def complete(self, messages, request, task, user_text, tools=None):
        self.last_messages = messages
        return self._response("Эхо (mock).")


class FakeRetriever:
    """Детерминированный ретривер: возвращает чанк документации проекта."""

    def retrieve(self, query, strategy="structural", top_k=5, mode="baseline",
                 top_k_candidates=None, min_score=None):
        return [{
            "chunk_id": "c1",
            "source": "docs/agent_architecture.md",
            "title": "Архитектура агента",
            "section": "Общая схема",
            "score": 0.9,
            "text": "Агент — FastAPI-приложение, собранное фабрикой create_app.",
        }]


class MixedRetriever:
    """Ретривер: общий корпус (статьи) впереди, документация проекта — ниже по score."""

    def retrieve(self, query, strategy="structural", top_k=5, mode="baseline",
                 top_k_candidates=None, min_score=None):
        return [
            {"chunk_id": "a1", "source": "articles/информационный_поиск.txt",
             "title": "информационный поиск", "section": "", "score": 0.6,
             "text": "Информационный поиск — процесс поиска информации."},
            {"chunk_id": "p1", "source": "docs/agent_data_model.md",
             "title": "Схемы данных", "section": "conversations", "score": 0.5,
             "text": "conversations — сессии (session_id, strategy, window_size)."},
            {"chunk_id": "a2", "source": "articles/n_грамма.txt",
             "title": "n-грамма", "section": "", "score": 0.4,
             "text": "N-грамма — последовательность из n элементов."},
        ]


class DocRetriever(MixedRetriever):
    """Ретривер с get_document: отдаёт полный текст документа проекта."""

    def get_document(self, source, strategy="structural"):
        return {
            "source": source,
            "title": "Схемы данных",
            "text": "conversations — сессии.\n\nworking_memory — рабочая память.\n\n"
                    "long_term_memory — постоянная память.\n\ninvariants — инварианты.",
            "n_chunks": 4,
        }


class FakeGitClient:
    """Фейковый git-MCP-клиент: возвращает детерминированный контекст."""

    SERVER_NAME = "git-mcp"

    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [{"name": "git_branch"}, {"name": "list_files"}, {"name": "git_diff"}]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "git_branch":
            return {"ok": True, "text": '{"current_branch": "main", "branches": ["main"]}'}
        if name == "list_files":
            return {"ok": True, "text": '{"count": 2, "truncated": false, "files": ["a.py", "b.py"]}'}
        return {"ok": True, "text": "diff --git a/x b/x"}


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(prefix="help_", suffix=".db")
    os.close(fd)
    provider = RecordingProvider()
    app = create_app(
        provider,
        HistoryStorage(path),
        mcp_clients=[FakeGitClient()],
        rag_retriever=FakeRetriever(),
    )
    with TestClient(app) as c:
        c.provider = provider
        yield c
    os.remove(path)


def _post(client, text):
    return client.post("/agent", json={
        "session_id": "sess1",
        "model": "mock-model",
        "messages": [{"role": "user", "content": text}],
    }).json()


def test_is_help_command():
    assert _is_help_command("/help") == ""
    assert _is_help_command("/help расскажи о структуре") == "расскажи о структуре"
    assert _is_help_command("  /HELP что это?") == "что это?"
    assert _is_help_command("/helper") is None
    assert _is_help_command("привет") is None


def test_help_returns_context_with_rag_and_git(client):
    r = _post(client, "/help расскажи о структуре проекта")

    assert r["help_context"] is not None
    assert r["help_context"]["question"] == "расскажи о структуре проекта"
    assert r["help_context"]["sources"][0]["source"] == "docs/agent_architecture.md"
    assert "git_branch" in r["help_context"]["git"]
    assert "list_files" in r["help_context"]["git"]

    system_text = "\n".join(
        m["content"] for m in client.provider.last_messages if m["role"] == "system"
    )
    assert "ассистент разработчика" in system_text
    assert "ДОКУМЕНТАЦИЯ ПРОЕКТА" in system_text
    assert "КОНТЕКСТ ПРОЕКТА (git)" in system_text
    assert "create_app" in system_text
    # Свободный формат, а не строгий RAG-формат с цитатами.
    assert "## Цитаты" not in system_text


def test_help_default_question(client):
    r = _post(client, "/help")
    assert r["help_context"]["question"] == "структура проекта"


def test_non_help_message_has_no_help_context(client):
    r = _post(client, "Привет")
    assert r["help_context"] is None
    system_text = "\n".join(
        m["content"] for m in client.provider.last_messages if m["role"] == "system"
    )
    assert "ассистент разработчика" not in system_text


def test_help_without_git_client():
    fd, path = tempfile.mkstemp(prefix="help_nogit_", suffix=".db")
    os.close(fd)
    provider = RecordingProvider()
    app = create_app(provider, HistoryStorage(path), rag_retriever=FakeRetriever())
    with TestClient(app) as c:
        r = c.post("/agent", json={
            "session_id": "s1",
            "model": "mock-model",
            "messages": [{"role": "user", "content": "/help"}],
        }).json()
        assert r["help_context"]["git"] == {}
        assert r["help_context"]["sources"][0]["source"] == "docs/agent_architecture.md"
    os.remove(path)


def test_collect_git_context_marks_unavailable_on_error():
    class BrokenGitClient:
        SERVER_NAME = "git-mcp"

        def call_tool(self, name, arguments):
            raise RuntimeError("connection refused")

    ctx = _collect_git_context([BrokenGitClient()])
    assert ctx["git_branch"].startswith("недоступен")
    assert ctx["list_files"].startswith("недоступен")


def test_is_project_doc():
    assert _is_project_doc({"source": "docs/agent_data_model.md"}) is True
    assert _is_project_doc({"source": "articles/x.txt"}) is False
    assert _is_project_doc({"source": "docs/rulm_README.md"}) is False
    assert _is_project_doc({}) is False


def test_help_filters_non_project_chunks():
    fd, path = tempfile.mkstemp(prefix="help_mixed_", suffix=".db")
    os.close(fd)
    provider = RecordingProvider()
    app = create_app(provider, HistoryStorage(path), rag_retriever=MixedRetriever())
    with TestClient(app) as c:
        r = c.post("/agent", json={
            "session_id": "s1",
            "model": "mock-model",
            "messages": [{"role": "user", "content": "/help выведи схему данных"}],
        }).json()
        sources = [s["source"] for s in (r["help_context"]["sources"] or [])]
        assert all(src.startswith("docs/agent_") for src in sources)
        assert "docs/agent_data_model.md" in sources
        assert "articles/информационный_поиск.txt" not in sources
    os.remove(path)


def test_help_expands_to_full_document():
    fd, path = tempfile.mkstemp(prefix="help_doc_", suffix=".db")
    os.close(fd)
    provider = RecordingProvider()
    app = create_app(provider, HistoryStorage(path), rag_retriever=DocRetriever())
    with TestClient(app) as c:
        c.provider = provider
        c.post("/agent", json={
            "session_id": "s1",
            "model": "mock-model",
            "messages": [{"role": "user", "content": "/help выведи схему данных"}],
        }).json()
        system_text = "\n".join(
            m["content"] for m in provider.last_messages if m["role"] == "system"
        )
        assert "ДОКУМЕНТАЦИЯ ПРОЕКТА (полные документы)" in system_text
        assert "long_term_memory — постоянная память" in system_text
        assert "invariants — инварианты" in system_text
    os.remove(path)
