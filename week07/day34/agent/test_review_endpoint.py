"""Тест эндпоинта /review агента (подход A) — без сети, на MockProvider."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # day32

from fastapi.testclient import TestClient  # noqa: E402

from agent_core import create_app  # noqa: E402
from mock_agent import MockProvider  # noqa: E402

DIFF = """\
diff --git a/agent/rag_client.py b/agent/rag_client.py
--- a/agent/rag_client.py
+++ b/agent/rag_client.py
@@ -28,6 +28,7 @@ class RagClient:
-        payload = {"query": query}
+        payload = {"query": query, "corpus": corpus}
"""


def test_review_endpoint_returns_review():
    app = create_app(MockProvider(), rag_retriever=None)
    client = TestClient(app)
    resp = client.post("/review", json={"diff": DIFF, "model": "deepseek-chat",
                                        "use_rag": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "review_text" in data
    assert "changed_files" in data
    assert data["changed_files"] == ["agent/rag_client.py"]
    # MockProvider эхо-отвечает, поэтому в тексте есть исходный diff.
    assert "corpus" in data["review_text"] or "Эхо" in data["review_text"]


def test_review_endpoint_rejects_empty_diff():
    app = create_app(MockProvider())
    client = TestClient(app)
    resp = client.post("/review", json={"diff": "", "model": "deepseek-chat"})
    assert resp.status_code == 400
