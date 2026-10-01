# rag_client.py
"""Клиент RAG-retrieval сервиса (HTTP) для агента.

Агент не тянет в свой venv тяжёлые зависимости (faiss / sentence-transformers):
ретрив чанков делает отдельный сервис в rag/ (src/server.py), а этот клиент
просто ходит к нему по HTTP — по аналогии с mcp_client.py.
"""
import os

import requests

RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://localhost:8891")
RAG_TIMEOUT = float(os.environ.get("RAG_TIMEOUT", "30"))


class RagClient:
    """Возвращает релевантные чанки по вопросу (интерфейс retrieve)."""

    SERVER_NAME = "rag-retrieval"

    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = (base_url or RAG_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else RAG_TIMEOUT

    def retrieve(self, query: str, strategy: str = "structural", top_k: int = 5) -> list:
        try:
            resp = requests.post(
                self.base_url + "/retrieve",
                json={"query": query, "strategy": strategy, "top_k": top_k},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"RAG-retrieval недоступен: {e}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"RAG-retrieval HTTP {resp.status_code}: {resp.text}")
        return (resp.json() or {}).get("chunks", [])
