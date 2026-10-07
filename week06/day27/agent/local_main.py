# local_main.py
"""Точка входа: агент на ЛОКАЛЬНОЙ модели (Ollama llama3.1), без облака.

Тот же пайплайн create_app(...), что и main.py, но провайдер —
LocalOllamaProvider, который ходит в локальный сервер Ollama
(OLLAMA_BASE_URL, по умолчанию http://127.0.0.1:11434). Ключ DeepSeek не нужен.

Запуск: uvicorn local_main:app --host 0.0.0.0 --port 8002
"""
from local_provider import LocalOllamaProvider
from agent_core import create_app
from mcp_client import FortuneMcpClient, CurrencyMcpClient, PipelineMcpClient
from rag_client import RagClient


app = create_app(
    LocalOllamaProvider(),
    mcp_clients=[FortuneMcpClient(), CurrencyMcpClient(), PipelineMcpClient()],
    rag_retriever=RagClient(),
)
