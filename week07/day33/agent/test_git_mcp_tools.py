# test_git_mcp_tools.py
"""Тесты git-MCP-клиента агента (без сети)."""
from agent_core import _find_git_client
from mcp_client import GitMcpClient


def test_git_client_defaults():
    c = GitMcpClient()
    assert c.SERVER_NAME == "git-mcp"
    assert c.base_url == "http://localhost:8892"


def test_find_git_client():
    class Other:
        SERVER_NAME = "currency-mcp"

    git = GitMcpClient()
    assert _find_git_client([Other(), git]) is git
    assert _find_git_client([Other()]) is None
    assert _find_git_client([]) is None
