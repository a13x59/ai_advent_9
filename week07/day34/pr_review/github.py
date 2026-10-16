"""Публикация ревью комментарием в PR (GitHub API).

Использует GITHUB_TOKEN (автоматически предоставляется в GitHub Actions).
Вне Actions можно передать персональный токен через переменную GITHUB_TOKEN.
"""
from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger("pr_review.github")

GITHUB_API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def post_pr_comment(body: str, *, pr_number: int, repo: str | None = None,
                    token: str | None = None) -> dict:
    """Публикует комментарий к issue/PR. Возвращает JSON ответа API."""
    token = token or os.environ.get("GITHUB_TOKEN", "")
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN не задан — не могу опубликовать комментарий")
    if not repo:
        raise RuntimeError("GITHUB_REPOSITORY не задан")
    if not pr_number:
        raise RuntimeError("номер PR не задан")

    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(
        url,
        json={"body": body},
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub API {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    logger.info("Комментарий опубликован: %s", data.get("html_url"))
    return data
