"""pr_review — автоматическое AI-ревью кода (День 32).

Пакет реализует два взаимозаменяемых подхода:
  - headless-скрипт review_pr.py (engine.py) — lean, без FastAPI;
  - эндпоинт /review агента (agent/) — полный агентский пайплайн.

Переключатели: REVIEW_BACKEND (script|agent) и REVIEW_PROVIDER (deepseek|ollama).
"""
from .engine import ReviewResult, run_review
from .prompt import split_sections, render_pr_comment

__all__ = ["ReviewResult", "run_review", "split_sections", "render_pr_comment"]
