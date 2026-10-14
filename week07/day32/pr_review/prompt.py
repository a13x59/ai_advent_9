"""Промпт и контракт вывода AI-ревью.

Единый для обоих подходов (headless-скрипт и эндпоинт /review агента), чтобы
результаты были сопоставимы.

Контракт вывода — три обязательные секции с фиксированными заголовками:
    ## Потенциальные баги
    ## Архитектурные проблемы
    ## Рекомендации
"""
from __future__ import annotations

import re

SECTION_BUGS = "Потенциальные баги"
SECTION_ARCH = "Архитектурные проблемы"
SECTION_RECO = "Рекомендации"

REVIEW_SYSTEM = (
    "Ты — старший инженер, выполняющий ревью pull request. "
    "Твой ответ — строгое, конкретное и полезное ревью на русском языке.\n\n"
    "Правила:\n"
    "- Анализируй ТОЛЬКО предоставленный diff и контекст; не выдумывай код, "
    "которого нет в материалах.\n"
    "- Для каждого замечания указывай файл и, по возможности, строку/функцию.\n"
    "- Разделяй критичность: критично / важно / незначительно.\n"
    "- Не хвали ради похвалы; если замечаний нет в какой-то секции — честно "
    "напиши «не обнаружено».\n"
    "- Ссылайся на контекст (документацию/связанный код), когда он подтверждает "
    "или опровергает замечание.\n\n"
    "Формат ответа — СТРОГО три секции с ровно такими заголовками:\n"
    "## Потенциальные баги\n"
    "<список или «не обнаружено»>\n\n"
    "## Архитектурные проблемы\n"
    "<список или «не обнаружено»>\n\n"
    "## Рекомендации\n"
    "<конкретные предложения по улучшению>"
)


def build_review_messages(diff_text: str, changed_files: list[str],
                          context_text: str = "", rag_text: str = "") -> list[dict]:
    """Сообщения для LLM: system + user (diff + изменённые файлы + контекст)."""
    parts = [
        f"## Изменённые файлы\n{', '.join(changed_files) if changed_files else '(нет)'}",
        f"## Diff\n```diff\n{diff_text}\n```",
    ]
    if context_text and context_text != "(контекст недоступен)":
        parts.append(f"## Контекст (полные файлы, связанный код, документация)\n{context_text}")
    if rag_text:
        parts.append(f"## RAG-контекст (семантический поиск)\n{rag_text}")

    user = "\n\n".join(parts) + "\n\nСделай ревью этого изменения по трём секциям."
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": user},
    ]


def split_sections(text: str) -> dict[str, str]:
    """Разбивает ответ модели на три секции по фиксированным заголовкам."""
    result = {SECTION_BUGS: "", SECTION_ARCH: "", SECTION_RECO: ""}
    order = [SECTION_BUGS, SECTION_ARCH, SECTION_RECO]
    pattern = "|".join(re.escape(s) for s in order)
    parts = re.split(rf"^##\s*(?:{pattern})\s*$", text or "", flags=re.M | re.I)
    # parts[0] — всё до первого заголовка (обычно пусто); дальше по порядку.
    bodies = [p.strip() for p in parts[1:]]
    for i, section in enumerate(order):
        if i < len(bodies):
            result[section] = bodies[i]
    return result


def render_review(review_text: str, *, title: str = "AI-ревью кода") -> str:
    """Форматирует ревью для комментария в PR (markdown)."""
    body = (review_text or "").strip() or "*(пустой ответ модели)*"
    return f"## 🤖 {title}\n\n{body}"


def render_pr_comment(review_text: str, changed_files: list[str] | None = None) -> str:
    """Итоговый markdown комментария в PR."""
    header = f"## 🤖 AI-ревью кода\n\n"
    if changed_files:
        files = ", ".join(f"`{f}`" for f in changed_files[:20])
        if len(changed_files) > 20:
            files += f" (и ещё {len(changed_files) - 20})"
        header += f"*Файлы:* {files}\n\n"
    return header + (review_text or "").strip()
