"""Работа с diff: разбор, изменённые файлы, фильтрация и нарезка под бюджет.

Никаких внешних зависимостей — только stdlib. Используется и headless-скриптом
(review_pr.py), и эндпоинтом /review агента.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Файлы, которые нет смысла ревьюить (сгенерированные/бинарные/зависимости).
GENERATED_PATTERNS = (
    r"(^|/)(\.venv|venv|node_modules|__pycache__|\.gradle|build|dist|target)(/|$)",
    r"\.(lock|lock\.json|min\.js|min\.css|map|jar|png|jpe?g|gif|svg|ico|woff2?|ttf|eot|pdf|zip|tar|gz|db|sqlite3?|pyc|class)$",
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Pipfile\.lock|Cargo\.lock|gradle\.lockfile|\.DS_Store)$",
    r"(^|/)(uv\.lock|requirements.*\.txt)$",
    r"(^|/)\.gitignore$",
)

DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
NEW_FILE_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$")
DEL_FILE_RE = re.compile(r"^--- a/(.+?)\s*$")


@dataclass
class FileDiff:
    """Один изменённый файл из unified diff."""
    path: str
    status: str = "modified"          # added | deleted | modified | renamed
    patch: str = ""                   # сырой unified-diff этого файла
    additions: int = 0
    deletions: int = 0

    @property
    def is_binary(self) -> bool:
        return "Binary files" in self.patch


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: ~4 символа на токен для кода/английского."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def is_generated(path: str) -> bool:
    """True, если файл не стоит ревьюить (сгенерированный/бинарный/служебный)."""
    return any(re.search(p, path) for p in GENERATED_PATTERNS)


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Разбирает unified diff в список FileDiff (по одному на файл).

    Поддерживает `diff --git a/x b/x` и однократный `+++ b/x` без `diff --git`
    (на случай, если diff пришёл не от git).
    """
    files: list[FileDiff] = []
    current: FileDiff | None = None
    buf: list[str] = []

    def _flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.patch = "\n".join(buf)
            current.additions = sum(1 for ln in buf if ln.startswith("+") and not ln.startswith("+++"))
            current.deletions = sum(1 for ln in buf if ln.startswith("-") and not ln.startswith("---"))
            files.append(current)
        current = None
        buf = []

    for line in diff_text.splitlines():
        m = DIFF_HEADER_RE.match(line)
        if m:
            _flush()
            path = m.group(2) or m.group(1)
            current = FileDiff(path=path)
            buf.append(line)
            continue
        nm = NEW_FILE_RE.match(line)
        if nm and current is None:
            # Одиночный файл без `diff --git`-шапки.
            current = FileDiff(path=nm.group(1))
        if current is not None:
            buf.append(line)

    _flush()

    # Определяем статус.
    for f in files:
        patch = f.patch
        if re.search(r"^new file mode", patch, re.M):
            f.status = "added"
        elif re.search(r"^deleted file mode", patch, re.M):
            f.status = "deleted"
        elif re.search(r"^similarity index|^rename from ", patch, re.M):
            f.status = "renamed"
        else:
            f.status = "modified"
    return files


def changed_files(diff_text: str) -> list[str]:
    """Список путей изменённых файлов (без сгенерированных)."""
    return [f.path for f in parse_diff(diff_text) if not f.is_binary and not is_generated(f.path)]


def chunk_diff(diff_text: str, max_tokens: int = 8000) -> list[str]:
    """Разбивает diff на блоки так, чтобы каждый укладывался в max_tokens.

    Каждый блок — целый файл (hunks одного файла не режутся). Очень большие
    отдельные файлы усекаются по hunks с пометкой. Возвращает список текстовых
    блоков, готовых к вставке в промпт.
    """
    files = [f for f in parse_diff(diff_text) if not f.is_binary and not is_generated(f.path)]
    blocks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        if current:
            blocks.append("\n".join(current))
            current = []
            current_tokens = 0

    for f in files:
        header = f"### {f.path} ({f.status}, +{f.additions}/-{f.deletions})"
        piece = f"{header}\n{f.patch}"
        piece_tokens = estimate_tokens(piece)
        if piece_tokens > max_tokens:
            # Один файл больше бюджета: усекаем по строкам, сохраняя заголовок.
            limit_chars = max_tokens * 4
            truncated = piece[:limit_chars] + f"\n... [усечено: файл {f.path} слишком велик]"
            _flush()
            blocks.append(truncated)
            continue
        if current and current_tokens + piece_tokens > max_tokens:
            _flush()
        current.append(piece)
        current_tokens += piece_tokens
    _flush()
    return blocks
