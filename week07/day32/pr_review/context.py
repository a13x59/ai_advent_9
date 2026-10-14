"""Детерминированный контекст для ревью (без тяжёлых зависимостей).

Три уровня контекста, которые работают всегда (в т.ч. в CI без torch/faiss):
  1) полный текущий текст изменённых файлов;
  2) локально импортируемые модули (связанный код);
  3) документация, релевантная изменённым областям.

Поверх этого engine может добавить 4-й уровень — векторный RAG по корпусам
docs/codebase (см. rag_client.py), когда поднят RAG-сервис.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../week07/day32

MAX_FILE_CHARS = 12_000
MAX_TOTAL_CHARS = 40_000

# Какой документации сопоставлять изменённую папку.
DOC_FOR_PREFIX = [
    ("agent/", ["agent/docs/architecture.md", "agent/docs/data-model.md",
                "agent/docs/api.md", "agent/docs/mcp-and-rag.md"]),
    ("rag/", ["rag/README.md"]),
    ("git-mcp/", ["git-mcp/README.md"]),
    ("pipeline-mcp/", ["pipeline-mcp/README.md"]),
    ("currency-mcp/", ["currency-mcp/README.md"]),
    ("fortune-telling-mcp/", ["fortune-telling-mcp/README.md"]),
    ("deploy/", ["deploy/README.md"]),
    ("llm-optimization/", ["llm-optimization/README.md"]),
]

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))",
    re.M,
)


@dataclass
class ContextBlock:
    kind: str            # file | related | docs
    title: str
    content: str


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [усечено]"


def read_current_file(rel_path: str) -> str | None:
    """Текущая полная версия файла из рабочего дерева (None, если нет/бинарный)."""
    p = ROOT / rel_path
    try:
        if not p.is_file():
            return None
        data = p.read_bytes()
        if b"\x00" in data[:4096]:
            return None  # бинарный
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


def _local_module_to_path(module: str, base_dir: str) -> str | None:
    """Пробует сопоставить импортируемый модуль локальному файлу проекта.

    Возвращает относительный путь (или None, если модуль внешний).
    """
    rel = module.replace(".", "/")
    candidates = [f"{base_dir}/{rel}.py"]
    if base_dir:
        candidates.append(f"{rel}.py")
    for cand in candidates:
        if (ROOT / cand).is_file():
            return cand
    # пакет: module/__init__.py
    candidates = [f"{base_dir}/{rel}/__init__.py"]
    if base_dir:
        candidates.append(f"{rel}/__init__.py")
    for cand in candidates:
        if (ROOT / cand).is_file():
            return cand
    return None


def build_full_file_context(changed_files: list[str]) -> list[ContextBlock]:
    """Уровень 1: полный текущий текст каждого изменённого файла."""
    blocks: list[ContextBlock] = []
    for path in changed_files:
        content = read_current_file(path)
        if content is None:
            blocks.append(ContextBlock("file", path, "(файл удалён или недоступен)"))
            continue
        blocks.append(ContextBlock("file", path, _truncate(content, MAX_FILE_CHARS)))
    return blocks


def build_related_context(changed_files: list[str]) -> list[ContextBlock]:
    """Уровень 2: локально импортируемые модули (Python), до N файлов."""
    seen: set[str] = set()
    blocks: list[ContextBlock] = []
    for path in changed_files:
        if not path.endswith(".py"):
            continue
        content = read_current_file(path)
        if not content:
            continue
        base_dir = str(Path(path).parent) if "/" in path else ""
        for m in _IMPORT_RE.finditer(content):
            module = m.group(1) or m.group(2)
            if not module:
                continue
            # относительные импорты (.foo, ..bar) — разрешаем от пакета файла.
            if module.startswith("."):
                pkg = Path(path).with_suffix("").as_posix().replace("/", ".")
                module = pkg + module
            target = _local_module_to_path(module, base_dir)
            if not target or target == path:
                continue
            if target in seen or target in set(changed_files):
                continue
            txt = read_current_file(target)
            if not txt:
                continue
            seen.add(target)
            blocks.append(ContextBlock("related", target, _truncate(txt, MAX_FILE_CHARS)))
            if len(blocks) >= 6:
                return blocks
    return blocks


def build_doc_context(changed_files: list[str]) -> list[ContextBlock]:
    """Уровень 3: документация, релевантная изменённым папкам."""
    chosen: list[str] = []
    for path in changed_files:
        for prefix, docs in DOC_FOR_PREFIX:
            if path.startswith(prefix):
                for d in docs:
                    if d not in chosen:
                        chosen.append(d)
    blocks: list[ContextBlock] = []
    for doc in chosen:
        content = read_current_file(doc)
        if content:
            blocks.append(ContextBlock("docs", doc, _truncate(content, MAX_FILE_CHARS)))
    return blocks


def build_deterministic_context(changed_files: list[str]) -> list[ContextBlock]:
    """Собирает уровни 1–3 с общим бюджетом."""
    blocks: list[ContextBlock] = []
    total = 0
    for b in build_full_file_context(changed_files) + \
             build_related_context(changed_files) + \
             build_doc_context(changed_files):
        if total + len(b.content) > MAX_TOTAL_CHARS and total > 0:
            break
        blocks.append(b)
        total += len(b.content)
    return blocks


def render_context(blocks: list[ContextBlock]) -> str:
    """Текстовое представление контекста для вставки в промпт."""
    if not blocks:
        return "(контекст недоступен)"
    out: list[str] = []
    for b in blocks:
        kind_label = {"file": "Файл", "related": "Связанный код", "docs": "Документация"}[b.kind]
        out.append(f"===== {kind_label}: {b.title} =====\n{b.content}")
    return "\n\n".join(out)
