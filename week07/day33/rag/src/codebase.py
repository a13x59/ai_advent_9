"""Корпус «codebase»: синхронизация симлинков на исходники проекта (День 32).

Единственный источник правды о том, ЧТО индексировать, — список SOURCE_GLOBS
здесь. Симлинки под rag/data/codebase/code/ — производная: их создаёт и чистит
функция sync_codebase(), которую вызывают:
  - CLI:        python rag/scripts/sync_codebase.py
  - пайплайн:   python -m src.pipeline run --corpus codebase (синхронизирует сам)

Поэтому при появлении/удалении файлов в проекте ничего делать вручную не нужно:
sync_codebase() добавляет новые ссылки и удаляет устаревшие (в т.ч. битые).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../week07/day32
DEST_ROOT = ROOT / "rag" / "data" / "codebase" / "code"

# Что индексируем в корпусе кода (относительно ROOT).
SOURCE_GLOBS = [
    "agent/*.py",
    "rag/src/*.py",
    "llm-optimization/*.py",
    "scripts/*.py",
    "deploy/*.sh",
    "pr_review/*.py",
    "git-mcp/src/**/*.kt",
    "pipeline-mcp/src/**/*.kt",
    "currency-mcp/src/**/*.kt",
    "fortune-telling-mcp/src/**/*.kt",
]

# Каталоги, которые никогда не индексируем.
EXCLUDE_PARTS = {".venv", "build", "__pycache__", "node_modules"}


def collect_sources(root: Path | None = None) -> list[Path]:
    """Файлы исходников, которые попадают в корпус (абсолютные пути)."""
    root = Path(root) if root else ROOT
    sources: list[Path] = []
    for pattern in SOURCE_GLOBS:
        for raw in glob.glob(str(root / pattern), recursive=True):
            p = Path(raw)
            if not p.is_file():
                continue
            if EXCLUDE_PARTS & set(p.parts):
                continue
            sources.append(p.resolve())
    return sorted(set(sources))


def sync_codebase(root: Path | None = None, dry_run: bool = False,
                  remove_stale: bool = True) -> dict:
    """Создаёт/обновляет симлинки корпуса и удаляет устаревшие.

    Идемпотентна. Возвращает статистику {total, created, updated, removed}.
    """
    root = Path(root) if root else ROOT
    dest = root / "rag" / "data" / "codebase" / "code"

    sources = collect_sources(root)
    planned: dict[Path, Path] = {}
    for src in sources:
        rel = src.relative_to(root)  # напр. agent/agent_core.py
        planned[dest / rel] = src

    removed = 0
    if remove_stale and dest.exists():
        for link in sorted(dest.rglob("*")):
            if link.is_symlink() and link not in planned:
                if not dry_run:
                    link.unlink()
                removed += 1
                print(f"remove stale: {link.relative_to(dest)}")

    created = updated = 0
    for link, src in sorted(planned.items()):
        rel_target = os.path.relpath(src, start=link.parent)
        if link.is_symlink():
            if os.readlink(link) == rel_target:
                continue
            if not dry_run:
                link.unlink()
            updated += 1
            print(f"relink: {link.relative_to(dest)} -> {rel_target}")
        elif link.exists():
            # Реальный файл вместо симлинка — не трогаем, только предупреждаем.
            print(f"WARN: файл существует и не симлинк, пропускаю: {link}")
            continue
        else:
            if not dry_run:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(rel_target)
            created += 1
            print(f"link: {link.relative_to(dest)} -> {rel_target}")

    stats = {"total": len(sources), "created": created,
             "updated": updated, "removed": removed}
    print(f"Итого источников: {stats['total']}, создано: {created}, "
          f"обновлено: {updated}, удалено устаревших: {removed}")
    return stats
