"""Загрузка документов из data/raw/ → список Document.

Поддерживаемые форматы:
  - articles/*.txt   — статьи (русская Википедия, plain text)
  - docs/*.md        — README (markdown)
  - code/*           — исходный код (.py и др.)
  - pdfs/*.pdf       — PDF (извлекается через PyMuPDF)

Первичные метаданные документа: source (путь), title, file_type.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Document

TEXT_EXTS = {".txt", ".md", ".py", ".rst", ".json", ".yaml", ".yml", ".js", ".ts"}


def _file_type(rel_path: Path) -> str:
    parts = rel_path.parts
    if "pdfs" in parts:
        return "pdf"
    if "docs" in parts:
        return "readme"
    if "code" in parts:
        return "code"
    if "articles" in parts:
        return "article"
    return "other"


def _humanize_title(stem: str, file_type: str) -> str:
    title = stem.replace("_", " ").strip()
    if file_type == "readme" and title.lower().endswith(" readme"):
        title = title[: -len(" readme")]
    return title or stem


def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def _clean_text(text: str) -> str:
    # нормализуем переводы строк и схлопываем лишние пробелы внутри строк,
    # но сохраняем структуру абзацев/заголовков
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines)


def load_documents(data_dir: str | Path) -> list[Document]:
    """Рекурсивно обходит data_dir и возвращает документы."""
    data = Path(data_dir)
    docs: list[Document] = []

    for path in sorted(data.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".db", ".sqlite", ".faiss", ".json"}:
            continue
        rel = path.relative_to(data)
        # индексируем только стандартные подкаталоги, не служебные файлы
        if rel.parts[0] not in {"articles", "docs", "code", "pdfs"}:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        elif path.suffix.lower() in TEXT_EXTS:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
        else:
            continue

        text = _clean_text(text)
        if not text.strip():
            continue

        ft = _file_type(rel)
        doc_id = rel.as_posix()
        docs.append(
            Document(
                doc_id=doc_id,
                source=doc_id,
                title=_humanize_title(path.stem, ft),
                file_type=ft,
                text=text,
            )
        )

    return docs
