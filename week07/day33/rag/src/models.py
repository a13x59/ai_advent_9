"""Общие структуры данных пайплайна."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Document:
    """Документ после загрузки (текст + первичные метаданные)."""

    doc_id: str        # уникальный id (относительный путь к файлу)
    source: str        # источник (путь к файлу относительно data/)
    title: str         # человекочитаемое название
    file_type: str     # article | readme | code | pdf
    text: str          # полный текст

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """Чанк с метаданными — единица индексации."""

    chunk_id: str      # уникальный id чанка
    source: str        # источник документа
    title: str         # название документа
    section: str       # заголовок раздела (для structural) или ""
    strategy: str      # fixed | structural
    char_start: int    # смещение начала в исходном тексте
    char_end: int      # смещение конца в исходном тексте
    text: str          # текст чанка

    def to_dict(self) -> dict:
        return asdict(self)
