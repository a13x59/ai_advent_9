"""Две стратегии разбиения на чанки + метаданные.

1. fixed_size_chunks  — по фиксированному размеру (окно в N слов
   с перекрытием overlap слов). Даёт равномерные по длине чанки.

2. structural_chunks  — по структуре:
     * markdown-заголовки  (#, ##, ...)
     * вики-заголовки      (== Раздел ==, === Подраздел ===)
     * для кода — границы функций/классов (def/class)
   Длинные секции дополнительно режутся по абзацам (без разрыва абзаца).

У каждого чанка заполняются метаданные:
   chunk_id, source, title, section, strategy, char_start, char_end, text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Chunk, Document

WORD_RE = re.compile(r"\S+")
MD_HEADING_RE = re.compile(r"^(\#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
WIKI_HEADING_RE = re.compile(r"^(={2,6})[ \t]*([^=\n]+?)[ \t]*=+[ \t]*$", re.M)
CODE_HEADING_RE = re.compile(
    r"^[ \t]*(?:(?:async[ \t]+)?def|class)[ \t]+(?P<name>[A-Za-z_]\w*)", re.M
)
FENCE_RE = re.compile(r"^```", re.M)


@dataclass
class _Span:
    start: int
    end: int


def _word_spans(text: str) -> list[_Span]:
    return [_Span(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _make_chunk(doc: Document, strategy: str, index: int, text: str,
                start: int, end: int, section: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:{strategy}:{index:04d}",
        source=doc.source,
        title=doc.title,
        section=section,
        strategy=strategy,
        char_start=start,
        char_end=end,
        text=text.strip(),
    )


# --------------------------------------------------------------------------- #
# 1) Фиксированный размер
# --------------------------------------------------------------------------- #
def fixed_size_chunks(doc: Document, max_words: int = 100,
                      overlap_words: int = 20) -> list[Chunk]:
    """Чанки фиксированной длины (в словах) с перекрытием."""
    spans = _word_spans(doc.text)
    if not spans:
        return []

    step = max(1, max_words - overlap_words)
    chunks: list[Chunk] = []
    start_idx = 0
    index = 0

    while start_idx < len(spans):
        end_idx = min(start_idx + max_words, len(spans))
        cs = spans[start_idx].start
        ce = spans[end_idx - 1].end
        text = doc.text[cs:ce]
        if text.strip():
            chunks.append(_make_chunk(doc, "fixed", index, text, cs, ce, ""))
            index += 1
        if end_idx == len(spans):
            break
        start_idx += step

    return chunks


# --------------------------------------------------------------------------- #
# 2) По структуре
# --------------------------------------------------------------------------- #
def _mask_code_fences(text: str) -> str:
    """Заменяет содержимое ```-блоков пробелами (для корректного поиска
    заголовков вне кода), сохраняя длину строк — смещения не меняются."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for ln in lines:
        if FENCE_RE.match(ln):
            in_fence = not in_fence
            out.append(" " * len(ln))
        elif in_fence:
            out.append(" " * len(ln))
        else:
            out.append(ln)
    return "\n".join(out)


def _heading_positions(text: str, file_type: str) -> list[tuple[int, int, str]]:
    """Возвращает (start, end, title) для каждого заголовка в порядке
    появления в тексте."""
    positions: list[tuple[int, int, str]] = []
    for m in MD_HEADING_RE.finditer(text):
        positions.append((m.start(), m.end(), m.group(2).strip()))
    for m in WIKI_HEADING_RE.finditer(text):
        positions.append((m.start(), m.end(), m.group(2).strip()))
    if file_type == "code":
        for m in CODE_HEADING_RE.finditer(text):
            positions.append((m.start(), m.end(), m.group("name")))
    positions.sort(key=lambda p: p[0])
    return positions


def _paragraph_spans(text: str) -> list[_Span]:
    """Абзацы, разделённые пустыми строками."""
    spans: list[_Span] = []
    start = 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        seg = text[start:m.start()]
        if seg.strip():
            spans.append(_Span(start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append(_Span(start, len(text)))
    return spans


def _split_long_section(text: str, base: int, max_chars: int) -> list[tuple[int, int, str]]:
    """Делит длинную секцию по абзацам, не разрывая абзац.

    text — текст секции (относительные смещения), base — абсолютное смещение
    начала секции в документе. Возвращает (абс. start, абс. end, текст).
    """
    paras = _paragraph_spans(text)
    if not paras:
        return [(base, base + len(text), text)]

    out: list[tuple[int, int, str]] = []
    cur_texts: list[str] = []
    cur_start = cur_end = 0

    for p in paras:
        ptext = text[p.start:p.end]
        if cur_texts and (p.end - cur_start) > max_chars:
            out.append((base + cur_start, base + cur_end, "\n\n".join(cur_texts)))
            cur_texts = [ptext]
            cur_start, cur_end = p.start, p.end
        else:
            if cur_texts:
                cur_end = p.end
            else:
                cur_start, cur_end = p.start, p.end
            cur_texts.append(ptext)

    if cur_texts:
        out.append((base + cur_start, base + cur_end, "\n\n".join(cur_texts)))
    return out


def structural_chunks(doc: Document, max_chars: int = 1500) -> list[Chunk]:
    """Чанки по структуре документа (заголовки/разделы/функции)."""
    masked = _mask_code_fences(doc.text)
    positions = _heading_positions(masked, doc.file_type)

    # сегменты (section, start, end) между заголовками
    segments: list[tuple[str, int, int]] = []
    if not positions:
        segments.append(("", 0, len(doc.text)))
    else:
        if positions[0][0] > 0:
            segments.append(("", 0, positions[0][0]))
        for i, (hs, _he, title) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(doc.text)
            segments.append((title, hs, end))

    chunks: list[Chunk] = []
    index = 0
    for section, s, e in segments:
        seg = doc.text[s:e]
        if not seg.strip():
            continue
        if (e - s) <= max_chars:
            chunks.append(_make_chunk(doc, "structural", index, seg, s, e, section))
            index += 1
        else:
            for cs, ce, ctext in _split_long_section(seg, s, max_chars):
                if ctext.strip():
                    chunks.append(
                        _make_chunk(doc, "structural", index, ctext, cs, ce, section)
                    )
                    index += 1

    return chunks


def chunk_document(doc: Document, strategy: str, **kwargs) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size_chunks(doc, **kwargs)
    if strategy == "structural":
        return structural_chunks(doc, **kwargs)
    raise ValueError(f"unknown strategy: {strategy}")
