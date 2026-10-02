"""Одноразовый скрипт сбора документов для RAG-индексации.

Собирает русскоязычные документы в data/raw/:
  - статьи русской Википедии (full plain text)      -> data/raw/articles/
  - README русских open-source проектов (GitHub raw) -> data/raw/docs/
  - фрагменты кода                                   -> data/raw/code/

Зависит только от стандартной библиотеки (urllib). Идемпотентен:
уже скачанные файлы не перекачивает. Пишет data/sources.md.
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
UA = {"User-Agent": "rag-course-assignment/1.0 (education)"}

WIKI_TITLES = [
    "Информационный поиск",
    "Векторное представление слов",
    "Обработка естественного языка",
    "Косинусное сходство",
    "Токенизация",
    "Трансформер (машинное обучение)",
    "Семантический поиск",
    "Лемматизация",
    "N-грамма",
    "Мешок слов",
    "Инвертированный индекс",
    "Word2vec",
    "Генерация, дополненная поиском",
]

README_URLS = {
    "natasha": "https://raw.githubusercontent.com/natasha/natasha/master/README.md",
    "razdel": "https://raw.githubusercontent.com/natasha/razdel/master/README.md",
    "rulm": "https://raw.githubusercontent.com/IlyaGusev/rulm/master/README.md",
}

CODE_URLS = {
    "razdel_tokenize.py": "https://raw.githubusercontent.com/natasha/razdel/master/razdel/segmenters/tokenize.py",
    "natasha_extractors.py": "https://raw.githubusercontent.com/natasha/natasha/master/natasha/extractors.py",
    "requests_models.py": "https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py",
}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wikipedia_text(title: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    url = "https://ru.wikipedia.org/w/api.php?" + params
    data = json.loads(_get(url).decode("utf-8"))
    for page in data["query"]["pages"].values():
        if "extract" in page and page["extract"].strip():
            return page["extract"]
    return None


def slugify(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE).strip("_").lower()
    return s or "doc"


def wiki_url(title: str) -> str:
    return "https://ru.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def repo_url(raw_url: str) -> str:
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+/[^/]+)/", raw_url)
    return "https://github.com/" + m.group(1) if m else raw_url


def blob_url(raw_url: str) -> str:
    m = re.match(
        r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)",
        raw_url,
    )
    if m:
        return f"https://github.com/{m.group(1)}/{m.group(2)}/blob/{m.group(3)}/{m.group(4)}"
    return raw_url


def _record(path: Path, text: str, label: str) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"- {label} — {len(text)} симв.", len(text)


def main() -> int:
    report: list[str] = []
    total_chars = 0
    files_written = 0

    # 1) Википедия
    for title in WIKI_TITLES:
        path = DATA / "articles" / (slugify(title) + ".txt")
        if path.exists() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8")
            report.append(f"- Википедия: [{title}]({wiki_url(title)}) — {len(text)} симв.")
            total_chars += len(text)
            files_written += 1
            continue
        try:
            time.sleep(1.2)  # против 429 rate-limit
            text = wikipedia_text(title)
            if not text:
                report.append(f"- [SKIP] Википедия: {title} (не найдена)")
                continue
            line, n = _record(path, text, f"Википедия: [{title}]({wiki_url(title)})")
            report.append(line)
            total_chars += n
            files_written += 1
        except Exception as exc:  # noqa: BLE001
            report.append(f"- [ERR] Википедия: {title} — {exc}")

    # 2) README
    for name, url in README_URLS.items():
        path = DATA / "docs" / f"{name}_README.md"
        if path.exists() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8")
            report.append(f"- README: [{name}]({repo_url(url)}) — {len(text)} симв.")
            total_chars += len(text)
            files_written += 1
            continue
        try:
            raw = _get(url).decode("utf-8", errors="replace")
            line, n = _record(path, raw, f"README: [{name}]({repo_url(url)})")
            report.append(line)
            total_chars += n
            files_written += 1
        except Exception as exc:  # noqa: BLE001
            report.append(f"- [ERR] README: {name} — {exc}")

    # 3) Код
    for name, url in CODE_URLS.items():
        path = DATA / "code" / name
        if path.exists() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8")
            report.append(f"- Код: [{name}]({blob_url(url)}) — {len(text)} симв.")
            total_chars += len(text)
            files_written += 1
            continue
        try:
            raw = _get(url).decode("utf-8", errors="replace")
            line, n = _record(path, raw, f"Код: [{name}]({blob_url(url)})")
            report.append(line)
            total_chars += n
            files_written += 1
        except Exception as exc:  # noqa: BLE001
            report.append(f"- [ERR] Код: {name} — {exc}")

    pages = total_chars / 1800
    header = (
        "# Источники документов\n\n"
        f"Суммарно: {total_chars:,} символов (~{pages:.1f} страниц), "
        f"файлов: {files_written}.\n\n"
        "Оценка страницы: ~1800 символов (задание требует 20–30 страниц).\n\n"
        "## Перечень\n\n"
    )
    (DATA / "sources.md").write_text(header + "\n".join(report) + "\n", encoding="utf-8")
    print(header)
    print("\n".join(report))
    print(f"\nИтого: {total_chars:,} симв. (~{pages:.1f} страниц), файлов: {files_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
