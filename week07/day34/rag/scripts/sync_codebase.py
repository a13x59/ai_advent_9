#!/usr/bin/env python3
"""CLI синхронизации корпуса «codebase» (День 32).

Тонкая обёртка над rag/src/codebase.py. Обычно запускать вручную не нужно:
`python -m src.pipeline run --corpus codebase` синхронизирует симлинки сам.

Использование:
    python rag/scripts/sync_codebase.py            # создать/обновить ссылки
    python rag/scripts/sync_codebase.py --dry-run  # только показать
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rag/

from src.codebase import sync_codebase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="не писать на диск")
    args = parser.parse_args()
    sync_codebase(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
