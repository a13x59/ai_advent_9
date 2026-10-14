#!/usr/bin/env python3
"""CLI AI-ревью PR (подход B — headless/lean).

Одна точка входа для локального запуска и GitHub Action. Переключатели:

  --backend script|agent   как выполнять ревью:
      script — локально, напрямую (engine.py), без FastAPI (по умолчанию);
      agent  — POST в работающий агент (endpoint /review), полный пайплайн.

  --provider deepseek|ollama   какой моделью ревьюить (для backend=script).

Примеры:
  python -m pr_review.review_pr --diff-file pr.diff --provider deepseek
  python -m pr_review.review_pr --base origin/main --head HEAD --backend agent
  python -m pr_review.review_pr --diff-file pr.diff --post-comment --pr-number 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # чтобы импортировать pr_review из любого места

from pr_review import engine as eng  # noqa: E402
from pr_review import github as gh  # noqa: E402
from pr_review.prompt import render_pr_comment  # noqa: E402

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")


def _git_diff(base: str | None, head: str | None) -> str:
    """Получение diff через git. base...head (three-dot) — изменения head от
    merge-base с base (то, что привносит PR)."""
    if base and head:
        cmd = ["git", "diff", f"{base}...{head}"]
    elif head:
        cmd = ["git", "diff", head]
    else:
        cmd = ["git", "diff", "HEAD"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git diff failed: {proc.stderr.strip()}")
    return proc.stdout


def _read_diff(args) -> str:
    if args.diff_file:
        return Path(args.diff_file).read_text(encoding="utf-8")
    return _git_diff(args.base, args.head)


def _run_agent_backend(diff_text: str, args) -> dict:
    import requests
    payload = {
        "diff": diff_text,
        "model": args.model or os.environ.get("REVIEW_MODEL", "deepseek-chat"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "use_rag": not args.no_rag,
    }
    resp = requests.post(AGENT_BASE_URL.rstrip("/") + "/review", json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"agent /review HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _run_script_backend(diff_text: str, args) -> dict:
    result = eng.run_review(
        diff_text,
        provider=args.provider,
        model=args.model,
        use_rag=not args.no_rag,
        rag_base_url=args.rag_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return {
        "review_text": result.review_text,
        "sections": result.sections,
        "changed_files": result.changed_files,
        "rag_chunks": result.rag_chunks,
        "provider": result.provider,
        "model": result.model,
        "prompt_tokens_estimate": result.prompt_tokens_estimate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-ревью PR")
    parser.add_argument("--diff-file", help="путь к файлу с unified diff")
    parser.add_argument("--base", help="базовая ветка/реф (git diff base...head)")
    parser.add_argument("--head", help="головной реф (по умолчанию HEAD)")
    parser.add_argument("--backend", choices=["script", "agent"],
                        default=os.environ.get("REVIEW_BACKEND", "script"))
    parser.add_argument("--provider", choices=["deepseek", "ollama"],
                        default=os.environ.get("REVIEW_PROVIDER", "deepseek"))
    parser.add_argument("--model", default=os.environ.get("REVIEW_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--rag-base-url", default=os.environ.get("RAG_BASE_URL"))
    parser.add_argument("--no-rag", action="store_true", help="отключить RAG")
    parser.add_argument("--post-comment", action="store_true", help="опубликовать комментарий в PR")
    parser.add_argument("--pr-number", type=int, default=os.environ.get("PR_NUMBER") and int(os.environ["PR_NUMBER"]))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--output", help="сохранить ревью в файл (markdown)")
    parser.add_argument("--json", action="store_true", help="вывести JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="только разобрать diff и показать план (без вызова LLM)")
    args = parser.parse_args(argv)

    diff_text = _read_diff(args)
    if not diff_text.strip():
        print("diff пуст — нечего ревьюить", file=sys.stderr)
        return 0

    if args.dry_run:
        from pr_review.diff_utils import changed_files, estimate_tokens
        files = changed_files(diff_text)
        print(f"backend={args.backend} provider={args.provider}")
        print(f"diff: {len(diff_text)} симв. (~{estimate_tokens(diff_text)} токенов)")
        print(f"изменённых файлов (после фильтра): {len(files)}")
        for f in files:
            print("  -", f)
        return 0

    if args.backend == "agent":
        data = _run_agent_backend(diff_text, args)
    else:
        data = _run_script_backend(diff_text, args)

    review_text = data.get("review_text", "")
    print(review_text)

    if args.output:
        Path(args.output).write_text(render_pr_comment(review_text, data.get("changed_files")),
                                     encoding="utf-8")

    if args.post_comment:
        body = render_pr_comment(review_text, data.get("changed_files"))
        comment = gh.post_pr_comment(body, pr_number=args.pr_number, repo=args.repo)
        data["comment_url"] = comment.get("html_url")

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
