"""Unit-тесты pr_review (без сети и тяжёлых зависимостей)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # day32

from pr_review.diff_utils import (  # noqa: E402
    changed_files, chunk_diff, is_generated, parse_diff,
)
from pr_review import context as pr_ctx  # noqa: E402
from pr_review.prompt import (  # noqa: E402
    build_review_messages, section_has_findings, split_sections,
)
from pr_review.engine import run_review  # noqa: E402
from pr_review.review_pr import should_fail  # noqa: E402

SAMPLE_DIFF = """\
diff --git a/agent/storage.py b/agent/storage.py
--- a/agent/storage.py
+++ b/agent/storage.py
@@ -10,6 +10,9 @@ def load_working(session_id, profile_id):
-    return conn.execute("SELECT * FROM working").fetchall()
+    rows = conn.execute("SELECT * FROM working WHERE session_id = ?", (session_id,))
+    return rows
diff --git a/yarn.lock b/yarn.lock
new file mode 100644
--- /dev/null
+++ b/yarn.lock
@@ -0,0 +1,3 @@
+some lockfile content
"""

REVIEW_WITH_SECTIONS = """\
Преамбула, которую надо отбросить.

## Потенциальные баги
- Возможен N+1 в `load_working`.

## Архитектурные проблемы
- Смешаны слои доступа к данным.

## Рекомендации
- Добавить индекс по session_id.
"""


def test_parse_diff_and_changed_files():
    files = parse_diff(SAMPLE_DIFF)
    assert [f.path for f in files] == ["agent/storage.py", "yarn.lock"]
    assert files[0].status == "modified"
    assert files[1].status == "added"


def test_changed_files_filters_generated():
    # yarn.lock — сгенерированный, должен отсеиваться.
    assert changed_files(SAMPLE_DIFF) == ["agent/storage.py"]


def test_is_generated():
    assert is_generated("node_modules/x.js")
    assert is_generated("app/build/out.js")
    assert is_generated("requirements.txt")
    assert not is_generated("src/main.kt")
    assert not is_generated("agent/storage.py")


def test_chunk_diff_respects_budget():
    blocks = chunk_diff(SAMPLE_DIFF, max_tokens=1000)
    assert len(blocks) >= 1
    assert "agent/storage.py" in blocks[0]
    assert "yarn.lock" not in blocks[0]  # отфильтрован как generated


def test_split_sections():
    s = split_sections(REVIEW_WITH_SECTIONS)
    assert "N+1" in s["Потенциальные баги"]
    assert "слои" in s["Архитектурные проблемы"]
    assert "индекс" in s["Рекомендации"]


def test_build_review_messages_contains_diff_and_files():
    msgs = build_review_messages(SAMPLE_DIFF, ["agent/storage.py"])
    assert msgs[0]["role"] == "system"
    assert "agent/storage.py" in msgs[1]["content"]
    assert "load_working" in msgs[1]["content"]


def test_should_fail():
    buggy = {"Потенциальные баги": "- SQL-инъекция",
             "Архитектурные проблемы": "не обнаружено"}
    assert should_fail(buggy, "bugs") is True
    assert should_fail(buggy, "all") is True
    assert should_fail(buggy, "none") is False
    clean = {"Потенциальные баги": "не обнаружено",
             "Архитектурные проблемы": "не обнаружено"}
    assert should_fail(clean, "bugs") is False
    assert should_fail(clean, "all") is False


def test_section_has_findings():
    assert section_has_findings("- Возможен N+1") is True
    assert section_has_findings("не обнаружено") is False
    assert section_has_findings("") is False
    assert section_has_findings("Проблем не обнаружено") is False
    # смешанный случай: есть реальная строка с замечанием
    assert section_has_findings("Критичных не обнаружено\n- Утечка соединения") is True


def test_normalize_path_handles_monorepo_prefix():
    assert pr_ctx.normalize_path("week07/day32/agent/rag_client.py") == "agent/rag_client.py"
    assert pr_ctx.normalize_path("agent/rag_client.py") == "agent/rag_client.py"
    # read_current_file должен читать файл и по репо-корневому, и по day32-пути
    assert pr_ctx.read_current_file("week07/day32/agent/rag_client.py")
    assert pr_ctx.read_current_file("agent/rag_client.py")


def test_run_review_with_fake_completion():
    def fake_completion(messages):
        assert messages[0]["role"] == "system"
        return REVIEW_WITH_SECTIONS

    result = run_review(SAMPLE_DIFF, provider="deepseek", use_rag=False,
                        completion=fake_completion)
    assert result.changed_files == ["agent/storage.py"]
    assert "N+1" in result.sections["Потенциальные баги"]
    assert result.provider == "deepseek"
