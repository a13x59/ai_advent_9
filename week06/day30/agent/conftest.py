# conftest.py
"""Настройка pytest: направляем синглтон хранилища во временный файл,
чтобы тесты не трогали рабочий agent_history.db проекта."""
import os
import tempfile


_tmp = tempfile.mkstemp(prefix="agent_test_", suffix=".db")
os.close(_tmp[0])
os.environ["AGENT_HISTORY_DB"] = _tmp[1]
# В тестах пайплайна rate limit не нужен — отключаем большим лимитом.
# Тесты самого лимита (test_service_limits.py) выставляют свой низкий лимит сами.
os.environ["AGENT_RATE_LIMIT_PER_MINUTE"] = "100000"


def pytest_sessionfinish(session, exitstatus):
    try:
        os.remove(_tmp[1])
    except OSError:
        pass
