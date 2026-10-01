# test_agent_flow.py
"""Интеграционные тесты пайплайна на мок-провайдере (без сети и без ключа).

Проверяют жизненный цикл задачи с явными переходами и гвардами, реакцию на
недопустимые переходы и корректность после паузы.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from agent_core import create_app
from storage import HistoryStorage
from mock_agent import create_mock_app, MockProvider


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(prefix="flow_", suffix=".db")
    os.close(fd)
    app = create_mock_app(path)
    with TestClient(app) as c:
        yield c
    os.remove(path)


def post_agent(client, text, session_id="sess1", task_id=None):
    body = {
        "session_id": session_id,
        "model": "mock-model",
        "messages": [{"role": "user", "content": text}],
    }
    if task_id:
        body["task_id"] = task_id
    return client.post("/agent", json=body).json()


def test_full_lifecycle_with_approval(client):
    # 1. Новая задача → планирование, план, ожидание утверждения.
    r = post_agent(client, "напиши калькулятор")
    t = r["task"]
    assert t["stage"] == "planning"
    assert t["expected_action"] == "confirm"
    assert t["guards"]["plan_approved"] is False
    assert len(t["plan"]) == 3
    task_id = t["task_id"]

    # 2. Реализация до утверждения плана невозможна.
    resp = client.post(f"/agent/sess1/tasks/{task_id}/transition", json={"stage": "execution"})
    assert resp.status_code == 409
    assert "план утверждён" in resp.json()["detail"]["reason"]

    # 3. Утверждаем план → execution.
    r = post_agent(client, "да")
    t = r["task"]
    assert t["stage"] == "execution"
    assert t["guards"]["plan_approved"] is True
    assert t["step_index"] == 1

    # 4. Проходим шаги до валидации.
    r = post_agent(client, "продолжай")
    assert r["task"]["step_index"] == 2
    assert r["task"]["stage"] == "execution"
    r = post_agent(client, "продолжай")
    assert r["task"]["step_index"] == 3
    r = post_agent(client, "продолжай")
    assert r["task"]["stage"] == "validation"
    assert r["task"]["expected_action"] == "confirm"
    assert r["task"]["guards"]["validation_passed"] is False

    # 5. Финал без подтверждения валидации невозможен.
    resp = client.post(f"/agent/sess1/tasks/{task_id}/transition", json={"stage": "done"})
    assert resp.status_code == 409
    assert "валидация подтверждена" in resp.json()["detail"]["reason"]

    # 6. Подтверждаем валидацию → done.
    r = post_agent(client, "да")
    t = r["task"]
    assert t["stage"] == "done"
    assert t["guards"]["validation_passed"] is True
    assert t["status"] == "done"


def test_forward_jump_rejected_and_logged(client):
    r = post_agent(client, "напиши калькулятор")
    task_id = r["task"]["task_id"]

    resp = client.post(f"/agent/sess1/tasks/{task_id}/transition", json={"stage": "done"})
    assert resp.status_code == 409
    assert "Нельзя перепрыгнуть этап" in resp.json()["detail"]["reason"]
    assert resp.json()["detail"]["allowed"] == ["planning"]

    # Отклонённая попытка попала в журнал переходов (accepted=0).
    transitions = client.get(f"/agent/sess1/tasks/{task_id}").json()["transitions"]
    assert any(tr["accepted"] == 0 and tr["to_stage"] == "done" for tr in transitions)


def test_transitions_endpoint(client):
    r = post_agent(client, "напиши калькулятор")
    task_id = r["task"]["task_id"]

    data = client.get(f"/agent/sess1/tasks/{task_id}/transitions").json()
    assert data["current_stage"] == "planning"
    assert data["allowed"] == ["planning"]
    assert any(b["stage"] == "execution" for b in data["blocked"])


def test_pause_resume_preserves_state(client):
    r = post_agent(client, "напиши калькулятор")
    task_id = r["task"]["task_id"]
    post_agent(client, "да")  # planning → execution, шаг 1

    paused = client.post(f"/agent/sess1/tasks/{task_id}/pause").json()["task"]
    assert paused["status"] == "paused"
    assert paused["stage"] == "execution"
    assert paused["step_index"] == 1
    assert paused["guards"]["plan_approved"] is True

    resumed = client.post(f"/agent/sess1/tasks/{task_id}/resume").json()["task"]
    assert resumed["status"] == "active"
    assert resumed["stage"] == "execution"
    assert resumed["step_index"] == 1
    assert resumed["guards"]["plan_approved"] is True

    # Продолжение после паузы идёт с того же места.
    r = post_agent(client, "продолжай")
    assert r["task"]["step_index"] == 2
    assert r["task"]["stage"] == "execution"


def test_new_imperative_creates_new_task_and_pauses_previous(client):
    post_agent(client, "напиши калькулятор")
    r = post_agent(client, "напиши тесты")
    # Новая задача стала активной, предыдущая ушла в паузу.
    tasks = client.get("/agent/sess1/tasks").json()["tasks"]
    active = [t for t in tasks if t["status"] == "active"]
    paused = [t for t in tasks if t["status"] == "paused"]
    assert len(active) == 1
    assert active[0]["task_id"] == r["task"]["task_id"]
    assert len(paused) == 1


def test_model_forward_jump_is_blocked_and_regenerated():
    """Модель попыталась перепрыгнуть этап: маркер отклонён, ответ перегенерирован,
    плохой ответ не попал в историю, отклонённая попытка записана в журнал."""
    class JumpyProvider(MockProvider):
        def __init__(self):
            super().__init__()
            self.first = True

        def complete(self, messages, request, task, user_text):
            if self.first and task is not None and task.get("stage") == "planning":
                self.first = False
                return {
                    "choices": [{"message": {"content":
                        "Сделал всё сразу!\n[STATE] {\"stage\": \"done\"}"}}]
                }
            return super().complete(messages, request, task, user_text)

    fd, path = tempfile.mkstemp(prefix="jump_", suffix=".db")
    os.close(fd)
    app = create_app(JumpyProvider(), HistoryStorage(path))
    try:
        with TestClient(app) as c:
            r = c.post("/agent", json={
                "session_id": "s2", "model": "m",
                "messages": [{"role": "user", "content": "напиши калькулятор"}],
            }).json()
            t = r["task"]
            # Переход отклонён — состояние осталось на planning.
            assert t["stage"] == "planning"
            assert t["expected_action"] == "confirm"

            # Плохой ответ не сохранился в истории задачи (был re-prompt).
            msgs = c.get(f"/agent/s2/tasks/{t['task_id']}").json()["task"]["messages"]
            assert not any("Сделал всё сразу" in (m.get("content") or "") for m in msgs)

            # Отклонённая попытка залогирована как accepted=0.
            trs = c.get(f"/agent/s2/tasks/{t['task_id']}").json()["transitions"]
            assert any(tr["accepted"] == 0 and tr["to_stage"] == "done" for tr in trs)
    finally:
        os.remove(path)
