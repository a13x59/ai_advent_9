# test_task_state.py
"""Юнит-тесты конечного автомата: гварды, перескоки, явное отклонение."""
from task_state import (
    DEFAULT_GUARDS,
    allowed_next_stages,
    blocked_next_stages,
    normalize_guards,
    validate_and_apply,
)


def make_task(stage="planning", guards=None, plan=None, step_index=0, step_total=0,
              expected_action="wait_user", status="active"):
    return {
        "task_id": "t1",
        "session_id": "s1",
        "title": "Задача",
        "stage": stage,
        "status": status,
        "step_index": step_index,
        "step_total": step_total,
        "step_label": "",
        "expected_action": expected_action,
        "plan": plan or [],
        "guards": dict(guards or DEFAULT_GUARDS),
        "resume_note": "",
    }


PLAN = [
    {"index": 1, "label": "Шаг 1", "state": "in_progress"},
    {"index": 2, "label": "Шаг 2", "state": "pending"},
]


def test_reject_invalid_stage():
    task = make_task(stage="planning")
    verdict = validate_and_apply(task, {"stage": "foo"}, source="model")
    assert verdict["accepted"] is False
    assert "Недопустимое состояние" in verdict["reason"]


def test_reject_forward_jump_planning_to_done():
    task = make_task(stage="planning")
    verdict = validate_and_apply(task, {"stage": "done"}, source="model")
    assert verdict["accepted"] is False
    assert "Нельзя перепрыгнуть этап" in verdict["reason"]


def test_reject_forward_jump_execution_to_done():
    task = make_task(stage="execution", guards={"plan_approved": True, "validation_passed": False},
                     plan=PLAN, step_index=1, step_total=2)
    verdict = validate_and_apply(task, {"stage": "done"}, source="model")
    assert verdict["accepted"] is False
    assert "Нельзя перепрыгнуть этап" in verdict["reason"]


def test_reject_execution_without_plan_approved():
    task = make_task(stage="planning")  # guards по умолчанию: plan_approved=False
    verdict = validate_and_apply(task, {"stage": "execution"}, source="model")
    assert verdict["accepted"] is False
    assert "план утверждён" in verdict["reason"]


def test_reject_done_without_validation_passed():
    task = make_task(stage="validation", guards={"plan_approved": True, "validation_passed": False},
                     plan=PLAN, step_index=2, step_total=2)
    verdict = validate_and_apply(task, {"stage": "done"}, source="model")
    assert verdict["accepted"] is False
    assert "валидация подтверждена" in verdict["reason"]


def test_accept_planning_self_loop_with_plan():
    task = make_task(stage="planning")
    verdict = validate_and_apply(
        task,
        {"stage": "planning", "expected_action": "confirm", "plan": PLAN, "step_total": 2},
        source="model",
    )
    assert verdict["accepted"] is True
    assert verdict["fields"]["stage"] == "planning"
    assert verdict["fields"]["expected_action"] == "confirm"
    assert len(verdict["fields"]["plan"]) == 2


def test_accept_approval_planning_to_execution():
    task = make_task(stage="planning", plan=PLAN, step_total=2)
    verdict = validate_and_apply(
        task,
        {
            "stage": "execution",
            "guards": {"plan_approved": True},
            "step_index": 1,
            "step_total": 2,
            "step_label": "Шаг 1",
            "expected_action": "wait_user",
        },
        source="user",
    )
    assert verdict["accepted"] is True
    assert verdict["fields"]["stage"] == "execution"
    assert verdict["fields"]["guards"]["plan_approved"] is True
    assert verdict["fields"]["step_index"] == 1


def test_accept_execution_to_validation():
    task = make_task(stage="execution", guards={"plan_approved": True, "validation_passed": False},
                     plan=PLAN, step_index=2, step_total=2)
    verdict = validate_and_apply(
        task, {"stage": "validation", "expected_action": "confirm"}, source="model"
    )
    assert verdict["accepted"] is True
    assert verdict["fields"]["stage"] == "validation"
    # на validation/done все шаги плана закрываются
    assert all(s["state"] == "done" for s in verdict["fields"]["plan"])


def test_accept_approval_validation_to_done():
    task = make_task(stage="validation", guards={"plan_approved": True, "validation_passed": False},
                     plan=PLAN, step_index=2, step_total=2)
    verdict = validate_and_apply(
        task,
        {
            "stage": "done",
            "guards": {"validation_passed": True},
            "step_index": 2,
            "step_total": 2,
            "expected_action": "done",
        },
        source="user",
    )
    assert verdict["accepted"] is True
    assert verdict["fields"]["stage"] == "done"
    assert verdict["fields"]["guards"]["validation_passed"] is True


def test_model_cannot_set_guards():
    # Модель не может сама себе «утвердить» план: guards из update игнорируются.
    task = make_task(stage="planning")
    verdict = validate_and_apply(
        task,
        {"stage": "execution", "guards": {"plan_approved": True}},
        source="model",
    )
    assert verdict["accepted"] is False
    assert "план утверждён" in verdict["reason"]


def test_allowed_and_blocked_reflect_guards():
    task = make_task(stage="planning")
    assert allowed_next_stages(task) == ["planning"]
    blocked = blocked_next_stages(task)
    assert any(b["stage"] == "execution" and "plan_approved" in b["missing_guards"] for b in blocked)

    approved = make_task(stage="planning", guards={"plan_approved": True, "validation_passed": False})
    assert allowed_next_stages(approved) == ["planning", "execution"]


def test_normalize_guards_defaults():
    assert normalize_guards(None) == DEFAULT_GUARDS
    assert normalize_guards({"plan_approved": True}) == {
        "plan_approved": True,
        "validation_passed": False,
    }
