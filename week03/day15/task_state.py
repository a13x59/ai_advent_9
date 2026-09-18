# task_state.py
"""Формализованный конечный автомат задачи (Task State Machine).

Состояние задачи описывается тремя обязательными аспектами:
  • stage           — этап задачи: planning → execution → validation → done;
  • step            — текущий шаг (step_index / step_total / step_label);
  • expected_action — ожидаемое действие: кто и что делает дальше.

Дополнительно у задачи есть ЖИЗНЕННЫЙ статус (status ∈ {active, paused, done}),
ортогональный этапу: пауза возможна на ЛЮБОМ этапе, а в одной сессии может быть
несколько задач, но активной (не на паузе) — только одна.

Архитектурное разделение:
  • модель (LLM) сообщает прогресс через скрытый маркер [STATE] {...} (реальный API)
    или используется детерминированный переход (мок/фолбэк);
  • этот модуль ВАЛИДИРУЕТ переход (только вперёд по этапам), нормализует
    значения и рендерит состояние в контекст.

Таким образом конечный автомат — источник истины: состояние отражает реальный
прогресс, не откатывается назад и переживает урезание окна истории.
"""
from typing import Any, Dict, List, Tuple

# Этапы задачи (конечный автомат).
STAGES: Tuple[str, ...] = ("planning", "execution", "validation", "done")

# Жизненный статус задачи (ортогонален этапу).
STATUSES: Tuple[str, ...] = ("active", "paused", "done")

# Ожидаемое действие — кто действует дальше.
EXPECTED_ACTIONS: Tuple[str, ...] = ("model_call", "wait_user", "confirm", "done")

# Состояния шагов плана.
PLAN_STATES: Tuple[str, ...] = ("pending", "in_progress", "done", "blocked")

# Таблица разрешённых переходов между этапами. Перескок через этап ЗАПРЕЩЁН:
# execution → done невозможно, только execution → validation → done.
ALLOWED_STAGE_TRANSITIONS: Dict[str, set] = {
    "planning": {"planning", "execution"},
    "execution": {"execution", "validation"},
    "validation": {"execution", "validation", "done"},
    "done": {"done"},
}


def is_valid_stage(value: Any) -> bool:
    return value in STAGES


def is_valid_status(value: Any) -> bool:
    return value in STATUSES


def is_valid_expected_action(value: Any) -> bool:
    return value in EXPECTED_ACTIONS


def normalize_stage(value: Any, default: str = "planning") -> str:
    return value if is_valid_stage(value) else default


def normalize_status(value: Any, default: str = "active") -> str:
    return value if is_valid_status(value) else default


def normalize_expected_action(value: Any, default: str = "wait_user") -> str:
    return value if is_valid_expected_action(value) else default


def can_transition(from_stage: str, to_stage: str) -> bool:
    """True, если переход между этапами разрешён таблицей."""
    if not is_valid_stage(from_stage) or not is_valid_stage(to_stage):
        return False
    return to_stage in ALLOWED_STAGE_TRANSITIONS.get(from_stage, set())


def forward_stage_path(from_stage: str, to_stage: str) -> List[str]:
    """Список промежуточных этапов для движения вперёд СТРОГО по таблице переходов.

    Например, execution → done вернёт ["validation", "done"] — перескок этапа
    невозможен, но каждый шаг цепочки легален.
    """
    if not is_valid_stage(from_stage) or not is_valid_stage(to_stage):
        return []
    fi = STAGES.index(from_stage)
    ti = STAGES.index(to_stage)
    if ti <= fi:
        return []
    return list(STAGES[fi + 1:ti + 1])


def coerce_transition(from_stage: str, requested_stage: Any) -> str:
    """Возвращает допустимый целевой этап (мягкая коэрция при запрете)."""
    requested = normalize_stage(requested_stage, from_stage)
    if can_transition(from_stage, requested):
        return requested
    return from_stage


def normalize_plan(plan: Any) -> List[dict]:
    """Приводит план шагов (от модели) к каноническому виду."""
    out: List[dict] = []
    for i, item in enumerate(plan or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        state = item.get("state") if item.get("state") in PLAN_STATES else "pending"
        try:
            index = int(item.get("index") or (i + 1))
        except (TypeError, ValueError):
            index = i + 1
        out.append({"index": index, "label": label, "state": state})
    return out


def build_resume_note(stage: str, step_index: int, step_total: int, step_label: str) -> str:
    """Краткое описание текущего положения — точка восстановления после паузы."""
    if stage == "done":
        return "Задача завершена."
    position = f"шаг {step_index}/{step_total}" if step_total else "этап планирования"
    label = f": {step_label}" if step_label else ""
    return f"этап {stage}, {position}{label}"


def apply_state_update(task: Dict[str, Any], update: Dict[str, Any]) -> Tuple[dict, dict]:
    """Валидирует предложение модели и возвращает (fields, transition).

    fields     — поля задачи для сохранения (без служебных timestamp).
    transition — словарь для журнала переходов.
    """
    current_stage = task.get("stage", "planning")
    new_stage = coerce_transition(current_stage, update.get("stage"))

    # Копия, чтобы не мутировать входной словарь задачи.
    plan = [dict(s) for s in (task.get("plan") or [])]
    if update.get("plan"):
        plan = normalize_plan(update["plan"])

    step_total = update.get("step_total")
    if step_total is None:
        step_total = len(plan) if plan else (task.get("step_total") or 0)
    step_total = int(step_total)

    step_index = update.get("step_index")
    if step_index is None:
        step_index = task.get("step_index") or 0
    step_index = int(step_index)

    step_label = str(update.get("step_label") or task.get("step_label") or "").strip()

    if new_stage == "done":
        status = "done"
        expected_action = "done"
    else:
        status = normalize_status(task.get("status"), "active")
        expected_action = normalize_expected_action(
            update.get("expected_action"), task.get("expected_action", "wait_user")
        )

    # Синхронизируем состояния шагов плана с фактическим прогрессом.
    # На этапах validation/done выполнение считается завершённым — все шаги done.
    if new_stage in ("validation", "done"):
        if step_total and step_index < step_total:
            step_index = step_total
        for s in plan:
            s["state"] = "done"
    else:
        for s in plan:
            idx = s.get("index", 0)
            if step_index and idx < step_index:
                s["state"] = "done"
            elif step_index and idx == step_index:
                s["state"] = "in_progress"

    fields = {
        "stage": new_stage,
        "status": status,
        "step_index": step_index,
        "step_total": step_total,
        "step_label": step_label,
        "expected_action": expected_action,
        "plan": plan,
        "resume_note": build_resume_note(new_stage, step_index, step_total, step_label),
    }
    transition = {
        "from_stage": current_stage,
        "to_stage": new_stage,
        "step_label": step_label,
        "expected_action": expected_action,
        "reason": str(update.get("reason") or ""),
    }
    return fields, transition


_STEP_MARKERS = {"done": "[x]", "in_progress": "[>]", "blocked": "[!]", "pending": "[ ]"}


def render_task_block(task: Dict[str, Any]) -> str:
    """Системный блок с текущим состоянием задачи для контекста модели."""
    if not task:
        return ""
    lines = ["Текущая задача (конечный автомат):"]
    title = (task.get("title") or "").strip()
    if title:
        lines.append(f"- Название: {title}")
    lines.append(f"- Этап: {task.get('stage', 'planning')}")
    lines.append(f"- Статус: {task.get('status', 'active')}")
    if task.get("step_total"):
        lines.append(
            f"- Текущий шаг: {task.get('step_index', 0)}/{task.get('step_total', 0)}"
            f" · {task.get('step_label') or '—'}"
        )
    lines.append(f"- Ожидаемое действие: {task.get('expected_action', 'wait_user')}")
    plan = task.get("plan") or []
    if plan:
        lines.append("- План:")
        for s in plan:
            marker = _STEP_MARKERS.get(s.get("state"), "[ ]")
            lines.append(f"  {marker} {s.get('index', '?')}. {s.get('label', '')}")
    resume_note = (task.get("resume_note") or "").strip()
    if resume_note:
        lines.append(f"- Где остановились: {resume_note}")
    return "\n".join(lines)


def render_resume_instruction(task: Dict[str, Any]) -> str:
    """Инструкция возобновления без повторных объяснений."""
    title = (task.get("title") or "Задача").strip()
    note = (task.get("resume_note") or "").strip()
    return (
        f"Задача «{title}» возобновлена после паузы. Ты остановился на: {note}. "
        "Продолжай с этого места — не объясняй задачу заново и не переспрашивай, что делать."
    )



