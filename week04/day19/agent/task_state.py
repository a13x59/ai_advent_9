# task_state.py
"""Формализованный конечный автомат задачи (Task State Machine).

Состояние задачи описывается тремя обязательными аспектами:
  • stage           — этап задачи: planning → execution → validation → done;
  • step            — текущий шаг (step_index / step_total / step_label);
  • expected_action — ожидаемое действие: кто и что делает дальше.

Дополнительно у задачи есть ЖИЗНЕННЫЙ статус (status ∈ {active, paused, done}),
ортогональный этапу: пауза возможна на ЛЮБОМ этапе, а в одной сессии может быть
несколько задач, но активной (не на паузе) — только одна.

Явные переходы между состояниями
--------------------------------
Переход разрешён ТОЛЬКО если одновременно выполнены:

  1) целевой этап есть в таблице ALLOWED_STAGE_TRANSITIONS для текущего;
  2) нет перескока: за один ход можно продвинуться максимум на ОДИН этап вперёд
     (обратные переходы по таблице, например validation → execution, разрешены);
  3) выполнены гварды (guards) — условия входа в целевой этап:
       • execution требует guards["plan_approved"] == true;
       • done      требует guards["validation_passed"] == true.

Гварды меняет ТОЛЬКО пользователь (source="user" / "manual"): модель не может
сама «утвердить» план или «подтвердить» валидацию. Валидация возвращает явный
вердикт (accepted / rejected), а не молчаливую коэрцию.
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

# Таблица разрешённых переходов между этапами. Переход выполняется ТОЛЬКО на один
# этап за ход (перескок отклоняется в validate_and_apply).
ALLOWED_STAGE_TRANSITIONS: Dict[str, set] = {
    "planning": {"planning", "execution"},
    "execution": {"execution", "validation"},
    "validation": {"execution", "validation", "done"},
    "done": {"done"},
}

# Гварды — условия, без которых нельзя ВОЙТИ в этап.
DEFAULT_GUARDS: Dict[str, bool] = {"plan_approved": False, "validation_passed": False}

# Какой гвард требуется для входа в этап.
STAGE_GUARDS: Dict[str, Tuple[str, ...]] = {
    "execution": ("plan_approved",),
    "done": ("validation_passed",),
}

# Человекочитаемые названия гвардов (для сообщений и UI).
GUARD_LABELS: Dict[str, str] = {
    "plan_approved": "план утверждён",
    "validation_passed": "валидация подтверждена",
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


def normalize_guards(guards: Any) -> Dict[str, bool]:
    """Приводит словарь гвардов к каноническому виду (все ключи + bool)."""
    out = dict(DEFAULT_GUARDS)
    if isinstance(guards, dict):
        for key in DEFAULT_GUARDS:
            out[key] = bool(guards.get(key, DEFAULT_GUARDS[key]))
    return out


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


def _stage_index(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError:
        return -1


def missing_guards_for(guards: Dict[str, bool], stage: str) -> List[str]:
    """Список невыполненных гвардов, требуемых для входа в stage."""
    return [g for g in STAGE_GUARDS.get(stage, ()) if not guards.get(g, False)]


def is_forward_jump(from_stage: str, to_stage: str) -> bool:
    """True, если переход идёт вперёд больше чем на один этап (перескок)."""
    return _stage_index(to_stage) > _stage_index(from_stage) + 1


def allowed_next_stages(task: Dict[str, Any]) -> List[str]:
    """Этапы, в которые можно перейти ПРЯМО СЕЙЧАС (таблица + один шаг + гварды)."""
    from_stage = normalize_stage(task.get("stage"))
    guards = normalize_guards(task.get("guards"))
    out: List[str] = []
    for to in STAGES:  # канонический порядок этапов (не порядок set)
        if to not in ALLOWED_STAGE_TRANSITIONS.get(from_stage, set()):
            continue
        if is_forward_jump(from_stage, to):
            continue
        if to != from_stage and missing_guards_for(guards, to):
            continue
        out.append(to)
    return out


def blocked_next_stages(task: Dict[str, Any]) -> List[dict]:
    """Этапы, легальные по таблице/шагу, но заблокированные невыполненным гвардом."""
    from_stage = normalize_stage(task.get("stage"))
    guards = normalize_guards(task.get("guards"))
    out: List[dict] = []
    for to in STAGES:
        if to not in ALLOWED_STAGE_TRANSITIONS.get(from_stage, set()):
            continue
        if is_forward_jump(from_stage, to):
            continue
        if to == from_stage:
            continue
        missing = missing_guards_for(guards, to)
        if missing:
            out.append({"stage": to, "missing_guards": missing})
    return out


def build_resume_note(stage: str, step_index: int, step_total: int, step_label: str) -> str:
    """Краткое описание текущего положения — точка восстановления после паузы."""
    if stage == "done":
        return "Задача завершена."
    position = f"шаг {step_index}/{step_total}" if step_total else "этап планирования"
    label = f": {step_label}" if step_label else ""
    return f"этап {stage}, {position}{label}"


def validate_and_apply(task: Dict[str, Any], update: Dict[str, Any], source: str = "model") -> dict:
    """Валидирует предложенный переход и возвращает явный вердикт.

    Никакой тихой коэрции: если переход недопустим, возвращается rejected с причиной
    и списком разрешённых этапов.

    Возвращает dict:
      accepted   — True/False;
      reason     — причина перехода (accepted) или отказа (rejected);
      allowed    — этапы, доступные сейчас;
      blocked    — этапы, заблокированные невыполненными гвардами;
      fields     — поля задачи для сохранения (только при accepted);
      transition — запись для журнала переходов (только при accepted).
    """
    update = update or {}
    from_stage = normalize_stage(task.get("stage"))
    allowed = allowed_next_stages(task)
    blocked = blocked_next_stages(task)

    verdict: Dict[str, Any] = {
        "accepted": False,
        "reason": "",
        "allowed": allowed,
        "blocked": blocked,
        "fields": None,
        "transition": None,
    }

    # 0. Недопустимое значение этапа — явный отказ.
    raw_stage = update.get("stage")
    if raw_stage is not None and not is_valid_stage(raw_stage):
        verdict["reason"] = (
            f"Недопустимое состояние: {raw_stage!r}. Допустимые: {', '.join(STAGES)}."
        )
        return verdict
    requested = normalize_stage(raw_stage, from_stage)

    # 1. Запрет перескока через этап (проверяем до таблицы, чтобы дать точную причину).
    if is_forward_jump(from_stage, requested):
        verdict["reason"] = (
            f"Нельзя перепрыгнуть этап: «{from_stage} → {requested}». "
            f"За один ход можно только: {', '.join(allowed) or '—'}."
        )
        return verdict

    # 2. Таблица переходов.
    if requested not in ALLOWED_STAGE_TRANSITIONS.get(from_stage, set()):
        verdict["reason"] = (
            f"Переход «{from_stage} → {requested}» запрещён. "
            f"Разрешено: {', '.join(allowed) or '—'}."
        )
        return verdict

    # 3. Гварды. Пользователь может выставить гвард этим же обновлением (approve),
    #    поэтому учитываем гварды из update для source=user/manual.
    guards = normalize_guards(task.get("guards"))
    if source in ("user", "manual"):
        up_guards = update.get("guards")
        if isinstance(up_guards, dict):
            for key in DEFAULT_GUARDS:
                if key in up_guards:
                    guards[key] = bool(up_guards[key])

    if requested != from_stage:
        missing = missing_guards_for(guards, requested)
        if missing:
            labels = ", ".join(GUARD_LABELS.get(g, g) for g in missing)
            verdict["reason"] = f"Переход в «{requested}» требует условия: {labels}."
            return verdict

    # --- Переход принят: вычисляем поля задачи. ---
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

    if requested == "done":
        status = "done"
        expected_action = "done"
    else:
        status = normalize_status(task.get("status"), "active")
        expected_action = normalize_expected_action(
            update.get("expected_action"), task.get("expected_action", "wait_user")
        )

    # Синхронизируем состояния шагов плана с фактическим прогрессом.
    if requested in ("validation", "done"):
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
        "stage": requested,
        "status": status,
        "step_index": step_index,
        "step_total": step_total,
        "step_label": step_label,
        "expected_action": expected_action,
        "plan": plan,
        "guards": guards,
        "resume_note": build_resume_note(requested, step_index, step_total, step_label),
    }
    transition = {
        "from_stage": from_stage,
        "to_stage": requested,
        "step_label": step_label,
        "expected_action": expected_action,
        "reason": str(update.get("reason") or ""),
    }
    verdict.update({
        "accepted": True,
        "reason": transition["reason"],
        "fields": fields,
        "transition": transition,
    })
    return verdict


_STEP_MARKERS = {"done": "[x]", "in_progress": "[>]", "blocked": "[!]", "pending": "[ ]"}


def render_task_block(task: Dict[str, Any]) -> str:
    """Системный блок с текущим состоянием задачи для контекста модели.

    Включает разрешённые переходы и состояние гвардов, чтобы модель знала, что
    ей доступно и какие переходы выполняет только пользователь.
    """
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

    guards = normalize_guards(task.get("guards"))
    lines.append(
        "- Гварды: "
        + f"{GUARD_LABELS['plan_approved']}={'да' if guards['plan_approved'] else 'нет'}, "
        + f"{GUARD_LABELS['validation_passed']}={'да' if guards['validation_passed'] else 'нет'}"
    )

    allowed = allowed_next_stages(task)
    blocked = blocked_next_stages(task)
    lines.append(f"- Разрешённые переходы: {', '.join(allowed) or '—'}")
    for b in blocked:
        labels = ", ".join(GUARD_LABELS.get(g, g) for g in b["missing_guards"])
        lines.append(f"  • этап «{b['stage']}» заблокирован: нужно {labels}")

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
