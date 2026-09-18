# main.py
import json
import logging
import math
import os
import re
import time
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from fastapi.middleware.cors import CORSMiddleware

from storage import storage, WORKING_KINDS, WORKING_STATES, LONG_TERM_KINDS, INVARIANT_CATEGORIES
from task_state import (
    apply_state_update,
    forward_stage_path,
    normalize_stage,
    render_task_block,
    render_resume_instruction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

app = FastAPI(title="AI Agent Service", description="Обработка запросов к DeepSeek")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация API DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1234567890")  # Заглушка — используем мок

# Цены DeepSeek (за 1M токенов) — для оценки стоимости.
INPUT_PRICE_PER_M = 0.14
OUTPUT_PRICE_PER_M = 0.28

# ------------------------------------------------------------
# Стратегии управления контекстом
# ------------------------------------------------------------
# sliding_window — окно последних N сообщений + вся память.
# sticky_facts    — факты (long_term с kind=fact) + рабочая память + окно N.
# branching       — полная история активной ветки + вся память.
STRATEGY_SLIDING_WINDOW = "sliding_window"
STRATEGY_STICKY_FACTS = "sticky_facts"
STRATEGY_BRANCHING = "branching"
STRATEGIES = {STRATEGY_SLIDING_WINDOW, STRATEGY_STICKY_FACTS, STRATEGY_BRANCHING}
DEFAULT_STRATEGY = STRATEGY_SLIDING_WINDOW
DEFAULT_WINDOW_SIZE = 10

# Параметры вспомогательного вызова (генерация предложений памяти).
SUGGEST_MAX_TOKENS = 512
SUGGEST_TEMPERATURE = 0.2

# Человекочитаемые названия категорий инвариантов.
INVARIANT_CATEGORY_LABELS = {
    "architecture": "архитектура",
    "decision": "техническое решение",
    "stack": "стек",
    "business_rule": "бизнес-правило",
    "custom": "прочее",
}

# Параметры отдельного LLM-вызова-классификатора, который проверяет запрос
# на соответствие инвариантам ДО основной генерации.
INVARIANT_CHECK_SYSTEM = (
    "Ты — строгий валидатор инвариантов. Решай ТОЛЬКО на основе текста инвариантов "
    "и запроса. Не смягчай правила и не придумывай исключений."
)
INVARIANT_CHECK_PROMPT = (
    "Ниже — нерушимые инварианты (правила) агента и запрос пользователя.\n"
    "Определи, нарушает ли запрос хотя бы один инвариант.\n"
    "Верни СТРОГО JSON-объект вида "
    "{\"violation\": <true|false>, \"invariant_name\": \"<имя нарушенного инварианта или пусто>\", "
    "\"reason\": \"<краткое объяснение или пусто>\"}.\n"
    "Если нарушений нет — violation=false, остальные поля пустые.\n"
    "Считай нарушением и случаи, когда запрос противоречит правилу косвенно.\n\n"
)
INVARIANT_CHECK_TEMPERATURE = 0.0
INVARIANT_CHECK_MAX_TOKENS = 256

SUGGEST_SYSTEM = "Ты — ассистент, который ведёт модель памяти агента."

SUGGEST_PROMPT = (
    "Проанализируй сообщение пользователя и предложи, что стоит сохранить в память. "
    "Верни СТРОГО JSON-объект вида {\"working\": [...], \"long_term\": [...]}.\n"
    "Имена полей элементов ДОЛЖНЫ быть на английском языке: \"key\", \"value\", \"kind\". "
    "Сами значения ключа и содержимого могут быть на русском.\n"
    "working — данные ТЕКУЩЕЙ задачи: поля key (ключ), value (значение), kind, state.\n"
    "  kind ∈ {goal, constraint, todo, result, context, note}; "
    "state ∈ {pending, in_progress, done, blocked} или null.\n"
    "long_term — профиль/решения/знания: поля key, value, kind, tags.\n"
    "  kind ∈ {profile, decision, knowledge, preference, agreement, fact}; "
    "tags — список строк.\n"
    "Объедини новое с уже известным. Не выдумывай лишнего. "
    "Если сохранять нечего — верни пустые списки."
)


# Модель для входных данных
class AgentRequest(BaseModel):
    messages: List[dict] = Field(..., description="История сообщений (role, content)")
    session_id: Optional[str] = Field(None, description="ID сессии для сохранения/восстановления контекста")
    strategy: str = Field(DEFAULT_STRATEGY, description="Стратегия управления контекстом")
    window_size: int = Field(DEFAULT_WINDOW_SIZE, ge=1, description="Число последних сообщений в окне (N)")
    model: str = Field(..., description="Название модели (например, deepseek-chat)")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(0, ge=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192)
    # Явные операции памяти: [{action: save|delete|move, layer, key, value, ...}]
    memory_ops: List[dict] = Field([], description="Явные операции над памятью (save/delete/move)")
    # Включить генерацию предложений памяти (не сохраняются автоматически — только на подтверждение)
    auto_suggest_memory: bool = Field(False, description="Генерировать предложения памяти после запроса")
    # Профиль пользователя (персонализация). Если не задан — берётся активный профиль.
    profile_id: Optional[str] = Field(None, description="ID профиля пользователя для этого запроса")
    # Задача (конечный автомат). Если task_id не задан — используется активная
    # задача сессии; задача создаётся моделью (tool call), только если запрос
    # действительно многошаговый.
    task_id: Optional[str] = Field(None, description="ID задачи, к которой относится запрос")


class BranchRequest(BaseModel):
    checkpoint: Optional[int] = Field(None, description="Индекс сообщения, от которого ветвимся")


class SwitchRequest(BaseModel):
    branch_id: str = Field(..., description="ID ветки, на которую переключаемся")


class MemoryOpsRequest(BaseModel):
    ops: List[dict] = Field(..., description="Список операций памяти (save/delete/move)")


class SuggestRequest(BaseModel):
    message: Optional[str] = Field(None, description="Сообщение для анализа (по умолчанию — последнее от пользователя)")
    model: str = Field("deepseek-chat", description="Модель для генерации предложений")


class ProfileRequest(BaseModel):
    """Профиль пользователя — персонализация (стиль, формат, ограничения)."""
    profile_id: Optional[str] = Field(None, description="ID профиля (если обновляем существующий)")
    name: str = Field(..., description="Уникальный ключ профиля (напр. ivan_dev)")
    display_name: Optional[str] = Field("", description="Отображаемое имя пользователя")
    style: Optional[str] = Field("", description="Стиль общения")
    format: Optional[str] = Field("", description="Формат ответа")
    constraints: Optional[str] = Field("", description="Ограничения")
    notes: Optional[str] = Field("", description="Дополнительные пожелания")
    is_active: Optional[bool] = Field(False, description="Сделать профиль активным после сохранения")


class SessionProfileRequest(BaseModel):
    """Привязка профиля к сессии (None — отвязать)."""
    profile_id: Optional[str] = Field(None, description="ID профиля; null/None — отвязать профиль от сессии")


class TaskCreateRequest(BaseModel):
    """Создание задачи в сессии."""
    title: Optional[str] = Field("", description="Название задачи")


class TaskTransitionRequest(BaseModel):
    """Ручной переход конечного автомата (для отладки/UI)."""
    stage: str = Field(..., description="Целевой этап")
    expected_action: Optional[str] = Field("wait_user", description="Ожидаемое действие")
    step_index: Optional[int] = Field(None, description="Текущий шаг")
    step_total: Optional[int] = Field(None, description="Всего шагов")
    step_label: Optional[str] = Field(None, description="Название текущего шага")
    plan: Optional[List[dict]] = Field(None, description="План шагов")
    reason: Optional[str] = Field("", description="Причина перехода")


class InvariantRequest(BaseModel):
    """Инвариант — жёсткое ограничение, которое агент не вправе нарушать.

    Скоуп задаётся тремя полями profile_id / session_id / task_id; пустое поле
    означает «без привязки к этому измерению». Все три пустые — глобальный инвариант.
    """
    invariant_id: Optional[str] = Field(None, description="ID инварианта (при обновлении)")
    profile_id: Optional[str] = Field("", description="Скоуп по профилю (пусто — без привязки)")
    session_id: Optional[str] = Field("", description="Скоуп по сессии (пусто — без привязки)")
    task_id: Optional[str] = Field("", description="Скоуп по задаче (пусто — без привязки)")
    category: str = Field("business_rule", description="Категория: architecture|decision|stack|business_rule|custom")
    name: str = Field(..., description="Короткое имя правила")
    statement: str = Field(..., description="Формулировка инварианта")
    rationale: Optional[str] = Field("", description="Почему это нельзя нарушать")
    deny_examples: Optional[List[str]] = Field([], description="Примеры нарушений (для мок-проверки и подсказки)")
    is_active: Optional[bool] = Field(True, description="Активен ли инвариант")


# ============================================================
# Подсчёт токенов
# ============================================================
_CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
    r"\u3040-\u30FF\uAC00-\uD7AF]"
)
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Приблизительный подсчёт количества токенов в строке."""
    if not text:
        return 0
    text = str(text)
    cjk = len(_CJK_RE.findall(text))
    without_cjk = _CJK_RE.sub(" ", text)
    words = len(_WORD_RE.findall(without_cjk))
    punct = len(_PUNCT_RE.findall(without_cjk))
    return cjk + math.ceil(words * 1.3) + punct


def estimate_messages_tokens(messages) -> int:
    """Приблизительный подсчёт токенов по списку сообщений диалога."""
    total = 0
    for message in messages or []:
        total += estimate_tokens(message.get("content", ""))
    return total


def call_deepseek_raw(messages, model, temperature=1.0, top_p=1.0, max_tokens=4096, top_k=0, stop=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop is not None:
        payload["stop"] = stop
    if top_k:
        payload["top_k"] = top_k

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
    return response.json()


def call_deepseek(messages, request: AgentRequest):
    return call_deepseek_raw(
        messages,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
        stop=request.stop,
    )


# ============================================================
# Классификация запроса и детерминированный переход автомата
# ============================================================
# Глаголы-императивы, указывающие на многошаговую задачу (детерминированный
# классификатор — одинаково работает и в моке, и с реальным API).
TASK_VERBS = (
    "напиши", "напишите", "сделай", "сделайте", "реализуй", "реализуйте",
    "создай", "создайте", "разработай", "разработайте", "исправь", "исправьте",
    "настрой", "настройте", "добавь", "добавьте", "сгенерируй", "сгенерируйте",
    "построй", "постройте", "перенеси", "мигрируй", "проверь", "протестируй",
    "составь", "составьте", "придумай", "придумайте", "подготовь", "подготовьте",
    "почини", "оптимизируй", "перепиши", "отрефактори",
)

# Слова-сигналы «продолжить текущую задачу».
CONTINUE_WORDS = ("продолжай", "продолжи", "дальше", "continue", "next", "ок", "да", "го")

# Вопросительные слова — признак, что пользователь задаёт НОВЫЙ вопрос,
# а не отвечает на вопрос агента (тогда автомат двигать не нужно).
QUESTION_WORDS = (
    "кто", "что", "почему", "зачем", "как", "какой", "какая", "какие", "какое",
    "когда", "где", "сколько", "чей", "чья", "чьё", "чем", "кому", "кого",
    "откуда", "куда", "ли",
)


def _detect_task(user_message: str) -> bool:
    """Эвристика «это задача, а не вопрос» по глаголам-императивам."""
    t = (user_message or "").strip().lower()
    if not t:
        return False
    for verb in TASK_VERBS:
        if re.search(r"\b" + re.escape(verb) + r"\b", t):
            return True
    return False


def _is_continue(user_message: str) -> bool:
    t = (user_message or "").strip().lower().rstrip("!.? ")
    return t in CONTINUE_WORDS or "продолж" in t or "дальше" in t


def _is_question(user_message: str) -> bool:
    """True, если сообщение похоже на новый вопрос (а не ответ агенту)."""
    t = (user_message or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first_word = (t.split() or [""])[0]
    return first_word in QUESTION_WORDS


def _mock_title(user_message: str) -> str:
    t = re.sub(r"\s+", " ", (user_message or "").strip())
    return t[:60] if t else "Задача"


def _mock_plan(user_message: str) -> List[dict]:
    """Наивный план шагов для мока (без реального API)."""
    t = (user_message or "").lower()
    if any(k in t for k in ("калькулятор", "calc")):
        labels = [
            "Спроектировать интерфейс калькулятора",
            "Написать код арифметики",
            "Проверить на тестовых примерах",
        ]
    elif any(k in t for k in ("тест", "test")):
        labels = [
            "Определить сценарии тестов",
            "Написать тесты",
            "Прогнать и зафиксировать результат",
        ]
    else:
        labels = [
            "Уточнить требования и ограничения",
            "Выполнить основную работу",
            "Проверить результат",
        ]
    return [{"index": i + 1, "label": label, "state": "pending"} for i, label in enumerate(labels)]


def _task_state_update(task: Optional[dict], user_message: str) -> Optional[dict]:
    """Детерминированный переход конечного автомата.

    Используется И в моке, И с реальным API — состояние задачи полностью формально
    и не зависит от «настроения» модели. Возвращает словарь-обновление или None
    (задача завершена, двигать некуда).
    """
    if task is None:
        if not _detect_task(user_message):
            return None  # «продолжай» без задачи — ничего не заводим
        title = _mock_title(user_message)
        plan = _mock_plan(user_message)
        plan[0]["state"] = "in_progress"
        return {
            "title": title,
            "stage": "execution",
            "step_index": 1,
            "step_total": len(plan),
            "step_label": plan[0]["label"],
            "expected_action": "wait_user",
            "plan": plan,
            "reason": "Обнаружена многошаговая задача, план готов",
        }

    stage = task["stage"]
    if stage == "planning":
        plan = _mock_plan(user_message)
        plan[0]["state"] = "in_progress"
        return {
            "stage": "execution",
            "step_index": 1,
            "step_total": len(plan),
            "step_label": plan[0]["label"],
            "expected_action": "wait_user",
            "plan": plan,
            "reason": "План готов, начинаю выполнение",
        }

    if stage == "execution":
        idx = task["step_index"]
        total = task["step_total"] or len(task["plan"])
        plan = [dict(s) for s in task["plan"]]
        for s in plan:
            if s["index"] == idx:
                s["state"] = "done"
        if idx >= total:
            return {
                "stage": "validation",
                "step_index": total,
                "step_total": total,
                "step_label": "проверка результата",
                "expected_action": "wait_user",
                "plan": plan,
                "reason": "Выполнение завершено, проверяю результат",
            }
        nxt = idx + 1
        for s in plan:
            if s["index"] == nxt:
                s["state"] = "in_progress"
        return {
            "stage": "execution",
            "step_index": nxt,
            "step_total": total,
            "step_label": plan[nxt - 1]["label"],
            "expected_action": "wait_user",
            "plan": plan,
            "reason": f"Шаг {idx} готов, перехожу к шагу {nxt}",
        }

    if stage == "validation":
        return {
            "stage": "done",
            "step_index": task["step_total"],
            "step_total": task["step_total"],
            "step_label": "завершено",
            "expected_action": "done",
            "plan": task["plan"],
            "reason": "Результат проверен, задача готова",
        }

    return None  # done


def _mock_reply(task: Optional[dict], user_message: str, was_paused: bool = False) -> str:
    """Описательный ответ мока по текущему (уже обновлённому) состоянию задачи."""
    if task is None:
        if _is_continue(user_message):
            return "Эхо (mock). Нет активной задачи — опишите, что нужно сделать."
        return f"Эхо (mock): {user_message}"
    if was_paused:
        return (
            f"Эхо (mock). Задача «{task['title']}» возобновлена. "
            f"Продолжаю без повторных объяснений: этап {task['stage']}, "
            f"шаг {task['step_index']}/{task['step_total']} ({task['step_label']})."
        )
    stage = task["stage"]
    if stage == "done":
        return "Эхо (mock). Валидация пройдена. Задача завершена (этап done)."
    if stage == "validation":
        return "Эхо (mock). Все шаги выполнены. Перехожу к валидации (этап validation)."
    if stage == "execution":
        idx = task["step_index"]
        total = task["step_total"]
        label = task["step_label"]
        plan = task["plan"] or []
        if idx == 1:
            plan_text = "\n".join(f"{s['index']}. {s['label']}" for s in plan)
            return (
                f"Эхо (mock). План задачи «{task['title']}»:\n{plan_text}\n\n"
                f"Начинаю шаг 1/{total}: {label}."
            )
        return f"Эхо (mock). Выполняю шаг {idx}/{total}: {label}."
    return f"Эхо (mock). Этап {stage}."


# Скрытый маркер прогресса: модель (реальный API) сообщает фактическое состояние.
STATE_MARKER_INSTRUCTION = (
    "В конце ответа, отдельной строкой, укажи фактический прогресс задачи строго в формате:\n"
    "[STATE] {\"stage\": \"<planning|execution|validation|done>\", "
    "\"step_index\": <номер шага>, \"step_total\": <всего шагов>, "
    "\"expected_action\": \"<model_call|wait_user|confirm|done>\"}\n"
    "Это технический маркер (не показывай его как часть ответа пользователю). "
    "stage=done ставь только если задача полностью выполнена. Если за этот ответ "
    "пройдено несколько шагов — укажи итоговый step_index."
)


def _extract_state_marker(content: str) -> tuple:
    """Вынимает строку с маркером [STATE] {...} из ответа модели.

    Возвращает (update, clean_content): update — словарь прогресса или None,
    clean_content — ответ без строки-маркера.
    """
    if not content:
        return None, content
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if "[STATE]" not in line:
            continue
        update = None
        json_part = line.split("[STATE]", 1)[1].strip()
        try:
            parsed = json.loads(json_part)
            if isinstance(parsed, dict):
                update = parsed
        except Exception:
            update = None
        del lines[i]
        return update, "\n".join(lines).strip()
    return None, content


def _apply_task_transition(agent_id: str, task: Optional[dict], task_update: Optional[dict]) -> Optional[dict]:
    """Применяет переход конечного автомата (создаёт задачу, если нужно).

    Переходы идут СТРОГО по таблице разрешённых переходов: если модель сообщает
    этап на несколько шагов вперёд (execution → done), цепочка выполняется
    поэтапно через промежуточные (execution → validation → done), без перескока.
    """
    if not task_update:
        return task
    if task is None:
        if not task_update.get("title") and not task_update.get("plan"):
            return task
        task = storage.create_task(agent_id, title=str(task_update.get("title") or ""))

    target_stage = normalize_stage(task_update.get("stage"), task["stage"])
    path = forward_stage_path(task["stage"], target_stage)

    result = task
    if not path:
        # Один легальный переход (или шаг внутри этапа).
        fields, transition = apply_state_update(result, task_update)
        return storage.update_task(
            result["task_id"], fields, source="model", reason=transition["reason"]
        )

    # Цепочка переходов без перескока этапов.
    for stage in path:
        hop = dict(task_update)
        hop["stage"] = stage
        if stage == "validation" and not hop.get("step_label"):
            hop["step_label"] = "проверка результата"
        fields, transition = apply_state_update(result, hop)
        result = storage.update_task(
            result["task_id"], fields, source="model", reason=transition["reason"]
        )
    return result


def _require_task(session_id: str, task_id: str) -> dict:
    task = storage.get_task(task_id)
    if task is None or task["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


def _resolve_task(session_id: str, task_id: Optional[str]) -> Optional[dict]:
    """Определяет задачу для запроса: явная → активная → None (задачу НЕ создаём
    заранее — она заводится моделью только когда запрос действительно многошаговый)."""
    if task_id:
        return _require_task(session_id, task_id)
    return storage.get_active_task(session_id)


# ============================================================
# Вспомогательные функции для подсчёта стоимости
# ============================================================
def _call_tokens(input_text, output_text, usage=None):
    usage = usage or {}
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    if inp is None:
        inp = estimate_tokens(input_text)
    if out is None:
        out = estimate_tokens(output_text)
    return inp, out


def _tokens_money(input_tokens, output_tokens) -> float:
    return (input_tokens / 1_000_000) * INPUT_PRICE_PER_M + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M


def parse_json_object(text: str) -> Optional[dict]:
    """Пытается распарсить JSON-объект из ответа модели."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


# ============================================================
# Генерация предложений памяти
# ============================================================
def _kind_or(kind, allowed, default):
    k = (kind or "").strip().lower()
    return k if k in allowed else default


def _state_or(state):
    s = (state or "").strip().lower()
    return s if s in WORKING_STATES else None


def normalize_suggestion_list(items, layer) -> List[dict]:
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        entry = {"key": key, "value": str(item.get("value") or "")}
        if layer == "working":
            entry["kind"] = _kind_or(item.get("kind"), WORKING_KINDS, "note")
            entry["state"] = _state_or(item.get("state"))
        else:
            entry["kind"] = _kind_or(item.get("kind"), LONG_TERM_KINDS, "fact")
            tags = item.get("tags")
            entry["tags"] = list(tags) if isinstance(tags, list) else []
        out.append(entry)
    return out


def flatten_suggestions(suggestion) -> List[dict]:
    """Превращает {working, long_term} в плоский список с полем layer."""
    pending = []
    for c in suggestion.get("working") or []:
        c = dict(c)
        c["layer"] = "working"
        pending.append(c)
    for c in suggestion.get("long_term") or []:
        c = dict(c)
        c["layer"] = "long_term"
        pending.append(c)
    return pending


def mock_suggest_memory(working, long_term, user_message) -> dict:
    """Наивное извлечение предложений памяти для мока (без реального API)."""
    result = {"working": [], "long_term": []}
    t = (user_message or "").lower()
    if any(k in t for k in ("цель", "задача", "нужно", "хочу", "сделай", "напиши")):
        result["working"].append({
            "key": "цель", "value": user_message, "kind": "goal", "state": "in_progress",
        })
    if any(k in t for k in ("нельзя", "огранич", "лимит", "без ", "не ")):
        result["working"].append({
            "key": "ограничения", "value": user_message, "kind": "constraint", "state": None,
        })
    if any(k in t for k in ("меня зовут", "моё имя", "мое имя", "представь")):
        result["long_term"].append({
            "key": "имя_пользователя", "value": user_message, "kind": "profile", "tags": [],
        })
    if any(k in t for k in ("предпочит", "нравится", "люблю", "лучше")):
        result["long_term"].append({
            "key": "предпочтения", "value": user_message, "kind": "preference", "tags": [],
        })
    if any(k in t for k in ("решил", "решение", "договорились", "запомни", "зафиксируй")):
        result["long_term"].append({
            "key": "решение", "value": user_message, "kind": "decision", "tags": [],
        })
    return result


def suggest_memory(working, long_term, user_message, model):
    """Генерирует предложения памяти: {working: [...], long_term: [...]}.

    Возвращает (suggestion, input_tokens, output_tokens). НИЧЕГО не сохраняет —
    пользователь явно подтверждает предложения.
    """
    if API_KEY == "sk-1234567890":
        result = mock_suggest_memory(working, long_term, user_message)
        return result, estimate_tokens(user_message), estimate_tokens(json.dumps(result, ensure_ascii=False))

    prompt_text = (
        SUGGEST_PROMPT
        + "\n\nТекущая рабочая память (JSON):\n"
        + json.dumps(working, ensure_ascii=False)
        + "\n\nДолговременная память (JSON):\n"
        + json.dumps(long_term, ensure_ascii=False)
        + "\n\nСообщение пользователя:\n"
        + (user_message or "")
    )
    msgs = [
        {"role": "system", "content": SUGGEST_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    data = call_deepseek_raw(msgs, model=model, temperature=SUGGEST_TEMPERATURE, max_tokens=SUGGEST_MAX_TOKENS)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = parse_json_object(content) or {}
    result = {
        "working": normalize_suggestion_list(parsed.get("working"), "working"),
        "long_term": normalize_suggestion_list(parsed.get("long_term"), "long_term"),
    }
    inp, out = _call_tokens(prompt_text, content, data.get("usage"))
    return result, inp, out


# ============================================================
# Формирование контекста из трёх слоёв памяти
# ============================================================
def memory_entries_to_text(entries, layer, max_entries=40) -> str:
    """Превращает список записей памяти в текстовый блок для модели."""
    if not entries:
        return ""
    total = len(entries)
    entries = sorted(entries, key=lambda e: (e.get("importance") or 0), reverse=True)
    truncated = total > max_entries
    entries = entries[:max_entries]

    if layer == "long_term":
        header = "Долговременная память (профиль, решения, знания):"
        lines = [f"- [{e.get('kind', 'fact')}] {e['key']}: {e['value']}" for e in entries]
    else:
        header = "Рабочая память (текущая задача):"
        lines = []
        for e in entries:
            state = f" · состояние: {e['state']}" if e.get("state") else ""
            lines.append(f"- [{e.get('kind', 'note')}] {e['key']}: {e['value']}{state}")
    if truncated:
        lines.append(f"- ... (ещё {total - max_entries} записей опущено)")
    return header + "\n" + "\n".join(lines)


def build_memory_system_block(working, long_term, long_term_kinds=None) -> str:
    """Собирает системный блок из рабочей и долговременной памяти."""
    lt_entries = long_term
    if long_term_kinds is not None:
        lt_entries = [e for e in long_term if e.get("kind") in long_term_kinds]
    parts = []
    lt_text = memory_entries_to_text(lt_entries, "long_term")
    if lt_text:
        parts.append(lt_text)
    wm_text = memory_entries_to_text(working, "working")
    if wm_text:
        parts.append(wm_text)
    return "\n\n".join(parts)


def build_profile_system_block(profile) -> str:
    """Собирает системный блок с профилем пользователя (персонализация).

    Этот блок подключается к КАЖДОМУ запросу к модели и стоит ПЕРВЫМ,
    поэтому предпочтения пользователя применяются автоматически.
    """
    if not profile:
        return ""
    lines = ["Профиль пользователя — ВСЕГДА следуй ему в каждом ответе:"]
    if profile.get("display_name"):
        lines.append(f"- Кто: {profile['display_name']}")
    if profile.get("style"):
        lines.append(f"- Стиль общения: {profile['style']}")
    if profile.get("format"):
        lines.append(f"- Формат ответа: {profile['format']}")
    if profile.get("constraints"):
        lines.append(f"- Ограничения: {profile['constraints']}")
    if profile.get("notes"):
        lines.append(f"- Дополнительно: {profile['notes']}")
    return "\n".join(lines)


def invariant_scope(inv) -> str:
    """Человекочитаемый скоуп инварианта по заполненным полям привязки."""
    parts = []
    if (inv.get("profile_id") or "").strip():
        parts.append("profile")
    if (inv.get("session_id") or "").strip():
        parts.append("session")
    if (inv.get("task_id") or "").strip():
        parts.append("task")
    return "+".join(parts) if parts else "global"


def build_invariants_system_block(invariants) -> str:
    """Системный блок с инвариантами — ЖЁСТКИМИ правилами для модели.

    В отличие от профиля (персонализация) и памяти (факты/решения), этот блок
    прямо требует от модели проверять ответ на соответствие и отказываться при
    нарушении. Не обрезается по importance.
    """
    if not invariants:
        return ""
    lines = [
        "ИНВАРИАНТЫ (нерушимые ограничения — нарушать их НЕЛЬЗЯ):",
        "Это не пожелания и не советы, а жёсткие правила. Проверяй КАЖДЫЙ ответ "
        "на соответствие каждому инварианту.",
        "Если запрос или предлагаемое решение нарушает хотя бы один инвариант — "
        "откажись и объясни, какой именно инвариант и почему он не может быть нарушен.",
    ]
    for inv in invariants:
        scope = invariant_scope(inv)
        cat = INVARIANT_CATEGORY_LABELS.get(inv.get("category"), inv.get("category"))
        lines.append(f"- [{scope} · {cat}] {inv.get('name')}: {inv.get('statement')}")
        if (inv.get("rationale") or "").strip():
            lines.append(f"    Почему нельзя: {inv.get('rationale')}")
    return "\n".join(lines)


def _find_invariant_by_name(invariants, name) -> Optional[dict]:
    name = (name or "").strip().lower()
    if not name:
        return None
    for inv in invariants:
        if (inv.get("name") or "").strip().lower() == name:
            return inv
    return None


def mock_check_invariants(invariants, user_text) -> dict:
    """Детерминированная заглушка проверки инвариантов (для мока, где нет LLM).

    Сверяет запрос с явными примерами нарушений (deny_examples), которые человек
    задаёт при создании инварианта. Это НЕ «поиск по ключевым словам» как основной
    механизм, а демо-стаб: в реальном режиме детекцию выполняет отдельный LLM-вызов.
    """
    text = (user_text or "").lower()
    for inv in invariants:
        for token in inv.get("deny_examples") or []:
            token = str(token).strip().lower()
            if not token:
                continue
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                return {
                    "violation": True,
                    "invariant_name": inv.get("name"),
                    "reason": f"запрос содержит запрещённое для этого инварианта: «{token}»",
                }
    return {"violation": False, "invariant_name": None, "reason": ""}


def check_invariants(invariants, user_text, model) -> dict:
    """Отдельный LLM-вызов: нарушает ли запрос хотя бы один инвариант.

    Возвращает вердикт {violation, invariant_name, reason, invariant?}. Ничего не
    сохраняет. Ключевых слов не использует — решение принимает модель-классификатор.
    """
    if not invariants:
        return {"violation": False, "invariant_name": None, "reason": ""}
    if API_KEY == "sk-1234567890":
        return mock_check_invariants(invariants, user_text)

    invariants_text = "\n".join(
        f"- {inv.get('name')}: {inv.get('statement')}"
        + (f" (почему нельзя: {inv.get('rationale')})" if (inv.get('rationale') or "").strip() else "")
        for inv in invariants
    )
    prompt_text = (
        INVARIANT_CHECK_PROMPT
        + "Инварианты:\n" + invariants_text
        + "\n\nЗапрос пользователя:\n" + (user_text or "")
    )
    msgs = [
        {"role": "system", "content": INVARIANT_CHECK_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    data = call_deepseek_raw(
        msgs, model=model,
        temperature=INVARIANT_CHECK_TEMPERATURE,
        max_tokens=INVARIANT_CHECK_MAX_TOKENS,
    )
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = parse_json_object(content) or {}
    if not parsed.get("violation"):
        return {"violation": False, "invariant_name": None, "reason": ""}
    name = str(parsed.get("invariant_name") or "").strip()
    return {
        "violation": True,
        "invariant_name": name,
        "invariant": _find_invariant_by_name(invariants, name),
        "reason": str(parsed.get("reason") or "").strip(),
    }


def build_refusal_reply(inv, verdict) -> str:
    """Детерминированный текст отказа при нарушении инварианта."""
    name = (verdict.get("invariant_name") or "").strip() or (inv.get("name") if inv else "")
    category = INVARIANT_CATEGORY_LABELS.get(inv.get("category"), inv.get("category")) if inv else ""
    statement = inv.get("statement") if inv else ""
    reason = (verdict.get("reason") or "").strip() or (inv.get("rationale") if inv else "") or ""
    lines = [f"⛔ Отказываюсь выполнять запрос: он нарушает инвариант «{name}»."]
    if category:
        lines.append(f"Категория: {category}.")
    if statement:
        lines.append(f"Правило: {statement}")
    if reason:
        lines.append(f"Почему это нельзя нарушать: {reason}")
    return "\n".join(lines)


def build_request_messages(strategy, conversation, working, long_term, window_size, profile=None, task=None, resume_instruction=None, report_state=False, invariants=None):
    """Собирает сообщения модели из инвариантов, профиля, задачи и трёх слоёв памяти.

    • инварианты       — системный блок жёстких ограничений (всегда, если заданы);
    • профиль          — системный блок (всегда, если задан);
    • состояние задачи — блок конечного автомата (если есть активная задача);
    • маркер прогресса — инструкция [STATE] (report_state=True, реальный API);
    • краткосрочная    — окно последних N (или полная ветка для branching);
    • рабочая          — системный блок (всегда);
    • долговременная   — системный блок (для sticky_facts — только факты kind=fact).
    """
    if strategy == STRATEGY_STICKY_FACTS:
        block = build_memory_system_block(working, long_term, long_term_kinds={"fact"})
    else:
        block = build_memory_system_block(working, long_term)

    messages = []
    invariants_block = build_invariants_system_block(invariants)
    if invariants_block:
        messages.append({"role": "system", "content": invariants_block})
    profile_block = build_profile_system_block(profile)
    if profile_block:
        messages.append({"role": "system", "content": profile_block})
    if task:
        task_block = render_task_block(task)
        if task_block:
            messages.append({"role": "system", "content": task_block})
        if resume_instruction:
            messages.append({"role": "system", "content": resume_instruction})
        if report_state:
            messages.append({"role": "system", "content": STATE_MARKER_INSTRUCTION})
    if block:
        messages.append({"role": "system", "content": block})
    if strategy == STRATEGY_BRANCHING:
        messages.extend(list(conversation))
    else:
        messages.extend(conversation[-window_size:])
    return messages


def build_context_summary(strategy, conversation, working, long_term, window_size, session_id):
    """Сводка по контексту и памяти для UI."""
    total = len(conversation)
    info = {
        "strategy": strategy,
        "total_messages": total,
        "window_size": window_size,
        "working_count": len(working),
        "long_term_count": len(long_term),
    }
    if strategy == STRATEGY_BRANCHING:
        current = storage.get_current_branch_id(session_id)
        info["branch"] = {"id": current}
        info["branches"] = storage.list_branches(session_id)
        info["sent_messages"] = total
        info["discarded_messages"] = 0
    else:
        info["sent_messages"] = min(total, window_size)
        info["discarded_messages"] = max(0, total - window_size)
    return info


# ============================================================
# Основной обработчик
# ============================================================
@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    agent_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    strategy = request.strategy if request.strategy in STRATEGIES else DEFAULT_STRATEGY
    window_size = request.window_size if request.window_size >= 1 else DEFAULT_WINDOW_SIZE

    try:
        # 1. Последнее сообщение пользователя — единственный новый ввод.
        current_user_message = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            None,
        )
        user_text = current_user_message.get("content", "") if current_user_message else ""

        # Профиль пользователя (персонализация). Порядок разрешения:
        # явный из запроса > привязанный к сессии > глобальный активный.
        profile = None
        if request.profile_id:
            profile = storage.get_profile(request.profile_id)
            if profile is None:
                raise HTTPException(status_code=400, detail=f"Профиль не найден: {request.profile_id}")
        else:
            bound_profile_id = storage.get_session_profile(agent_id)
            profile = storage.get_profile(bound_profile_id) if bound_profile_id else None
            if profile is None:
                profile = storage.get_active_profile()

        # Рабочая память в разрезе профиля (у каждого профиля свой список задач),
        # долговременная — общие записи + привязанные к профилю.
        working_profile_id = profile["profile_id"] if profile else ""

        # 2. Сессия: только метаданные (стратегия/профиль). Сообщения хранятся
        #    отдельно: у задачи — в самой задаче, у обычного чата — в сессии.
        if storage.get_session_meta(agent_id) is None:
            storage.create_session(
                agent_id, strategy=strategy, window_size=window_size,
                messages=[],
                profile_id=profile["profile_id"] if profile else None,
            )
        else:
            storage.set_session_meta(agent_id, strategy, window_size)

        # Привязка профиля к сессии — намеренное действие: только при явном profile_id
        # в запросе (новая сессия уже получила профиль при create_session). Простое
        # наследование глобального активного профиля привязку НЕ перезаписывает.
        if request.profile_id and profile is not None:
            storage.set_session_profile(agent_id, profile["profile_id"])

        # 2. Явные операции памяти — пользователь выбирает, что и куда сохранять.
        memory_ops_applied = []
        for op in request.memory_ops or []:
            try:
                memory_ops_applied.append(storage.apply_memory_op(agent_id, op, source="manual"))
            except Exception as e:
                logger.warning("Ошибка применения memory_op %s: %s", op, e)
        working = storage.load_working(agent_id, working_profile_id)
        long_term = storage.load_long_term_for_profile(profile["profile_id"] if profile else None)

        # 2.5. Задача (конечный автомат). Классификация запроса:
        #      • новая инструкция («напиши/сделай/…») → ВСЕГДА новая задача,
        #        переданный task_id игнорируется, текущая активная уходит в паузу;
        #      • «продолжай/дальше» → продолжить целевую задачу (резюм, если на паузе);
        #      • ответ на вопрос агента (когда задача ждёт ввода: wait_user/confirm)
        #        → продвинуть задачу, даже если это короткий ответ «2», «первый», «да»;
        #      • вопрос/общение → ответить, автомат не двигать и задачу не создавать.
        is_new_task = _detect_task(user_text)
        is_continue = _is_continue(user_text)

        if is_new_task:
            task = None
            was_paused = False
            resume_instruction = None
            mode = "task"
        else:
            task = _resolve_task(agent_id, request.task_id)
            was_paused = False
            resume_instruction = None
            awaiting = (
                task is not None
                and task["status"] == "active"
                and task.get("expected_action") in ("wait_user", "confirm")
            )
            if is_continue and task is not None:
                was_paused = task["status"] == "paused"
                if was_paused:
                    # Продолжаем с того же места без повторных объяснений.
                    task = storage.resume_task(task["task_id"])
                resume_instruction = render_resume_instruction(task) if was_paused else None
                mode = "task"
            elif awaiting and not _is_question(user_text):
                # Пользователь ответил на уточняющий вопрос агента — двигаем задачу.
                mode = "task"
            else:
                mode = "chat"

        # 2.55. Инварианты (жёсткие ограничения). Проверяем запрос ДО создания
        #        задачи и ДО основной генерации отдельным LLM-вызовом. При
        #        нарушении — детерминированный отказ: задачу не создаём и
        #        конечный автомат не двигаем.
        task_id_for_scope = task["task_id"] if task else ""
        invariants = storage.get_applicable_invariants(
            profile["profile_id"] if profile else None, agent_id, task_id_for_scope
        )
        refusal_inv = None
        refusal_reason = ""
        if invariants:
            try:
                verdict = check_invariants(invariants, user_text, request.model)
            except Exception as e:
                logger.warning("Ошибка проверки инвариантов: %s", e)
                verdict = {"violation": False, "invariant_name": None, "reason": ""}
            if verdict.get("violation"):
                refusal_inv = verdict.get("invariant") or _find_invariant_by_name(
                    invariants, verdict.get("invariant_name")
                )
                refusal_reason = (verdict.get("reason") or "").strip()
        refused = refusal_inv is not None
        violated_invariant = None
        if refused:
            task = None
            mode = "chat"
            violated_invariant = {
                "invariant_id": refusal_inv.get("invariant_id"),
                "name": refusal_inv.get("name"),
                "category": refusal_inv.get("category"),
                "scope": invariant_scope(refusal_inv),
                "statement": refusal_inv.get("statement"),
                "reason": refusal_reason or (refusal_inv.get("rationale") or ""),
            }

        # 2.6. Для НОВОЙ задачи заранее заводим оболочку с планом, чтобы модель
        #      видела план в контексте и следовала ему.
        is_mock = API_KEY == "sk-1234567890"
        task_was_none = mode == "task" and task is None
        if task_was_none:
            task = _apply_task_transition(agent_id, None, _task_state_update(None, user_text))

        # 2.7. Контекст диалога. У каждой задачи СВОЯ история сообщений (изоляция:
        #      при переключении задач модель не видит чужие сообщения), обычный чат
        #      без задачи идёт в историю сессии.
        if task is not None:
            conversation = storage.load_task_messages(task["task_id"]) or []
        else:
            conversation = storage.load(agent_id) or []
        if current_user_message and (not conversation or conversation[-1] != current_user_message):
            conversation.append(current_user_message)

        # 3. Предложения памяти (опционально). Ничего не сохраняется автоматически.
        suggest_cost = 0.0
        pending_memory = []
        if request.auto_suggest_memory and current_user_message and not refused:
            try:
                suggestion, s_inp, s_out = suggest_memory(
                    working, long_term, current_user_message.get("content", ""), request.model
                )
                pending_memory = flatten_suggestions(suggestion)
                suggest_cost += _tokens_money(s_inp, s_out)
            except Exception as e:
                logger.warning("Ошибка генерации предложений памяти: %s", e)

        # 4. Формируем контекст из профиля, состояния задачи и трёх слоёв памяти.
        request_messages = build_request_messages(
            strategy, conversation, working, long_term, window_size,
            profile=profile, task=task, resume_instruction=resume_instruction,
            report_state=(mode == "task" and not is_mock),
            invariants=invariants,
        )

        raw_context_tokens = estimate_messages_tokens(conversation)
        context_tokens = estimate_messages_tokens(request_messages)

        # 5. Вызов модели + переход конечного автомата.
        #    • отказ по инварианту — детерминированный ответ, основная модель НЕ
        #      вызывается и автомат не двигается;
        #    • мок — детерминированный переход (код) + описательный ответ;
        #    • реальный API — модель сама сообщает фактический прогресс маркером
        #      [STATE] {...} (его вырезаем из ответа); без маркера — детерминированный
        #      фолбэк. Это держит состояние в синхроне с реальной работой модели.
        if refused:
            content = build_refusal_reply(refusal_inv, {
                "invariant_name": refusal_inv.get("name"),
                "reason": refusal_reason or refusal_inv.get("rationale"),
            })
            prompt_tokens = estimate_messages_tokens(request_messages)
            completion_tokens = estimate_tokens(content)
            data = {
                "id": f"refusal-{uuid.uuid4()}",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        elif is_mock:
            if mode == "task":
                if not task_was_none:
                    task = _apply_task_transition(
                        agent_id, task, _task_state_update(task, user_text)
                    )
                content = _mock_reply(task, user_text, was_paused)
            else:
                content = f"Эхо (mock): {user_text}"
            prompt_tokens = estimate_messages_tokens(request_messages)
            completion_tokens = estimate_tokens(content)
            data = {
                "id": f"mock-{uuid.uuid4()}",
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        else:
            data = call_deepseek(request_messages, request)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if mode == "task":
                marker_update, content = _extract_state_marker(content)
                if marker_update is not None:
                    task = _apply_task_transition(agent_id, task, marker_update)
                elif not task_was_none:
                    # Маркера нет — детерминированный фолбэк на один шаг.
                    task = _apply_task_transition(
                        agent_id, task, _task_state_update(task, user_text)
                    )

        duration = time.time() - start_time

        usage = data.get("usage", {})

        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = context_tokens
        if context_tokens > 0:
            history_tokens = round(raw_context_tokens * (prompt_tokens / context_tokens))
        else:
            history_tokens = raw_context_tokens
        total_tokens = prompt_tokens + completion_tokens

        # 6. Сохраняем ответ в историю: сообщения задачи — в саму задачу,
        #    сообщения обычного чата — в сессию.
        conversation.append({"role": "assistant", "content": content})
        if task is not None:
            storage.save_task_messages(task["task_id"], conversation)
        else:
            storage.save(agent_id, conversation)
        storage.set_session_meta(agent_id, strategy, window_size)

        context_summary = build_context_summary(strategy, conversation, working, long_term, window_size, agent_id)

        cost = (
            (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
            + (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
            + suggest_cost
        )

        return {
            "session_id": agent_id,
            "response": content,
            "strategy": strategy,
            "profile": {
                "id": profile["profile_id"],
                "name": profile["name"],
                "display_name": profile["display_name"],
            } if profile else None,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "history_tokens": history_tokens,
                "total_tokens": total_tokens,
            },
            "context": context_summary,
            "memory": {"working": working, "long_term": long_term},
            "task": task,
            "invariants": invariants,
            "refused": refused,
            "violated_invariant": violated_invariant,
            "pending_memory": pending_memory,
            "memory_ops_applied": memory_ops_applied,
            "duration": round(duration, 3),
            "cost": round(cost, 6),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Память (три слоя)
# ============================================================
@app.get("/agent/{session_id}/memory")
async def get_memory(session_id: str):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return storage.list_memory(session_id)


@app.post("/agent/{session_id}/memory")
async def apply_memory_ops(session_id: str, request: MemoryOpsRequest):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    applied = []
    for op in request.ops:
        applied.append(storage.apply_memory_op(session_id, op, source="manual"))
    return {"applied": applied, "memory": storage.list_memory(session_id)}


@app.post("/agent/{session_id}/memory/suggest")
async def suggest_endpoint(session_id: str, request: SuggestRequest):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    conversation = storage.load(session_id) or []
    message = request.message or next(
        (m for m in reversed(conversation) if m.get("role") == "user"), None
    )
    if message is None:
        raise HTTPException(status_code=400, detail="Нет сообщения пользователя для анализа")
    working = storage.load_working(session_id, storage.resolve_working_profile_id(session_id))
    long_term = storage.load_long_term_for_profile(storage.get_session_profile(session_id))
    suggestion, s_inp, s_out = suggest_memory(working, long_term, message.get("content", ""), request.model)
    return {
        "pending_memory": flatten_suggestions(suggestion),
        "cost": round(_tokens_money(s_inp, s_out), 6),
    }


# Глобальная долговременная память (общая для всех сессий).
# Опционально — фильтр по профилю (?profile_id=...): общие записи + записи профиля.
@app.get("/memory/longterm")
async def list_long_term(profile_id: Optional[str] = None):
    if profile_id:
        entries = storage.load_long_term_for_profile(profile_id)
    else:
        entries = storage.load_long_term()
    return {"long_term": entries}


# Добавление/удаление в долговременной памяти НЕ зависит от диалога/сессии.
@app.post("/memory/longterm")
async def apply_long_term_ops(request: MemoryOpsRequest):
    applied = []
    for op in request.ops:
        action = (op.get("action") or "").strip().lower()
        layer = (op.get("layer") or "").strip().lower()
        if action not in ("save", "delete") or layer != "long_term":
            raise HTTPException(
                status_code=400,
                detail="Глобальный эндпоинт принимает только save/delete для long_term",
            )
        applied.append(storage.apply_memory_op(None, op, source="manual"))
    return {"applied": applied, "long_term": storage.load_long_term()}


# ============================================================
# Инварианты (жёсткие ограничения агента)
# ============================================================
@app.get("/invariants")
async def list_invariants():
    return {"invariants": storage.list_invariants()}


@app.post("/invariants")
async def save_invariant(request: InvariantRequest):
    try:
        inv = storage.save_invariant(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"invariant": inv, "invariants": storage.list_invariants()}


@app.post("/invariants/{invariant_id}/toggle")
async def toggle_invariant(invariant_id: str):
    inv = storage.get_invariant(invariant_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Инвариант не найден")
    return {"invariant": storage.set_invariant_active(invariant_id, not inv["is_active"])}


@app.delete("/invariants/{invariant_id}")
async def delete_invariant(invariant_id: str):
    if not storage.delete_invariant(invariant_id):
        raise HTTPException(status_code=404, detail="Инвариант не найден")
    return {"deleted": True, "invariant_id": invariant_id}


# ============================================================
# Профиль пользователя (персонализация)
# ============================================================
def _profiles_payload() -> dict:
    active = storage.get_active_profile()
    return {
        "profiles": storage.list_profiles(),
        "active_profile_id": active["profile_id"] if active else None,
    }


@app.get("/profiles")
async def list_profiles():
    return _profiles_payload()


@app.get("/profiles/{profile_id}")
async def get_profile(profile_id: str):
    profile = storage.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    return {"profile": profile}


@app.post("/profiles")
async def save_profile(request: ProfileRequest):
    try:
        profile = storage.save_profile({
            "profile_id": request.profile_id,
            "name": request.name,
            "display_name": request.display_name,
            "style": request.style,
            "format": request.format,
            "constraints": request.constraints,
            "notes": request.notes,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if request.is_active:
        storage.set_active_profile(profile["profile_id"])
    payload = _profiles_payload()
    payload["profile"] = profile
    return payload


@app.post("/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    if not storage.set_active_profile(profile_id):
        raise HTTPException(status_code=404, detail="Профиль не найден")
    return _profiles_payload()


@app.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    if not storage.delete_profile(profile_id):
        raise HTTPException(status_code=404, detail="Профиль не найден")
    return _profiles_payload()


# ============================================================
# История, ветки, сессии
# ============================================================
@app.get("/agent/{session_id}")
async def get_agent_history(session_id: str):
    meta = storage.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    messages = storage.load(session_id)
    working = storage.load_working(session_id, storage.resolve_working_profile_id(session_id))
    profile_id = storage.get_session_profile(session_id)
    long_term = storage.load_long_term_for_profile(profile_id)
    branches = storage.list_branches(session_id)
    tasks = storage.list_tasks(session_id)
    active_task_id = next((t["task_id"] for t in tasks if t["status"] == "active"), None)
    invariants = storage.get_applicable_invariants(profile_id, session_id, active_task_id)
    return {
        "session_id": session_id,
        "messages": messages,
        "strategy": meta["strategy"],
        "window_size": meta["window_size"],
        "profile_id": profile_id,
        "profile": storage.get_profile(profile_id) if profile_id else None,
        "working": working,
        "long_term": long_term,
        "current_branch": meta["current_branch"],
        "branches": branches,
        "tasks": tasks,
        "active_task_id": active_task_id,
        "invariants": invariants,
    }


@app.get("/agent/{session_id}/profile")
async def get_session_profile(session_id: str):
    meta = storage.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    profile_id = meta.get("profile_id")
    active = storage.get_active_profile()
    return {
        "session_id": session_id,
        "profile_id": profile_id,
        "profile": storage.get_profile(profile_id) if profile_id else None,
        "profiles": storage.list_profiles(),
        "active_profile_id": active["profile_id"] if active else None,
    }


@app.post("/agent/{session_id}/profile")
async def set_session_profile(session_id: str, request: SessionProfileRequest):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if request.profile_id and storage.get_profile(request.profile_id) is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    storage.set_session_profile(session_id, request.profile_id)
    profile_id = storage.get_session_profile(session_id)
    return {
        "session_id": session_id,
        "profile_id": profile_id,
        "profile": storage.get_profile(profile_id) if profile_id else None,
    }


@app.get("/agent/{session_id}/branches")
async def list_session_branches(session_id: str):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {
        "branches": storage.list_branches(session_id),
        "current_branch": storage.get_current_branch_id(session_id),
    }


@app.post("/agent/{session_id}/branch")
async def create_branches(session_id: str, request: BranchRequest):
    messages = storage.load(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    total = len(messages)
    checkpoint = total if request.checkpoint is None else request.checkpoint
    checkpoint = max(0, min(checkpoint, total))
    prefix = messages[:checkpoint]

    existing_names = {b["name"] for b in storage.list_branches(session_id)}

    def unique_name(base):
        name = base
        i = 2
        while name in existing_names:
            name = f"{base} {i}"
            i += 1
        existing_names.add(name)
        return name

    branch_a = storage.create_branch(session_id, unique_name("Ветка A"), prefix, checkpoint)
    branch_b = storage.create_branch(session_id, unique_name("Ветка B"), prefix, checkpoint)
    storage.switch_branch(session_id, branch_a)

    return {
        "checkpoint": checkpoint,
        "current_branch": branch_a,
        "messages": list(prefix),
        "branches": storage.list_branches(session_id),
    }


@app.post("/agent/{session_id}/switch")
async def switch_branch(session_id: str, request: SwitchRequest):
    if not storage.switch_branch(session_id, request.branch_id):
        raise HTTPException(status_code=404, detail="Ветка не найдена")
    return {
        "current_branch": request.branch_id,
        "messages": storage.load_branch(request.branch_id),
        "branches": storage.list_branches(session_id),
    }


@app.get("/agents")
async def list_agents():
    return {"sessions": storage.list_sessions()}


@app.delete("/agent/{session_id}")
async def delete_agent_history(session_id: str):
    if not storage.delete(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"deleted": True, "session_id": session_id}


# ============================================================
# Задачи (конечный автомат)
# ============================================================
@app.get("/agent/{session_id}/tasks")
async def list_tasks(session_id: str):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    tasks = storage.list_tasks(session_id)
    active = next((t["task_id"] for t in tasks if t["status"] == "active"), None)
    return {"tasks": tasks, "active_task_id": active}


@app.post("/agent/{session_id}/tasks")
async def create_task(session_id: str, request: TaskCreateRequest):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    task = storage.create_task(session_id, title=(request.title or ""))
    return {"task": task}


@app.get("/agent/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    task = _require_task(session_id, task_id)
    return {"task": task, "transitions": storage.list_task_transitions(task_id)}


@app.post("/agent/{session_id}/tasks/{task_id}/pause")
async def pause_task(session_id: str, task_id: str):
    _require_task(session_id, task_id)
    return {"task": storage.pause_task(task_id)}


@app.post("/agent/{session_id}/tasks/{task_id}/resume")
async def resume_task(session_id: str, task_id: str):
    _require_task(session_id, task_id)
    return {"task": storage.resume_task(task_id)}


@app.post("/agent/{session_id}/tasks/{task_id}/transition")
async def transition_task(session_id: str, task_id: str, request: TaskTransitionRequest):
    task = _require_task(session_id, task_id)
    fields, transition = apply_state_update(task, request.model_dump())
    task = storage.update_task(
        task_id, fields, source="manual", reason=transition["reason"]
    )
    return {"task": task}


@app.delete("/agent/{session_id}/tasks/{task_id}")
async def delete_task(session_id: str, task_id: str):
    _require_task(session_id, task_id)
    if not storage.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {"deleted": True, "task_id": task_id}


# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
