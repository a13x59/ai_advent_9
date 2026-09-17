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

from storage import storage, WORKING_KINDS, WORKING_STATES, LONG_TERM_KINDS
from task_state import (
    TASK_STATE_SYSTEM,
    TASK_STATE_TOOL,
    apply_state_update,
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
    # задача сессии, либо создаётся новая (с task_title).
    task_id: Optional[str] = Field(None, description="ID задачи, к которой относится запрос")
    task_title: Optional[str] = Field(None, description="Название новой задачи (если создаём)")


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


def call_deepseek_raw(messages, model, temperature=1.0, top_p=1.0, max_tokens=4096, top_k=0, stop=None, tools=None):
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
    if tools is not None:
        payload["tools"] = tools

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
    return response.json()


def call_deepseek(messages, request: AgentRequest, tools=None):
    return call_deepseek_raw(
        messages,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
        stop=request.stop,
        tools=tools,
    )


# ============================================================
# Конечный автомат задачи: извлечение перехода из ответа модели
# ============================================================
def extract_task_update(data) -> Optional[dict]:
    """Достаёт аргументы tool call update_task_state из ответа DeepSeek."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") == "update_task_state":
            args = fn.get("arguments") or "{}"
            try:
                return json.loads(args)
            except Exception:
                return None
    return None


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


def mock_task_turn(task: dict, user_message: str) -> tuple:
    """Детерминированный мок конечного автомата. Возвращает (content, task_update)."""
    stage = task["stage"]
    content = ""
    update = None

    if stage == "planning":
        plan = _mock_plan(user_message)
        total = len(plan)
        plan[0]["state"] = "in_progress"
        content = (
            f"Эхо (mock). Составил план задачи «{task['title']}»:\n"
            + "\n".join(f"{s['index']}. {s['label']}" for s in plan)
            + "\n\nПерехожу к выполнению (этап execution)."
        )
        update = {
            "stage": "execution",
            "step_index": 1,
            "step_total": total,
            "step_label": plan[0]["label"],
            "expected_action": "wait_user",
            "plan": plan,
            "reason": "План готов, начинаю выполнение",
        }
    elif stage == "execution":
        idx = task["step_index"]
        total = task["step_total"] or len(task["plan"])
        plan = [dict(s) for s in task["plan"]]
        for s in plan:
            if s["index"] == idx:
                s["state"] = "done"
        if idx >= total:
            content = "Эхо (mock). Все шаги выполнены. Перехожу к валидации (этап validation)."
            update = {
                "stage": "validation",
                "step_index": total,
                "step_total": total,
                "step_label": "проверка результата",
                "expected_action": "wait_user",
                "plan": plan,
                "reason": "Выполнение завершено, проверяю результат",
            }
        else:
            nxt = idx + 1
            for s in plan:
                if s["index"] == nxt:
                    s["state"] = "in_progress"
            content = (
                f"Эхо (mock). Выполнил шаг {idx}/{total}: {plan[idx - 1]['label']}.\n"
                f"Следующий шаг {nxt}/{total}: {plan[nxt - 1]['label']}."
            )
            update = {
                "stage": "execution",
                "step_index": nxt,
                "step_total": total,
                "step_label": plan[nxt - 1]["label"],
                "expected_action": "wait_user",
                "plan": plan,
                "reason": f"Шаг {idx} готов, перехожу к шагу {nxt}",
            }
    elif stage == "validation":
        content = "Эхо (mock). Валидация пройдена. Задача завершена (этап done)."
        update = {
            "stage": "done",
            "step_index": task["step_total"],
            "step_total": task["step_total"],
            "step_label": "завершено",
            "expected_action": "done",
            "plan": task["plan"],
            "reason": "Результат проверен, задача готова",
        }
    else:  # done
        content = "Эхо (mock). Задача уже завершена."
        update = None

    return content, update


def _require_task(session_id: str, task_id: str) -> dict:
    task = storage.get_task(task_id)
    if task is None or task["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


def _resolve_or_create_task(session_id: str, task_id: Optional[str], title: Optional[str]) -> dict:
    """Определяет задачу для запроса: явная → активная → новая."""
    if task_id:
        return _require_task(session_id, task_id)
    task = storage.get_active_task(session_id)
    if task is None:
        task = storage.create_task(session_id, title=(title or ""))
    return task


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


def build_request_messages(strategy, conversation, working, long_term, window_size, profile=None, task=None, resume_instruction=None):
    """Собирает сообщения модели из профиля, состояния задачи и трёх слоёв памяти.

    • профиль          — системный блок (всегда, если задан);
    • состояние задачи — инструкция контроллера + блок конечного автомата;
    • краткосрочная    — окно последних N (или полная ветка для branching);
    • рабочая          — системный блок (всегда);
    • долговременная   — системный блок (для sticky_facts — только факты kind=fact).
    """
    if strategy == STRATEGY_STICKY_FACTS:
        block = build_memory_system_block(working, long_term, long_term_kinds={"fact"})
    else:
        block = build_memory_system_block(working, long_term)

    messages = []
    profile_block = build_profile_system_block(profile)
    if profile_block:
        messages.append({"role": "system", "content": profile_block})
    if task:
        messages.append({"role": "system", "content": TASK_STATE_SYSTEM})
        task_block = render_task_block(task)
        if task_block:
            messages.append({"role": "system", "content": task_block})
        if resume_instruction:
            messages.append({"role": "system", "content": resume_instruction})
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
        # 1. Восстановление сессии.
        conversation = storage.load(agent_id)

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
        # working = storage.load_working(agent_id, working_profile_id)
        # long_term = storage.load_long_term_for_profile(profile["profile_id"] if profile else None)

        current_user_message = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            None,
        )

        if conversation is None:
            conversation = list(request.messages)
            storage.create_session(
                agent_id, strategy=strategy, window_size=window_size,
                messages=conversation,
                profile_id=profile["profile_id"] if profile else None,
            )
        else:
            storage.set_session_meta(agent_id, strategy, window_size)
            if current_user_message and (not conversation or conversation[-1] != current_user_message):
                conversation.append(current_user_message)

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

        # 2.5. Задача (конечный автомат). Явная → активная → новая.
        task = _resolve_or_create_task(agent_id, request.task_id, request.task_title)
        was_paused = task["status"] == "paused"
        if was_paused:
            # Сообщение пришло к задаче на паузе → возобновляем и продолжаем с того же места.
            task = storage.resume_task(task["task_id"])
        resume_instruction = render_resume_instruction(task) if was_paused else None

        # 3. Предложения памяти (опционально). Ничего не сохраняется автоматически.
        suggest_cost = 0.0
        pending_memory = []
        if request.auto_suggest_memory and current_user_message:
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
        )

        raw_context_tokens = estimate_messages_tokens(conversation)
        context_tokens = estimate_messages_tokens(request_messages)

        # 5. Вызов модели. Один вызов возвращает и ответ, и предложение перехода
        #    конечного автомата (tool call / мок) — экономим вызовы.
        task_update = None
        if API_KEY == "sk-1234567890":
            content, task_update = mock_task_turn(
                task, current_user_message.get("content", "") if current_user_message else ""
            )
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
            data = call_deepseek(request_messages, request, tools=[TASK_STATE_TOOL])
            task_update = extract_task_update(data)

        duration = time.time() - start_time

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
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

        # 5.5. Применяем переход конечного автомата (валидация в task_state).
        if task_update:
            fields, transition = apply_state_update(task, task_update)
            task = storage.update_task(
                task["task_id"], fields, source="model", reason=transition["reason"]
            )

        # 6. Сохраняем ответ в краткосрочную память (активная ветка).
        conversation.append({"role": "assistant", "content": content})
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
