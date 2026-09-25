# agent_core.py
"""Общий пайплайн агента и фабрика FastAPI-приложения.

Этот модуль НЕ содержит сетевых вызовов к DeepSeek и НЕ содержит моков:
все точки взаимодействия с моделью вынесены в интерфейс AgentProvider.

  • main.py      — реальный провайдер (DeepSeekProvider) + create_app(...);
  • mock_agent.py — детерминированный провайдер (MockProvider) для тестов.

Логика обработки запроса (классификация → инварианты → конечный автомат →
сборка контекста → генерация → сохранение) единая для обоих провайдеров.
"""
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from storage import (
    HistoryStorage,
    storage as default_storage,
    WORKING_KINDS,
    WORKING_STATES,
    LONG_TERM_KINDS,
    INVARIANT_CATEGORIES,
)
from task_state import (
    GUARD_LABELS,
    allowed_next_stages,
    blocked_next_stages,
    render_task_block,
    render_resume_instruction,
    validate_and_apply,
)

logger = logging.getLogger("agent")

# ------------------------------------------------------------
# Цены DeepSeek (за 1M токенов) — для оценки стоимости.
# ------------------------------------------------------------
INPUT_PRICE_PER_M = 0.14
OUTPUT_PRICE_PER_M = 0.28

# ------------------------------------------------------------
# Стратегии управления контекстом
# ------------------------------------------------------------
STRATEGY_SLIDING_WINDOW = "sliding_window"
STRATEGY_STICKY_FACTS = "sticky_facts"
STRATEGY_BRANCHING = "branching"
STRATEGIES = {STRATEGY_SLIDING_WINDOW, STRATEGY_STICKY_FACTS, STRATEGY_BRANCHING}
DEFAULT_STRATEGY = STRATEGY_SLIDING_WINDOW
DEFAULT_WINDOW_SIZE = 10

# Параметры вспомогательных вызовов (предложения памяти).
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

# Максимальное число повторных генераций при недопустимом маркере [STATE].
MAX_TRANSITION_RETRIES = 2

# Модели DeepSeek, поддерживающие нативный function calling (tools).
# deepseek-reasoner и легаси-модели (coder/v4-flash) его не поддерживают.
TOOL_CAPABLE_MODELS = {"deepseek-chat"}

# ------------------------------------------------------------
# Промпты (используются реальным провайдером в main.py).
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Провайдер модели
# ------------------------------------------------------------
class AgentProvider(ABC):
    """Граница «модель»: реальный DeepSeek или детерминированный мок."""

    @abstractmethod
    def complete(self, messages: List[dict], request: "AgentRequest",
                 task: Optional[dict], user_text: str,
                 tools: Optional[List[dict]] = None) -> dict:
        """Основная генерация. Возвращает DeepSeek-совместимый dict вида
        {"choices": [{"message": {"content": ..., "tool_calls": [...]}}], "usage": {...}}.
        Содержимое content МОЖЕТ содержать строку-маркер [STATE] {...}.
        tools — опциональный список определений инструментов (function calling).
        """

    @abstractmethod
    def suggest_memory(self, working: List[dict], long_term: List[dict],
                       user_message: str, model: str):
        """Возвращает (suggestion, input_tokens, output_tokens).
        suggestion = {"working": [...], "long_term": [...]}.
        """

    @abstractmethod
    def check_invariants(self, invariants: List[dict], user_text: str, model: str) -> dict:
        """Возвращает вердикт {"violation": bool, "invariant_name": str,
        "reason": str, "invariant": dict|None}."""


# ------------------------------------------------------------
# Модели входных данных
# ------------------------------------------------------------
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
    memory_ops: List[dict] = Field([], description="Явные операции над памятью (save/delete/move)")
    auto_suggest_memory: bool = Field(False, description="Генерировать предложения памяти после запроса")
    enable_tools: bool = Field(False, description="Разрешить вызов MCP-инструментов (function calling)")
    profile_id: Optional[str] = Field(None, description="ID профиля пользователя для этого запроса")
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
    profile_id: Optional[str] = Field(None, description="ID профиля (если обновляем существующий)")
    name: str = Field(..., description="Уникальный ключ профиля (напр. ivan_dev)")
    display_name: Optional[str] = Field("", description="Отображаемое имя пользователя")
    style: Optional[str] = Field("", description="Стиль общения")
    format: Optional[str] = Field("", description="Формат ответа")
    constraints: Optional[str] = Field("", description="Ограничения")
    notes: Optional[str] = Field("", description="Дополнительные пожелания")
    is_active: Optional[bool] = Field(False, description="Сделать профиль активным после сохранения")


class SessionProfileRequest(BaseModel):
    profile_id: Optional[str] = Field(None, description="ID профиля; null/None — отвязать профиль от сессии")


class TaskCreateRequest(BaseModel):
    title: Optional[str] = Field("", description="Название задачи")


class TaskTransitionRequest(BaseModel):
    """Ручной переход конечного автомата (для отладки/тестов/UI)."""
    stage: str = Field(..., description="Целевой этап")
    expected_action: Optional[str] = Field(None, description="Ожидаемое действие")
    step_index: Optional[int] = Field(None, description="Текущий шаг")
    step_total: Optional[int] = Field(None, description="Всего шагов")
    step_label: Optional[str] = Field(None, description="Название текущего шага")
    plan: Optional[List[dict]] = Field(None, description="План шагов")
    guards: Optional[dict] = Field(None, description="Гварды (plan_approved/validation_passed) — только ручное/отладочное управление")
    reason: Optional[str] = Field("", description="Причина перехода")


class InvariantRequest(BaseModel):
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


# ------------------------------------------------------------
# Подсчёт токенов
# ------------------------------------------------------------
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
    total = 0
    for message in messages or []:
        total += estimate_tokens(message.get("content", ""))
    return total


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


# ------------------------------------------------------------
# Классификация запроса
# ------------------------------------------------------------
TASK_VERBS = (
    "напиши", "напишите", "сделай", "сделайте", "реализуй", "реализуйте",
    "создай", "создайте", "разработай", "разработайте", "исправь", "исправьте",
    "настрой", "настройте", "добавь", "добавьте", "сгенерируй", "сгенерируйте",
    "построй", "постройте", "перенеси", "мигрируй", "проверь", "протестируй",
    "составь", "составьте", "придумай", "придумайте", "подготовь", "подготовьте",
    "почини", "оптимизируй", "перепиши", "отрефактори",
)

CONTINUE_WORDS = ("продолжай", "продолжи", "дальше", "continue", "next", "ок", "да", "го")

# Слова-подтверждения (утверждение плана / подтверждение валидации).
APPROVAL_WORDS = (
    "да", "ок", "окей", "утверждаю", "подтверждаю", "подтвердить", "утвердить",
    "принято", "согласен", "согласна", "завершай", "завершить", "финализируй",
    "yes", "ok", "okay", "approve", "confirm",
)

QUESTION_WORDS = (
    "кто", "что", "почему", "зачем", "как", "какой", "какая", "какие", "какое",
    "когда", "где", "сколько", "чей", "чья", "чьё", "чем", "кому", "кого",
    "откуда", "куда", "ли",
)


def _detect_task(user_message: str) -> bool:
    t = (user_message or "").strip().lower()
    if not t:
        return False
    return any(re.search(r"\b" + re.escape(verb) + r"\b", t) for verb in TASK_VERBS)


def _is_continue(user_message: str) -> bool:
    t = (user_message or "").strip().lower().rstrip("!.? ")
    return t in CONTINUE_WORDS or "продолж" in t or "дальше" in t


def _is_approval(user_message: str) -> bool:
    t = (user_message or "").strip().lower().rstrip("!.? ")
    return t in APPROVAL_WORDS


def _is_question(user_message: str) -> bool:
    t = (user_message or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first_word = (t.split() or [""])[0]
    return first_word in QUESTION_WORDS


# ------------------------------------------------------------
# Маркер прогресса [STATE]
# ------------------------------------------------------------
def build_state_marker_instruction(task: Optional[dict]) -> str:
    """Инструкция модели: как сообщать прогресс и какие переходы разрешены."""
    allowed = allowed_next_stages(task) if task else list()
    blocked = blocked_next_stages(task) if task else list()
    lines = [
        "В конце ответа, отдельной строкой, укажи фактический прогресс задачи строго в формате:",
        '[STATE] {"stage": "<planning|execution|validation|done>", '
        '"step_index": <номер шага>, "step_total": <всего шагов>, '
        '"step_label": "<подпись шага>", '
        '"expected_action": "<model_call|wait_user|confirm|done>", '
        '"plan": [{"index": 1, "label": "..."}]}',
        "Это технический маркер — не показывай его как часть ответа пользователю.",
        "Правила переходов (нарушение = маркер отклоняется и ответ перегенерируется):",
        f"- Разрешённые переходы из текущего этапа: {', '.join(allowed) or '—'}.",
        "- За один ответ можно перейти максимум на ОДИН этап вперёд (перескок — ошибка).",
        "- stage=execution ставь ТОЛЬКО после того, как пользователь утвердил план (guard plan_approved).",
        "- stage=done ставь ТОЛЬКО после того, как пользователь подтвердил валидацию (guard validation_passed).",
        "- Переходы в execution и в done выполняет пользователь, а не ты — не ставь их сам.",
        "- Если составляешь план (stage=planning) — заполни поле plan и поставь expected_action=confirm.",
        "- Если выполнение завершено — переходи в validation и поставь expected_action=confirm.",
    ]
    for b in blocked:
        labels = ", ".join(GUARD_LABELS.get(g, g) for g in b["missing_guards"])
        lines.append(f"- Этап «{b['stage']}» сейчас заблокирован: нужно {labels}.")
    return "\n".join(lines)


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


# ------------------------------------------------------------
# Утверждение плана / подтверждение валидации (действия пользователя)
# ------------------------------------------------------------
def _approval_update(task: dict) -> Optional[dict]:
    """Обновление, которое выполняет пользователь при подтверждении (approve)."""
    stage = task.get("stage")
    if stage == "planning":
        plan = task.get("plan") or []
        first = plan[0] if plan else {"index": 1, "label": "выполнение"}
        return {
            "stage": "execution",
            "guards": {"plan_approved": True},
            "step_index": 1,
            "step_total": len(plan) or 1,
            "step_label": first.get("label") or "выполнение",
            "expected_action": "wait_user",
            "reason": "план утверждён пользователем",
        }
    if stage == "validation":
        total = task.get("step_total") or 0
        return {
            "stage": "done",
            "guards": {"validation_passed": True},
            "step_index": total,
            "step_total": total,
            "step_label": "завершено",
            "expected_action": "done",
            "reason": "валидация подтверждена пользователем",
        }
    return None


def _approval_ack(task: dict) -> str:
    """Детерминированный ответ на подтверждение пользователя."""
    stage = task.get("stage")
    if stage == "execution":
        total = task.get("step_total") or 0
        label = task.get("step_label") or "выполнение"
        return f"✅ План утверждён. Перехожу к выполнению — шаг 1/{total}: {label}."
    if stage == "done":
        return "✅ Валидация подтверждена. Задача завершена."
    return "✅ Подтверждено."


def _rejection_message(verdict: dict) -> str:
    allowed = verdict.get("allowed") or []
    return (
        "Твой маркер [STATE] отклонён конечным автоматом. "
        f"Причина: {verdict.get('reason') or 'недопустимый переход'}. "
        f"Разрешённые переходы сейчас: {', '.join(allowed) or '—'}. "
        "Исправь состояние и повтори ответ с корректным маркером [STATE]."
    )


def _run_task_turn(provider: AgentProvider, request: AgentRequest, messages: List[dict],
                   task: dict, task_was_none: bool, store: HistoryStorage, user_text: str):
    """Один модельный ход задачи: маркер → валидация → (при отказе) повторная генерация.

    Недопустимый переход НЕ сохраняется в историю; вместо этого модель получает
    системное пояснение и перегенерирует ответ (до MAX_TRANSITION_RETRIES раз).
    Возвращает (task, content, data).
    """
    retries = 0
    data: dict = {}
    content = ""
    while True:
        data = provider.complete(messages, request, task, user_text)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        marker, content = _extract_state_marker(content)

        if marker is None:
            # Новая задача обязана вернуть план через маркер — просим его.
            if task_was_none and retries < 1:
                messages.append({
                    "role": "system",
                    "content": "Не хватает маркера [STATE]. Укажи фактический прогресс задачи.",
                })
                retries += 1
                continue
            # Маркера нет — прогресс не заявлен, состояние не меняем.
            break

        verdict = validate_and_apply(task, marker, source="model")
        if verdict["accepted"]:
            task = store.update_task(
                task["task_id"], verdict["fields"],
                source="model", reason=verdict["transition"]["reason"],
            )
            break

        # Отклонённый переход: логируем и повторяем с пояснением.
        target = marker.get("stage") or task["stage"]
        store.record_rejected_transition(task, target, verdict["reason"], source="model")
        if retries < MAX_TRANSITION_RETRIES:
            messages.append({"role": "system", "content": _rejection_message(verdict)})
            retries += 1
            continue
        content = (content + "\n\n⚠️ " + verdict["reason"]).strip()
        break
    return task, content, data


# ------------------------------------------------------------
# Инструменты (MCP) — нативный function calling DeepSeek
# ------------------------------------------------------------
def _tools_enabled(request: AgentRequest, mcp_client) -> bool:
    """Инструменты активны: есть клиент, включены запросом и модель поддерживает tools."""
    if mcp_client is None:
        return False
    if not getattr(request, "enable_tools", False):
        return False
    return (getattr(request, "model", "") or "").strip() in TOOL_CAPABLE_MODELS


def _deepseek_tools(mcp_tools: List[dict]) -> List[dict]:
    """Маппинг MCP-инструментов (inputSchema) в формат tools DeepSeek.

    inputSchema у MCP — это уже JSON Schema объекта (type/properties/required),
    поэтому переносим его в поле parameters практически без изменений.
    """
    out = []
    for t in mcp_tools or []:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        if not name:
            continue
        schema = t.get("inputSchema") or {}
        if not isinstance(schema, dict):
            schema = {}
        parameters = dict(schema)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": parameters,
            },
        })
    return out


def _run_tool_turn(provider: AgentProvider, request: AgentRequest, messages: List[dict],
                   mcp_client, user_text: str):
    """Один ход чата с инструментами (двухшаговый function calling).

    Раунд 1 — модель получает список инструментов и может вернуть tool_calls;
    раунд 2 — после подстановки результатов модель пишет финальный ответ,
    использующий эти результаты.

    Возвращает (content, tool_calls_info, data), где data — данные ПОСЛЕДНЕГО
    вызова модели (для usage/статистики).
    """
    tool_calls_info: List[dict] = []
    data: dict = {}

    try:
        mcp_tools = mcp_client.list_tools()
    except Exception as e:
        logger.warning("Не удалось получить список инструментов MCP: %s", e)
        mcp_tools = []
    tools = _deepseek_tools(mcp_tools)

    if not tools:
        data = provider.complete(messages, request, None, user_text)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, tool_calls_info, data

    # Раунд 1: модель с инструментами.
    data = provider.complete(messages, request, None, user_text, tools=tools)
    message = data.get("choices", [{}])[0].get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return message.get("content", "") or "", tool_calls_info, data

    # Подставляем ассистентский tool_calls и результаты инструментов.
    messages.append({
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": tool_calls,
    })
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = (fn.get("name") or "").strip()
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        try:
            res = mcp_client.call_tool(name, arguments)
            if isinstance(res, dict):
                text = res.get("text", "")
                ok = bool(res.get("ok", False))
                error = res.get("error", "")
            else:
                text = str(res)
                ok = False
                error = ""
        except Exception as e:
            text = f"Ошибка вызова инструмента: {e}"
            ok = False
            error = str(e)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "content": text,
        })
        tool_calls_info.append({
            "name": name,
            "arguments": arguments,
            "result": text,
            "ok": ok,
            "error": error,
        })

    # Раунд 2: финальный ответ, использующий результаты инструментов.
    data = provider.complete(messages, request, None, user_text)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, tool_calls_info, data


# ------------------------------------------------------------
# Генерация предложений памяти
# ------------------------------------------------------------
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
    pending = []
    for c in (suggestion or {}).get("working") or []:
        c = dict(c)
        c["layer"] = "working"
        pending.append(c)
    for c in (suggestion or {}).get("long_term") or []:
        c = dict(c)
        c["layer"] = "long_term"
        pending.append(c)
    return pending


# ------------------------------------------------------------
# Формирование контекста
# ------------------------------------------------------------
def memory_entries_to_text(entries, layer, max_entries=40) -> str:
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
    parts = []
    if (inv.get("profile_id") or "").strip():
        parts.append("profile")
    if (inv.get("session_id") or "").strip():
        parts.append("session")
    if (inv.get("task_id") or "").strip():
        parts.append("task")
    return "+".join(parts) if parts else "global"


def build_invariants_system_block(invariants) -> str:
    if not invariants:
        return ""
    lines = [
        "ИНВАРИАНТЫ (нерушимые ограничения — нарушать их НЕЛЬЗЯ):",
        "Это не пожелания и не советы, а жёсткие правила. Проверяй КАЖДЫЙ ответ "
        "на соответствие каждому инварианту.",
        "Если запрос или предлагаемое решение нарушает хотя бы один инвариант — "
        "откажись и объясни, какой именно инвариант и почему он не может быть нарушен.",
        "Это ПОЛНЫЙ актуальный список. Ограничения, упомянутые в истории диалога, "
        "но не перечисленные здесь, больше НЕ действуют — не применяй их.",
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


def build_refusal_reply(inv, verdict) -> str:
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


def build_request_messages(strategy, conversation, working, long_term, window_size,
                           profile=None, task=None, resume_instruction=None,
                           report_state=False, invariants=None):
    """Собирает сообщения модели из инвариантов, профиля, задачи и трёх слоёв памяти."""
    conversation = [m for m in (conversation or []) if not m.get("invariant_refusal")]

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
            messages.append({"role": "system", "content": build_state_marker_instruction(task)})
    if block:
        messages.append({"role": "system", "content": block})
    if strategy == STRATEGY_BRANCHING:
        messages.extend(list(conversation))
    else:
        messages.extend(conversation[-window_size:])
    return messages


def build_context_summary(store, strategy, conversation, working, long_term, window_size, session_id):
    total = len(conversation)
    info = {
        "strategy": strategy,
        "total_messages": total,
        "window_size": window_size,
        "working_count": len(working),
        "long_term_count": len(long_term),
    }
    if strategy == STRATEGY_BRANCHING:
        current = store.get_current_branch_id(session_id)
        info["branch"] = {"id": current}
        info["branches"] = store.list_branches(session_id)
        info["sent_messages"] = total
        info["discarded_messages"] = 0
    else:
        info["sent_messages"] = min(total, window_size)
        info["discarded_messages"] = max(0, total - window_size)
    return info


# ------------------------------------------------------------
# Вспомогательные для задач
# ------------------------------------------------------------
def _require_task(store: HistoryStorage, session_id: str, task_id: str) -> dict:
    task = store.get_task(task_id)
    if task is None or task["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


def _resolve_task(store: HistoryStorage, session_id: str, task_id: Optional[str]) -> Optional[dict]:
    if task_id:
        return _require_task(store, session_id, task_id)
    return store.get_active_task(session_id)


# ------------------------------------------------------------
# Фабрика FastAPI-приложения
# ------------------------------------------------------------
def create_app(provider: AgentProvider, store: Optional[HistoryStorage] = None,
               mcp_client=None) -> FastAPI:
    """Собирает FastAPI-приложение со всеми эндпоинтами на едином пайплайне.

    provider   — реальный (DeepSeek) или детерминированный (Mock) провайдер;
    store      — хранилище (по умолчанию глобальный синглтон; в тестах — отдельный);
    mcp_client — клиент MCP-инструментов (list_tools/call_tool); None = без инструментов.
    """
    store = store or default_storage

    app = FastAPI(title="AI Agent Service", description="Обработка запросов к DeepSeek")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

            # Профиль пользователя.
            profile = None
            if request.profile_id:
                profile = store.get_profile(request.profile_id)
                if profile is None:
                    raise HTTPException(status_code=400, detail=f"Профиль не найден: {request.profile_id}")
            else:
                bound_profile_id = store.get_session_profile(agent_id)
                profile = store.get_profile(bound_profile_id) if bound_profile_id else None
                if profile is None:
                    profile = store.get_active_profile()
            working_profile_id = profile["profile_id"] if profile else ""

            # 2. Сессия.
            if store.get_session_meta(agent_id) is None:
                store.create_session(
                    agent_id, strategy=strategy, window_size=window_size,
                    messages=[],
                    profile_id=profile["profile_id"] if profile else None,
                )
            else:
                store.set_session_meta(agent_id, strategy, window_size)

            if request.profile_id and profile is not None:
                store.set_session_profile(agent_id, profile["profile_id"])

            # 2.1 Явные операции памяти.
            memory_ops_applied = []
            for op in request.memory_ops or []:
                try:
                    memory_ops_applied.append(store.apply_memory_op(agent_id, op, source="manual"))
                except Exception as e:
                    logger.warning("Ошибка применения memory_op %s: %s", op, e)
            working = store.load_working(agent_id, working_profile_id)
            long_term = store.load_long_term_for_profile(profile["profile_id"] if profile else None)

            # 2.5 Классификация запроса (задача / продолжение / подтверждение / чат).
            is_new_task = _detect_task(user_text)
            is_continue = _is_continue(user_text)

            task = None
            mode = "chat"
            was_paused = False
            resume_instruction = None

            if is_new_task:
                mode = "task"
            else:
                task = _resolve_task(store, agent_id, request.task_id)
                if (
                    task is not None
                    and task["status"] == "active"
                    and task.get("expected_action") == "confirm"
                    and _is_approval(user_text)
                ):
                    mode = "approve"
                elif is_continue and task is not None:
                    was_paused = task["status"] == "paused"
                    if was_paused:
                        task = store.resume_task(task["task_id"])
                    resume_instruction = render_resume_instruction(task) if was_paused else None
                    mode = "task"
                elif (
                    task is not None
                    and task["status"] == "active"
                    and task.get("expected_action") == "wait_user"
                    and not _is_question(user_text)
                ):
                    mode = "task"
                else:
                    mode = "chat"

            # 2.55 Инварианты.
            task_id_for_scope = task["task_id"] if task else ""
            invariants = store.get_applicable_invariants(
                profile["profile_id"] if profile else None, agent_id, task_id_for_scope
            )
            refusal_inv = None
            refusal_reason = ""
            if invariants:
                try:
                    verdict = provider.check_invariants(invariants, user_text, request.model)
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

            # 2.6 Для новой задачи заранее заводим оболочку (planning): план заполнит
            #     модель через маркер [STATE] и остановится с expected_action=confirm.
            task_was_none = mode == "task" and task is None
            if task_was_none:
                title = re.sub(r"\s+", " ", user_text).strip()[:60] or "Задача"
                task = store.create_task(agent_id, title=title)

            # 2.7 Контекст диалога (у задачи своя история, у чата — сессии).
            if task is not None:
                conversation = store.load_task_messages(task["task_id"]) or []
            else:
                conversation = store.load(agent_id) or []
            if current_user_message and (not conversation or conversation[-1] != current_user_message):
                if refused:
                    conversation.append({**current_user_message, "invariant_refusal": True})
                else:
                    conversation.append(current_user_message)

            # 3. Предложения памяти (опционально).
            suggest_cost = 0.0
            pending_memory = []
            if request.auto_suggest_memory and current_user_message and not refused:
                try:
                    suggestion, s_inp, s_out = provider.suggest_memory(
                        working, long_term, current_user_message.get("content", ""), request.model
                    )
                    pending_memory = flatten_suggestions(suggestion)
                    suggest_cost += _tokens_money(s_inp, s_out)
                except Exception as e:
                    logger.warning("Ошибка генерации предложений памяти: %s", e)

            # 4. Контекст для модели.
            request_messages = build_request_messages(
                strategy, conversation, working, long_term, window_size,
                profile=profile, task=task, resume_instruction=resume_instruction,
                report_state=(mode == "task"),
                invariants=invariants,
            )

            raw_context_tokens = estimate_messages_tokens(conversation)
            context_tokens = estimate_messages_tokens(request_messages)

            # 5. Генерация + переход конечного автомата.
            tools_active = _tools_enabled(request, mcp_client)
            tool_calls_info: List[dict] = []
            data: dict = {}
            if refused:
                content = build_refusal_reply(refusal_inv, {
                    "invariant_name": refusal_inv.get("name"),
                    "reason": refusal_reason or refusal_inv.get("rationale"),
                })
            elif mode == "approve":
                verdict = validate_and_apply(task, _approval_update(task), source="user")
                if verdict["accepted"]:
                    task = store.update_task(
                        task["task_id"], verdict["fields"],
                        source="user", reason=verdict["transition"]["reason"],
                    )
                content = _approval_ack(task)
            elif mode == "task":
                task, content, data = _run_task_turn(
                    provider, request, request_messages, task, task_was_none, store, user_text
                )
            else:
                if tools_active:
                    content, tool_calls_info, data = _run_tool_turn(
                        provider, request, request_messages, mcp_client, user_text
                    )
                else:
                    data = provider.complete(request_messages, request, None, user_text)
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 5.1 Результаты успешных вызовов инструментов — в рабочую память.
            if tool_calls_info:
                for tc in tool_calls_info:
                    if tc.get("ok") and tc.get("result"):
                        try:
                            store.save_working_entry(agent_id, {
                                "key": f"mcp:{tc.get('name')}",
                                "value": tc["result"],
                                "kind": "result",
                                "state": "done",
                                "profile_id": working_profile_id,
                            })
                        except Exception as e:
                            logger.warning("Ошибка сохранения результата инструмента: %s", e)
                working = store.load_working(agent_id, working_profile_id)

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

            # 6. Сохраняем ответ в историю.
            if refused:
                conversation.append({"role": "assistant", "content": content, "invariant_refusal": True})
            else:
                conversation.append({"role": "assistant", "content": content})
            if task is not None:
                store.save_task_messages(task["task_id"], conversation)
            else:
                store.save(agent_id, conversation)
            store.set_session_meta(agent_id, strategy, window_size)

            context_summary = build_context_summary(
                store, strategy, conversation, working, long_term, window_size, agent_id
            )

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
                "tool_calls": tool_calls_info,
                "tools_active": tools_active,
                "pending_memory": pending_memory,
                "memory_ops_applied": memory_ops_applied,
                "duration": round(duration, 3),
                "cost": round(cost, 6),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------
    # Память (три слоя)
    # ------------------------------------------------------------
    @app.get("/agent/{session_id}/memory")
    async def get_memory(session_id: str):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        return store.list_memory(session_id)

    @app.post("/agent/{session_id}/memory")
    async def apply_memory_ops(session_id: str, request: MemoryOpsRequest):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        applied = []
        for op in request.ops:
            applied.append(store.apply_memory_op(session_id, op, source="manual"))
        return {"applied": applied, "memory": store.list_memory(session_id)}

    @app.post("/agent/{session_id}/memory/suggest")
    async def suggest_endpoint(session_id: str, request: SuggestRequest):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        conversation = store.load(session_id) or []
        message = request.message or next(
            (m for m in reversed(conversation) if m.get("role") == "user"), None
        )
        if message is None:
            raise HTTPException(status_code=400, detail="Нет сообщения пользователя для анализа")
        working = store.load_working(session_id, store.resolve_working_profile_id(session_id))
        long_term = store.load_long_term_for_profile(store.get_session_profile(session_id))
        suggestion, s_inp, s_out = provider.suggest_memory(
            working, long_term, message.get("content", ""), request.model
        )
        return {
            "pending_memory": flatten_suggestions(suggestion),
            "cost": round(_tokens_money(s_inp, s_out), 6),
        }

    @app.get("/memory/longterm")
    async def list_long_term(profile_id: Optional[str] = None):
        if profile_id:
            entries = store.load_long_term_for_profile(profile_id)
        else:
            entries = store.load_long_term()
        return {"long_term": entries}

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
            applied.append(store.apply_memory_op(None, op, source="manual"))
        return {"applied": applied, "long_term": store.load_long_term()}

    # ------------------------------------------------------------
    # Инварианты
    # ------------------------------------------------------------
    @app.get("/invariants")
    async def list_invariants():
        return {"invariants": store.list_invariants()}

    @app.post("/invariants")
    async def save_invariant(request: InvariantRequest):
        try:
            inv = store.save_invariant(request.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"invariant": inv, "invariants": store.list_invariants()}

    @app.post("/invariants/{invariant_id}/toggle")
    async def toggle_invariant(invariant_id: str):
        inv = store.get_invariant(invariant_id)
        if inv is None:
            raise HTTPException(status_code=404, detail="Инвариант не найден")
        return {"invariant": store.set_invariant_active(invariant_id, not inv["is_active"])}

    @app.delete("/invariants/{invariant_id}")
    async def delete_invariant(invariant_id: str):
        if not store.delete_invariant(invariant_id):
            raise HTTPException(status_code=404, detail="Инвариант не найден")
        return {"deleted": True, "invariant_id": invariant_id}

    # ------------------------------------------------------------
    # Профиль пользователя
    # ------------------------------------------------------------
    def _profiles_payload() -> dict:
        active = store.get_active_profile()
        return {
            "profiles": store.list_profiles(),
            "active_profile_id": active["profile_id"] if active else None,
        }

    @app.get("/profiles")
    async def list_profiles():
        return _profiles_payload()

    @app.get("/profiles/{profile_id}")
    async def get_profile(profile_id: str):
        profile = store.get_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Профиль не найден")
        return {"profile": profile}

    @app.post("/profiles")
    async def save_profile(request: ProfileRequest):
        try:
            profile = store.save_profile({
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
            store.set_active_profile(profile["profile_id"])
        payload = _profiles_payload()
        payload["profile"] = profile
        return payload

    @app.post("/profiles/{profile_id}/activate")
    async def activate_profile(profile_id: str):
        if not store.set_active_profile(profile_id):
            raise HTTPException(status_code=404, detail="Профиль не найден")
        return _profiles_payload()

    @app.delete("/profiles/{profile_id}")
    async def delete_profile(profile_id: str):
        if not store.delete_profile(profile_id):
            raise HTTPException(status_code=404, detail="Профиль не найден")
        return _profiles_payload()

    # ------------------------------------------------------------
    # История, ветки, сессии
    # ------------------------------------------------------------
    @app.get("/agent/{session_id}")
    async def get_agent_history(session_id: str):
        meta = store.get_session_meta(session_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        messages = store.load(session_id)
        working = store.load_working(session_id, store.resolve_working_profile_id(session_id))
        profile_id = store.get_session_profile(session_id)
        long_term = store.load_long_term_for_profile(profile_id)
        branches = store.list_branches(session_id)
        tasks = store.list_tasks(session_id)
        active_task_id = next((t["task_id"] for t in tasks if t["status"] == "active"), None)
        invariants = store.get_applicable_invariants(profile_id, session_id, active_task_id)
        return {
            "session_id": session_id,
            "messages": messages,
            "strategy": meta["strategy"],
            "window_size": meta["window_size"],
            "profile_id": profile_id,
            "profile": store.get_profile(profile_id) if profile_id else None,
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
        meta = store.get_session_meta(session_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        profile_id = meta.get("profile_id")
        active = store.get_active_profile()
        return {
            "session_id": session_id,
            "profile_id": profile_id,
            "profile": store.get_profile(profile_id) if profile_id else None,
            "profiles": store.list_profiles(),
            "active_profile_id": active["profile_id"] if active else None,
        }

    @app.post("/agent/{session_id}/profile")
    async def set_session_profile(session_id: str, request: SessionProfileRequest):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        if request.profile_id and store.get_profile(request.profile_id) is None:
            raise HTTPException(status_code=404, detail="Профиль не найден")
        store.set_session_profile(session_id, request.profile_id)
        profile_id = store.get_session_profile(session_id)
        return {
            "session_id": session_id,
            "profile_id": profile_id,
            "profile": store.get_profile(profile_id) if profile_id else None,
        }

    @app.get("/agent/{session_id}/branches")
    async def list_session_branches(session_id: str):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        return {
            "branches": store.list_branches(session_id),
            "current_branch": store.get_current_branch_id(session_id),
        }

    @app.post("/agent/{session_id}/branch")
    async def create_branches(session_id: str, request: BranchRequest):
        messages = store.load(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        total = len(messages)
        checkpoint = total if request.checkpoint is None else request.checkpoint
        checkpoint = max(0, min(checkpoint, total))
        prefix = messages[:checkpoint]

        existing_names = {b["name"] for b in store.list_branches(session_id)}

        def unique_name(base):
            name = base
            i = 2
            while name in existing_names:
                name = f"{base} {i}"
                i += 1
            existing_names.add(name)
            return name

        branch_a = store.create_branch(session_id, unique_name("Ветка A"), prefix, checkpoint)
        branch_b = store.create_branch(session_id, unique_name("Ветка B"), prefix, checkpoint)
        store.switch_branch(session_id, branch_a)

        return {
            "checkpoint": checkpoint,
            "current_branch": branch_a,
            "messages": list(prefix),
            "branches": store.list_branches(session_id),
        }

    @app.post("/agent/{session_id}/switch")
    async def switch_branch(session_id: str, request: SwitchRequest):
        if not store.switch_branch(session_id, request.branch_id):
            raise HTTPException(status_code=404, detail="Ветка не найдена")
        return {
            "current_branch": request.branch_id,
            "messages": store.load_branch(request.branch_id),
            "branches": store.list_branches(session_id),
        }

    @app.get("/agents")
    async def list_agents():
        return {"sessions": store.list_sessions()}

    @app.delete("/agent/{session_id}")
    async def delete_agent_history(session_id: str):
        if not store.delete(session_id):
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        return {"deleted": True, "session_id": session_id}

    # ------------------------------------------------------------
    # Задачи (конечный автомат)
    # ------------------------------------------------------------
    @app.get("/agent/{session_id}/tasks")
    async def list_tasks(session_id: str):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        tasks = store.list_tasks(session_id)
        active = next((t["task_id"] for t in tasks if t["status"] == "active"), None)
        return {"tasks": tasks, "active_task_id": active}

    @app.post("/agent/{session_id}/tasks")
    async def create_task(session_id: str, request: TaskCreateRequest):
        if store.get_session_meta(session_id) is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        task = store.create_task(session_id, title=(request.title or ""))
        return {"task": task}

    @app.get("/agent/{session_id}/tasks/{task_id}")
    async def get_task(session_id: str, task_id: str):
        task = _require_task(store, session_id, task_id)
        return {"task": task, "transitions": store.list_task_transitions(task_id)}

    @app.get("/agent/{session_id}/tasks/{task_id}/transitions")
    async def get_task_transitions(session_id: str, task_id: str):
        task = _require_task(store, session_id, task_id)
        return {
            "task_id": task["task_id"],
            "current_stage": task["stage"],
            "guards": task.get("guards") or {},
            "allowed": allowed_next_stages(task),
            "blocked": blocked_next_stages(task),
        }

    @app.post("/agent/{session_id}/tasks/{task_id}/pause")
    async def pause_task(session_id: str, task_id: str):
        _require_task(store, session_id, task_id)
        return {"task": store.pause_task(task_id)}

    @app.post("/agent/{session_id}/tasks/{task_id}/resume")
    async def resume_task(session_id: str, task_id: str):
        _require_task(store, session_id, task_id)
        return {"task": store.resume_task(task_id)}

    @app.post("/agent/{session_id}/tasks/{task_id}/transition")
    async def transition_task(session_id: str, task_id: str, request: TaskTransitionRequest):
        task = _require_task(store, session_id, task_id)
        verdict = validate_and_apply(task, request.model_dump(), source="manual")
        if not verdict["accepted"]:
            store.record_rejected_transition(
                task, request.stage, verdict["reason"], source="manual"
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": verdict["reason"],
                    "allowed": verdict["allowed"],
                    "blocked": verdict["blocked"],
                },
            )
        task = store.update_task(
            task_id, verdict["fields"], source="manual", reason=verdict["transition"]["reason"]
        )
        return {"task": task}

    @app.delete("/agent/{session_id}/tasks/{task_id}")
    async def delete_task(session_id: str, task_id: str):
        _require_task(store, session_id, task_id)
        if not store.delete_task(task_id):
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return {"deleted": True, "task_id": task_id}

    return app
