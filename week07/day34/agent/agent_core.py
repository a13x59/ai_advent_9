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
import os
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
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

# Префиксы имён локальных моделей (Ollama и т.п.), поддерживающих function calling.
# Используется в _tools_enabled: локальные модели тоже умеют вызывать MCP-инструменты.
LOCAL_TOOL_CAPABLE_PREFIXES = ("llama", "qwen")

# Максимальное число раундов вызова инструментов в многокруговом цикле
# _run_tool_loop (цепочка search → summarize → save_to_file). Защита от зацикливания.
MAX_TOOL_CYCLES = 8

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
    "  kind ∈ {goal, constraint, todo, result, context, note, clarification, term}; "
    "state ∈ {pending, in_progress, done, blocked} или null.\n"
    "long_term — профиль/решения/знания: поля key, value, kind, tags.\n"
    "  kind ∈ {profile, decision, knowledge, preference, agreement, fact}; "
    "tags — список строк.\n"
    "Объедини новое с уже известным. Не выдумывай лишнего. "
    "Если сохранять нечего — верни пустые списки."
)

# ------------------------------------------------------------
# Ассистент поддержки (/support)
# ------------------------------------------------------------
# Детерминированный ответ при офтопике: вопрос не относится к продукту, и в базе
# знаний нет релевантных чанков. Тикет при этом НЕ создаётся и НЕ дополняется.
SUPPORT_ABSTAIN_REPLY = "Нет информации по этому вопросу."

# Аргумент --ticket_id=<N> внутри команды /support.
_SUPPORT_TICKET_ARG_RE = re.compile(r"--ticket_id\s*=\s*['\"]?([A-Za-z0-9_\-]+)['\"]?")


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
    num_ctx: Optional[int] = Field(None, ge=256, le=131072, description="Контекстное окно (локальные модели Ollama); None — дефолт сервера")
    memory_ops: List[dict] = Field([], description="Явные операции над памятью (save/delete/move)")
    auto_suggest_memory: bool = Field(False, description="Генерировать предложения памяти после запроса")
    enable_tools: bool = Field(False, description="Разрешить вызов MCP-инструментов (function calling)")
    rag: bool = Field(False, description="Режим RAG: дополнить запрос релевантными чанками из базы знаний")
    rag_mode: str = Field("baseline", description="Режим RAG: baseline|rewrite|filter|rewrite+filter|rerank")
    rag_top_k: int = Field(5, ge=1, le=50, description="Топ-K после второго этапа")
    rag_top_k_candidates: Optional[int] = Field(None, ge=1, le=200, description="Топ-K до фильтра/реранка")
    rag_min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог отсечения (filter-режим)")
    profile_id: Optional[str] = Field(None, description="ID профиля пользователя для этого запроса")
    task_id: Optional[str] = Field(None, description="ID задачи, к которой относится запрос")


class ReviewRequest(BaseModel):
    """Запрос AI-ревью diff (День 32, подход A — через агента)."""
    diff: str = Field(..., description="Unified diff для ревью")
    model: str = Field("deepseek-chat", description="Название модели")
    temperature: Optional[float] = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192)
    use_rag: bool = Field(True, description="Использовать RAG (код + документация)")


class SupportRequest(BaseModel):
    """Запрос ассистента поддержки (День 33, эндпоинт /support)."""
    message: str = Field(..., description="Сообщение: '/support <вопрос>' или '/support --ticket_id=<N> <вопрос>', либо продолжение диалога")
    session_id: Optional[str] = Field(None, description="ID сессии поддержки (для продолжения диалога в контексте тикета)")
    model: str = Field("deepseek-chat", description="Название модели")
    temperature: Optional[float] = Field(0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=8192)
    customer_email: Optional[str] = Field(None, description="E-mail покупателя для привязки нового тикета")
    rag_top_k: int = Field(5, ge=1, le=50, description="Топ-K чанков FAQ")
    rag_mode: str = Field("baseline", description="Режим RAG: baseline|filter|rerank")


class FilesRequest(BaseModel):
    """Запрос файлового ассистента (День 34, эндпоинт /files).

    message — задача на уровне ЦЕЛИ (например «найди все места использования X»,
    «сгенерируй ADR по изменениям», «проверь файлы по инварианту»); ассистент сам
    решает, какие файлы читать/искать/менять.
    apply   — False (по умолчанию): dry-run, вернуть только diff без записи;
              True: применить изменения на диск и вернуть итоговый diff.
    """
    message: str = Field(..., description="Цель/задача для файлового ассистента")
    session_id: Optional[str] = Field(None, description="ID сессии (для истории диалога)")
    model: str = Field("deepseek-chat", description="Название модели")
    temperature: Optional[float] = Field(0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192)
    apply: bool = Field(False, description="Применить изменения (True) или только показать diff (False)")
    max_cycles: int = Field(8, ge=1, le=20, description="Максимум раундов вызова инструментов")


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


def _is_help_command(user_message: str):
    """Распознаёт команду /help.

    Возвращает строку вопроса после «/help» (может быть пустой) или None, если
    сообщение не является командой /help. Совпадение только по слову /help с
    границей (пробел/конец), чтобы не зацепить «/helper» и т.п.
    """
    t = (user_message or "").strip()
    low = t.lower()
    if low == "/help":
        return ""
    if low.startswith("/help") and len(low) > 5 and low[5].isspace():
        return t[5:].strip()
    return None


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
def _tools_enabled(request: AgentRequest, mcp_clients) -> bool:
    """Инструменты активны: есть хотя бы один клиент, включены запросом и модель поддерживает tools."""
    if not mcp_clients:
        return False
    if not getattr(request, "enable_tools", False):
        return False
    model = (getattr(request, "model", "") or "").strip()
    return model in TOOL_CAPABLE_MODELS or model.lower().startswith(LOCAL_TOOL_CAPABLE_PREFIXES)


def _normalize_mcp_clients(mcp_client, mcp_clients) -> List:
    """Приводит один/несколько MCP-клиентов к дедуплицированному списку."""
    clients = list(mcp_clients or [])
    if mcp_client is not None:
        clients.append(mcp_client)
    seen: List = []
    for c in clients:
        if c is not None and c not in seen:
            seen.append(c)
    return seen


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


def _collect_mcp_tools(mcp_clients):
    """Собирает инструменты со всех MCP-клиентов и строит реестр маршрутизации.

    Возвращает (tools, registry):
      tools     — список MCP-описаний инструментов (inputSchema → формат DeepSeek);
      registry  — dict «имя инструмента» -> объект клиента. Ключ задаёт «что
                  вызвать» (имя инструмента), значение — «куда» (клиент, который
                  его обслуживает). Имя сервера для трейса берётся из
                  client.SERVER_NAME.

    При совпадении имени инструмента у РАЗНЫХ клиентов поднимается ValueError:
    молчаливый «первый выигрывает» скрывает ошибки конфигурации.
    """
    tools: List[dict] = []
    registry: dict = {}
    for client in (mcp_clients or []):
        server = getattr(client, "SERVER_NAME", None) or type(client).__name__
        try:
            client_tools = client.list_tools()
        except Exception as e:
            logger.warning("MCP-клиент %s недоступен (list_tools): %s", server, e)
            continue
        for t in (client_tools or []):
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            if not name:
                continue
            if name in registry:
                if registry[name] is client:
                    continue  # дубль от того же клиента — игнорируем
                existing_server = getattr(registry[name], "SERVER_NAME", None) \
                    or type(registry[name]).__name__
                raise ValueError(
                    f"Конфликт имён MCP-инструмента '{name}': уже зарегистрирован "
                    f"на сервере '{existing_server}', повторно объявлен "
                    f"на сервере '{server}'"
                )
            registry[name] = client
            tools.append(t)
    return tools, registry


def _run_tool_loop(provider: AgentProvider, request: AgentRequest, messages: List[dict],
                   mcp_clients, user_text: str, max_cycles: int = MAX_TOOL_CYCLES):
    """Многокруговой цикл вызова инструментов (нативный function calling DeepSeek).

    Модель получает инструменты со всех MCP-серверов и вызывает их последовательно:
    результат каждого раунда подставляется обратно в диалог, и модель снова решает,
    вызвать ли следующий инструмент. Так данные одного инструмента передаются
    следующему, в том числе между разными серверами (search → summarize →
    save_to_file; get_rate → magic_8_ball и т.п.).

    Цикл завершается, когда модель возвращает ответ без tool_calls (финальный ответ).
    Если лимит max_cycles исчерпан раньше — делается финальный вызов без инструментов,
    чтобы гарантированно получить текстовый ответ.

    max_cycles=1 — частный случай «один раунд вызовов + финальный ответ».

    Возвращает (content, tool_calls_info, data), где tool_calls_info — ПОЛНЫЙ трейс
    вызовов (имя, сервер, аргументы, результат, статус).
    """
    tool_calls_info: List[dict] = []
    data: dict = {}

    tools, registry = _collect_mcp_tools(mcp_clients)
    deepseek_tools = _deepseek_tools(tools)
    if not deepseek_tools:
        data = provider.complete(messages, request, None, user_text)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, tool_calls_info, data

    tool_names = ", ".join(t["function"]["name"] for t in deepseek_tools)
    logger.info("MCP tools доступны модели: %s (max_cycles=%s)", tool_names, max_cycles)
    messages.append({
        "role": "system",
        "content": (
            f"Тебе доступны MCP-инструменты: {tool_names}. "
            "Когда запрос требует фактических данных (курсы валют, гадания, поиск, "
            "суммирование, сохранение файла и т.п.), вызывай подходящий инструмент "
            "и отвечай на основе его результата, а не по памяти. Если нужно несколько "
            "шагов — вызывай инструменты последовательно, передавая результат "
            "предыдущего в аргументы следующего. Когда данных достаточно, дай финальный ответ."
        ),
    })

    cycles = 0
    while cycles < max_cycles:
        data = provider.complete(messages, request, None, user_text, tools=deepseek_tools)
        message = data.get("choices", [{}])[0].get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # Модель больше не запрашивает инструменты — это финальный ответ.
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

            client = registry.get(name)
            server = getattr(client, "SERVER_NAME", None) if client else None
            if client is None:
                text = f"Инструмент '{name}' недоступен."
                ok = False
                error = "нет клиента для инструмента"
            else:
                try:
                    logger.info("Вызов MCP-инструмента '%s' (сервер %s) args=%s", name, server, arguments)
                    res = client.call_tool(name, arguments)
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
                "server": server,
                "arguments": arguments,
                "result": text,
                "ok": ok,
                "error": error,
            })
        cycles += 1

    # Лимит циклов исчерпан, а финального ответа так и не было: добиваем вызовом без tools.
    logger.warning("MCP tool loop достиг лимита циклов %s", max_cycles)
    data = provider.complete(messages, request, None, user_text)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
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


def build_rag_system_block(chunks) -> str:
    """Формирует system-блок с найденными чанками (режим RAG)."""
    lines = [
        "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (RAG) — используй его как первоисточник для ответа. "
        "Отвечай ТОЛЬКО на русском и ТОЛЬКО на основе этого контекста.",
        "ФОРМАТ ОТВЕТА (обязателен):",
        "## Ответ",
        "2–4 предложения; каждое утверждение заканчивай ссылкой [n].",
        "## Источники",
        "- [n] source: <путь> | section: <раздел> | chunk_id: <id>",
        "## Цитаты",
        "- [n] «<дословный фрагмент из чанка>»",
        "Каждое утверждение помечай ссылкой [n] и подтверждай дословной цитатой; "
        "утверждение без цитаты писать нельзя. Источники указывай с source, section "
        "и chunk_id. Если в контексте нет ответа — скажи «не знаю» и попроси уточнить вопрос.",
    ]
    for i, c in enumerate(chunks or [], 1):
        title = c.get("title") or ""
        section = c.get("section") or ""
        src = c.get("source") or ""
        cid = c.get("chunk_id") or ""
        header = f"[{i}] {title}" + (f" — {section}" if section else "")
        lines.append(f"\n{header}\nИсточник: {src}\nchunk_id: {cid}\n{c.get('text', '')}")
    return "\n".join(lines)


# Документация проекта в RAG-корпусе лежит под префиксом docs/agent_* (симлинки
# на agent/README.md и agent/docs/*.md). Всё остальное (articles/*, code/*, чужие
# README) — материалы прошлых заданий, и /help их использовать не должен.
PROJECT_DOC_SOURCE_PREFIX = "docs/agent_"


def _is_project_doc(chunk) -> bool:
    src = (chunk.get("source") or "") if isinstance(chunk, dict) else ""
    return src.startswith(PROJECT_DOC_SOURCE_PREFIX)


def _expand_project_documents(rag_retriever, project_chunks, max_docs=3, max_chars=4000) -> list:
    """Расширяет найденные чанки до ПОЛНЫХ документов проекта.

    Обычный ретрив возвращает лишь топ-N чанков, а полный и подробный ответ
    (например, «выведи схему БД») требует всего документа целиком. Для каждого
    уникального source из project_chunks запрашиваем полный текст через
    rag_retriever.get_document.
    """
    if not project_chunks:
        return []
    getter = getattr(rag_retriever, "get_document", None)
    if getter is None:
        return []
    sources = []
    for c in project_chunks:
        src = c.get("source")
        if src and src not in sources:
            sources.append(src)
    docs = []
    for src in sources[:max_docs]:
        try:
            d = getter(src, strategy="structural")
        except Exception as e:  # noqa: BLE001 — сервис может быть недоступен
            logger.warning("Ошибка получения полного документа %s: %s", src, e)
            continue
        text = (d.get("text") or "").strip() if isinstance(d, dict) else ""
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (обрезано)"
        docs.append({
            "source": src,
            "title": (d.get("title") if isinstance(d, dict) else "") or src,
            "text": text,
        })
    return docs


def build_help_system_block(chunks, git_context, full_docs=None) -> str:
    """Формирует system-блок «ассистент разработчика» для команды /help.

    При наличии full_docs (полные тексты документов проекта) используется их
    содержимое целиком вместо топ-N чанков — так модель отвечает полно и подробно.
    """
    lines = [
        "Ты — ассистент разработчика, который помогает разобраться в устройстве проекта.",
        "Отвечай на русском, полно и подробно, опираясь на приведённые ниже "
        "документацию проекта и контекст git-репозитория. Перечисляй все детали, "
        "которые есть в документации, не сокращай их без необходимости.",
        "Не выдумывай фактов, которых нет в контексте. Если информации не хватает — "
        "скажи, чего именно не хватает.",
        "В конце ответа приведи раздел «Источники» списком использованных файлов и разделов.",
    ]
    if full_docs:
        lines.append("\nДОКУМЕНТАЦИЯ ПРОЕКТА (полные документы):")
        for i, d in enumerate(full_docs or [], 1):
            title = d.get("title") or d.get("source") or ""
            src = d.get("source") or ""
            lines.append(f"\n[{i}] {title}\nИсточник: {src}\n{d.get('text', '')}")
    elif chunks:
        lines.append("\nДОКУМЕНТАЦИЯ ПРОЕКТА:")
        for i, c in enumerate(chunks or [], 1):
            title = c.get("title") or ""
            section = c.get("section") or ""
            src = c.get("source") or ""
            text = str(c.get("text", ""))
            # Крупные чанки (таблицы API/схем) ограничиваем, чтобы блок /help
            # не превышал бюджет контекста (AGENT_MAX_PROMPT_TOKENS).
            if len(text) > 900:
                text = text[:900] + "…"
            header = f"[{i}] {title}" + (f" — {section}" if section else "")
            lines.append(f"\n{header}\nИсточник: {src}\n{text}")
    if git_context:
        lines.append("\nКОНТЕКСТ ПРОЕКТА (git):")
        for key, value in git_context.items():
            value = str(value)
            if len(value) > 1500:
                value = value[:1500] + "\n... (сокращено)"
            lines.append(f"- {key}:\n{value}")
    return "\n".join(lines)


def _find_git_client(clients):
    """Возвращает MCP-клиент git-mcp из списка клиентов (или None)."""
    for c in clients or []:
        if getattr(c, "SERVER_NAME", None) == "git-mcp":
            return c
    return None


def _collect_git_context(clients) -> dict:
    """Детерминированно собирает git-контекст проекта (ветка + список файлов).

    Вызывается напрямую (не через function-calling цикл), чтобы контекст проекта
    всегда попадал в /help, даже если инструменты модели выключены.
    """
    client = _find_git_client(clients)
    if client is None:
        return {}
    context: dict = {}
    # Список файлов ограничиваем (40): полный список из ~170 путей раздувает
    # контекст /help и вылетает за AGENT_MAX_PROMPT_TOKENS, а структура проекта
    # и так описана в документации (RAG).
    for tool, args in (("git_branch", {}), ("list_files", {"max_files": 40})):
        try:
            res = client.call_tool(tool, args)
            if isinstance(res, dict) and res.get("ok"):
                context[tool] = res.get("text", "")
            else:
                context[tool] = "недоступен"
        except Exception as e:  # noqa: BLE001 — git-mcp может быть не запущен
            context[tool] = f"недоступен: {e}"
    return context


# ------------------------------------------------------------
# Ассистент поддержки: парсинг команды, MCP-тикеты, системный блок
# ------------------------------------------------------------
def _parse_support_command(message: str) -> dict:
    """Разбирает сообщение поддержки.

    Возвращает {"question", "ticket_ref", "is_command"}:
      - '/support <вопрос>'                        → is_command=True, ticket_ref=None;
      - '/support --ticket_id=<N> <вопрос>'        → is_command=True, ticket_ref=<N>;
      - обычное сообщение (продолжение диалога)    → is_command=False, ticket_ref=None.
    """
    text = (message or "").strip()
    if not text.startswith("/support"):
        return {"question": text, "ticket_ref": None, "is_command": False}
    rest = text[len("/support"):].strip()
    m = _SUPPORT_TICKET_ARG_RE.search(rest)
    if m:
        ticket_ref = m.group(1)
        question = _SUPPORT_TICKET_ARG_RE.sub("", rest).strip()
    else:
        ticket_ref = None
        question = rest.strip()
    return {"question": question, "ticket_ref": ticket_ref, "is_command": True}


def _find_support_client(clients):
    """Возвращает MCP-клиент support-mcp из списка клиентов (или None)."""
    for c in clients or []:
        if getattr(c, "SERVER_NAME", None) == "support-mcp":
            return c
    return None


def _support_tool_call(client, name: str, arguments: dict):
    """Вызывает инструмент support-mcp и возвращает (data_or_None, error_or_None).

    data — распарсенный JSON из результата инструмента (или {} если тело пустое);
    error — текст ошибки, если инструмент вернул isError или результат не JSON.
    """
    res = client.call_tool(name, arguments)
    if not isinstance(res, dict) or not res.get("ok"):
        err = (res.get("text") if isinstance(res, dict) else "недоступен") or "ошибка"
        return None, str(err)
    text = (res.get("text") or "").strip()
    if not text:
        return {}, None
    try:
        return json.loads(text), None
    except (ValueError, TypeError):
        return {"text": text}, None


def build_support_system_block(ticket: Optional[dict], chunks: list) -> str:
    """Формирует system-блок ассистента поддержки: персона + тикет + FAQ-чанки.

    В отличие от build_rag_system_block (жёсткий формат с [n]-цитатами), здесь
    «человеческий» ответ для клиента: без маркдаун-заголовков и без обязательного
    списка источников.
    """
    lines = [
        "Ты — дружелюбный ассистент поддержки интернет-магазина велосипедов «ВелоДом».",
        "Отвечай на русском языке, вежливо и по существу, простым языком.",
        "Отвечай ТОЛЬКО на основе приведённых ниже данных тикета и базы знаний; "
        "не выдумывай фактов, которых нет в контексте.",
        "Это чат с клиентом, а не отчёт: не используй маркдаун-заголовки и не "
        "добавляй нумерованный список источников в конец ответа.",
    ]
    if ticket:
        lines.append("\nДАННЫЕ ТИКЕТА:")
        lines.append(f"- Номер: {ticket.get('number') or ''}")
        lines.append(f"- Тема: {ticket.get('subject') or ''}")
        lines.append(f"- Статус: {ticket.get('status') or ''}")
        first = ticket.get("first_message")
        if first:
            lines.append(f"- Первое сообщение клиента: {first}")
        customer = ticket.get("customer")
        if isinstance(customer, dict):
            name = customer.get("name") or ""
            email = customer.get("email") or ""
            plan = customer.get("plan") or ""
            if name or email:
                lines.append(
                    f"- Покупатель: {name}" + (f" ({email})" if email else "")
                    + (f", тариф {plan}" if plan else "")
                )
    if chunks:
        lines.append("\nБАЗА ЗНАНИЙ (FAQ и документация):")
        for i, c in enumerate(chunks or [], 1):
            if not isinstance(c, dict):
                continue
            title = c.get("title") or ""
            section = c.get("section") or ""
            header = f"[{i}] {title}" + (f" — {section}" if section else "")
            lines.append(f"\n{header}\n{c.get('text', '')}")
    return "\n".join(lines)


def _render_ticket_summary(ticket: Optional[dict]) -> str:
    """Человекочитаемая сводка тикета (для '/support --ticket_id=<N>' без вопроса)."""
    number = ticket.get("number") or ""
    subject = ticket.get("subject") or ""
    status = ticket.get("status") or ""
    first = ticket.get("first_message") or ""
    lines = [f"Тикет №{number}", f"Тема: {subject}", f"Статус: {status}"]
    if first:
        lines.append(f"Первое сообщение: {first}")
    customer = ticket.get("customer")
    if isinstance(customer, dict):
        name = customer.get("name") or ""
        email = customer.get("email") or ""
        plan = customer.get("plan") or ""
        if name or email:
            lines.append(
                f"Покупатель: {name}" + (f" ({email})" if email else "")
                + (f", тариф {plan}" if plan else "")
            )
    lines.append("")
    lines.append("Задайте вопрос по этому тикету — ассистент ответит с учётом его контекста.")
    return "\n".join(lines)


# ------------------------------------------------------------
# Файловый ассистент (/files)
# ------------------------------------------------------------
def _find_files_client(clients):
    """Возвращает MCP-клиент files-mcp из списка клиентов (или None)."""
    for c in clients or []:
        if getattr(c, "SERVER_NAME", None) == "files-mcp":
            return c
    return None


def _files_tool_call(client, name: str, arguments: dict):
    """Вызывает инструмент files-mcp и возвращает (data_or_None, error_or_None).

    data — распарсенный JSON из результата инструмента (или {} если тело пустое);
    error — текст ошибки, если инструмент вернул isError или результат не JSON.
    """
    res = client.call_tool(name, arguments)
    if not isinstance(res, dict) or not res.get("ok"):
        err = (res.get("text") if isinstance(res, dict) else "недоступен") or "ошибка"
        return None, str(err)
    text = (res.get("text") or "").strip()
    if not text:
        return {}, None
    try:
        return json.loads(text), None
    except (ValueError, TypeError):
        return {"text": text}, None


def build_files_system_block(apply: bool) -> str:
    """Персона файлового ассистента: работа по ЦЕЛИ + режим dry-run/apply."""
    mode = (
        "РЕЖИМ ПРИМЕНИТЬ (apply): после твоих правок изменения будут сохранены на диск "
        "автоматически. Проверь результат через diff перед финальным ответом."
        if apply
        else "РЕЖИМ ПРЕДПРОСМОТР (dry-run): НИЧЕГО не сохраняй на диск. "
        "Только покажи план и итоговый diff (вызови инструмент diff)."
    )
    return "\n".join([
        "Ты — файловый ассистент проекта. Пользователь задаёт ЦЕЛЬ, а не конкретную команду.",
        "Работай с файлами САМ, в несколько шагов:",
        "1) разведай структуру (list_dir) и найди нужные файлы (search_files);",
        "2) прочитай файлы (read_file; большие файлы читай диапазонами строк);",
        "3) проанализируй содержимое и выполни цель;",
        "4) вноси изменения через edit_file (точечная замена) или write_file "
        "(новый файл / полная перезапись).",
        "Инструменты правки НЕ пишут на диск сразу, а возвращают diff — это нормально, "
        "diff не означает, что файл уже сохранён.",
        mode,
        "Отвечай на русском, по существу, без воды.",
        "Не выдумывай содержимое файлов: всегда читай файл перед его правкой.",
        "В конце ответа перечисли: что сделано, какие файлы затронуты, и приведи diff "
        "изменений (если они есть).",
    ])


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


def _trim_messages_to_budget(messages: List[dict], budget_tokens: int):
    """Обрезает старые сообщения ДИАЛОГА, пока контекст не влезет в бюджет.

    Системные блоки (инварианты/профиль/память/RAG), стоящие в начале списка,
    не трогаем — они являются обязательным контекстом. Удаляются только старые
    user/assistant/tool сообщения (самые старые — первыми); ПОСЛЕДНЕЕ сообщение
    (текущий запрос пользователя) не выбрасывается никогда.

    Возвращает список сообщений либо None, если даже последнее сообщение вместе
    с системными блоками не влезает в бюджет (сигнал переполнения контекста).
    """
    if not messages:
        return messages
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        idx += 1
    head = messages[:idx]
    tail = list(messages[idx:])
    if not tail:
        return head
    last = tail[-1]
    tail = tail[:-1]
    while tail and estimate_messages_tokens(head + tail + [last]) > budget_tokens:
        tail = tail[1:]
    if estimate_messages_tokens(head + tail + [last]) > budget_tokens:
        return None  # даже один текущий запрос не влезает — переполнение
    return head + tail + [last]


# ------------------------------------------------------------
# Rate limit (per-IP, скользящее окно) — на уровне агента
# ------------------------------------------------------------
class _SlidingWindowLimiter:
    """Простой потокобезопасный лимитер «скользящее окно» в памяти.

    Подходит для одного процесса (uvicorn). Для нескольких воркеров нужен
    общий redis-бэкенд — в рамках задания достаточно in-memory.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = max(1, int(limit))
        self.window = float(window_seconds)
        self._hits: dict = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


class RateLimitMiddleware:
    """ASGI-middleware: ограничение частоты запросов по IP клиента.

    IP берётся из заголовка X-Forwarded-For (ПОСЛЕДНЕЕ значение — его добавляет
    ближайший прокси, т.е. nginx; спуфить слева бессмысленно) или из socket
    при прямом подключении. OPTIONS (CORS preflight) пропускается без учёта.
    """

    def __init__(self, app, limiter: _SlidingWindowLimiter = None,
                 exclude_paths=None):
        self.app = app
        self.limiter = limiter or _SlidingWindowLimiter(5, 60.0)
        self.exclude_paths = set(exclude_paths or [])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if scope.get("path", "") in self.exclude_paths:
            await self.app(scope, receive, send)
            return
        if not self.limiter.hit(self._client_ip(scope)):
            await self._send_429(send)
            return
        await self.app(scope, receive, send)

    def _client_ip(self, scope) -> str:
        headers = dict(scope.get("headers") or [])
        xff = headers.get(b"x-forwarded-for")
        if xff:
            # Последнее значение — IP ближайшего прокси (реальный клиент).
            return xff.decode("latin-1").split(",")[-1].strip() or "unknown"
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def _send_429(self, send):
        body = json.dumps({
            "detail": "Превышен лимит запросов (rate limit). Повторите позже.",
        }).encode("utf-8")
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"retry-after", str(int(self.limiter.window)).encode("ascii")),
            (b"x-ratelimit-limit", str(self.limiter.limit).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 429, "headers": headers})
        await send({"type": "http.response.body", "body": body})


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
class _AgentRagAdapter:
    """Адаптирует RagClient агента (retrieve → dict) к интерфейсу
    pr_review.rag_client (retrieve → list[chunk])."""

    def __init__(self, rag_retriever):
        self._r = rag_retriever

    def retrieve(self, query: str, corpus: str = "codebase", top_k: int = 5,
                 strategy: str = "structural", mode: str = "baseline") -> list:
        resp = self._r.retrieve(query, corpus=corpus, top_k=top_k,
                                strategy=strategy, mode=mode)
        if isinstance(resp, dict):
            chunks = resp.get("chunks") or []
        else:
            chunks = resp or []
        for c in chunks:
            if isinstance(c, dict):
                c.setdefault("corpus", corpus)
        return chunks


def create_app(provider: AgentProvider, store: Optional[HistoryStorage] = None,
               mcp_client=None, mcp_clients=None, rag_retriever=None) -> FastAPI:
    """Собирает FastAPI-приложение со всеми эндпоинтами на едином пайплайне.

    provider    — реальный (DeepSeek) или детерминированный (Mock) провайдер;
    store       — хранилище (по умолчанию глобальный синглтон; в тестах — отдельный);
    mcp_client  — один клиент MCP-инструментов (list_tools/call_tool);
    mcp_clients — список клиентов MCP-инструментов (если сервисов несколько);
                  None/пусто = без инструментов.
    rag_retriever — объект с методом retrieve(query, ...) -> list[chunk] или
                  dict {chunks: [...], ...}; при request.rag=True найденные
                  чанки добавляются в контекст. None = RAG отключён.
    """
    store = store or default_storage
    clients = _normalize_mcp_clients(mcp_client, mcp_clients)

    # Настройки сервиса (переопределяются через переменные окружения).
    # Rate limit — на уровне агента (per-IP), т.к. Ollama лимиты не умеет.
    rate_limit_per_minute = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "20"))
    max_prompt_tokens = int(os.environ.get("AGENT_MAX_PROMPT_TOKENS", "2048"))

    app = FastAPI(title="AI Agent Service", description="Обработка запросов к модели")
    app.state.rate_limiter = _SlidingWindowLimiter(rate_limit_per_minute, 60.0)
    # Rate limit добавляем ПЕРЕД CORS: тогда CORS окажется внешним слоем и
    # корректно добавит заголовки даже к ответам 429.
    app.add_middleware(
        RateLimitMiddleware,
        limiter=app.state.rate_limiter,
        exclude_paths={"/healthz", "/limits"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "ai-agent", "provider": type(provider).__name__}

    @app.get("/limits")
    def limits():
        return {
            "rate_limit": f"{rate_limit_per_minute}/minute",
            "max_prompt_tokens": max_prompt_tokens,
            "ollama_num_ctx": os.environ.get("OLLAMA_NUM_CTX", "2048"),
            "ollama_max_concurrency": os.environ.get("OLLAMA_MAX_CONCURRENCY", "2"),
        }

    @app.post("/review")
    def review_endpoint(request: ReviewRequest):
        """AI-ревью diff (День 32, подход A — полный агентский пайплайн).

        Получает unified diff, собирает контекст (детерминированный + RAG по
        корпусам docs/codebase) и генерирует ревью через провайдера агента
        (DeepSeek в main.py или локальная Ollama в local_main.py).
        """
        # pr_review лежит рядом с agent/, поэтому добавляем корень проекта в path.
        import sys
        from pathlib import Path as _Path
        _root = _Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        try:
            from pr_review import context as pr_ctx
            from pr_review import prompt as pr_prompt
            from pr_review import rag_client as pr_rag
            from pr_review.diff_utils import changed_files as pr_changed_files
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"pr_review недоступен: {e}")

        diff_text = request.diff or ""
        if not diff_text.strip():
            raise HTTPException(status_code=400, detail="diff пуст")

        files = pr_changed_files(diff_text)

        context_blocks = pr_ctx.build_deterministic_context(files)
        context_text = pr_ctx.render_context(context_blocks)

        rag_chunks: list = []
        rag_text = ""
        if request.use_rag and rag_retriever is not None:
            try:
                adapter = _AgentRagAdapter(rag_retriever)
                rag_chunks = pr_rag.retrieve_review_context(
                    files, diff_text, client=adapter, top_k=4)
                rag_text = pr_rag.render_rag_context(rag_chunks)
            except Exception as e:  # noqa: BLE001 — RAG опционален
                logger.warning("RAG недоступен для /review: %s", e)

        messages = pr_prompt.build_review_messages(diff_text, files, context_text, rag_text)
        agent_req = AgentRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        data = provider.complete(messages, agent_req, task=None,
                                 user_text=diff_text, tools=None)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") \
            if isinstance(data, dict) else str(data)

        return {
            "review_text": content,
            "sections": pr_prompt.split_sections(content),
            "changed_files": files,
            "rag_chunks": rag_chunks,
            "context_sources": [b.title for b in context_blocks],
            "provider": type(provider).__name__,
            "model": request.model,
        }

    @app.post("/support")
    def support_endpoint(request: SupportRequest):
        """Ассистент поддержки пользователей (День 33).

        Флоу:
          1) офтопик-гейт: RAG по корпусу support ДО создания тикета — если
             релевантных чанков нет, отвечаем «нет информации» и тикет не трогаем;
          2) резолв/создание тикета через support-mcp:
             - '/support <вопрос>'         → всегда создаёт новый тикет;
             - '/support --ticket_id=<N>'  → подгружает тикет N;
             - обычное сообщение           → продолжает привязанный тикет сессии
                                              (или создаёт новый, если привязки нет);
          3) привязка «сессия → тикет», сборка системного блока (персона + тикет +
             FAQ) и генерация «человеческого» ответа через провайдера.
        """
        sid = request.session_id or uuid.uuid4().hex
        start = time.time()

        parsed = _parse_support_command(request.message)
        question = parsed["question"].strip()
        is_command = parsed["is_command"]
        ticket_ref = parsed["ticket_ref"]

        support_client = _find_support_client(clients)
        if support_client is None:
            raise HTTPException(
                status_code=503,
                detail="support-mcp не подключён: добавьте SupportMcpClient в create_app.",
            )

        # '/support --ticket_id=<N>' без вопроса: подгружаем тикет и показываем его сводку.
        if ticket_ref and not question:
            data, _err = _support_tool_call(support_client, "get_ticket", {"ticket_id": ticket_ref})
            if data is None:
                raise HTTPException(status_code=404, detail=f"Тикет не найден: {ticket_ref}")
            ticket = data
            ticket_number = ticket.get("number")
            if store.get_session_meta(sid) is None:
                store.create_session(sid, messages=[])
            store.set_support_ticket_id(sid, ticket_number)
            return {
                "session_id": sid,
                "response": _render_ticket_summary(ticket),
                "ticket": ticket,
                "ticket_number": ticket_number,
                "ticket_created": False,
                "abstained": False,
                "sources": [],
                "duration": round(time.time() - start, 3),
            }

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Пустой вопрос: используйте '/support <вопрос>'.",
            )

        # 1. Офтопик-гейт (RAG по корпусу support), до создания тикета.
        rag_chunks: list = []
        rag_sources: list = []
        if rag_retriever is not None:
            try:
                rag_result = rag_retriever.retrieve(
                    question,
                    strategy="structural",
                    top_k=request.rag_top_k,
                    mode=request.rag_mode,
                    corpus="support",
                )
                if isinstance(rag_result, dict) and "chunks" in rag_result:
                    rag_chunks = rag_result.get("chunks") or []
                else:
                    rag_chunks = rag_result or []
                below = bool(isinstance(rag_result, dict) and rag_result.get("below_relevance"))
            except Exception as e:  # noqa: BLE001 — RAG опционален
                logger.warning("Ошибка RAG-ретрива для /support: %s", e)
                rag_chunks = []
                below = False
        else:
            below = False

        rag_sources = [
            {
                "n": i,
                "source": (c.get("source") or ""),
                "title": (c.get("title") or ""),
                "section": (c.get("section") or ""),
                "score": c.get("score"),
            }
            for i, c in enumerate(rag_chunks or [], 1)
        ]

        if not rag_chunks or below:
            return {
                "session_id": sid,
                "response": SUPPORT_ABSTAIN_REPLY,
                "ticket": None,
                "ticket_number": None,
                "ticket_created": False,
                "abstained": True,
                "sources": [],
                "duration": round(time.time() - start, 3),
            }

        # 2. Резолв/создание тикета через MCP.
        ticket: Optional[dict] = None
        ticket_number: Optional[str] = None
        created_now = False

        def _create_args() -> dict:
            args = {"subject": question[:80] or "Вопрос", "message": question}
            if request.customer_email:
                args["customer_email"] = request.customer_email
            return args

        if ticket_ref:
            data, _err = _support_tool_call(support_client, "get_ticket", {"ticket_id": ticket_ref})
            if data is None:
                raise HTTPException(status_code=404, detail=f"Тикет не найден: {ticket_ref}")
            ticket = data
            ticket_number = ticket.get("number")
        elif is_command:
            data, err = _support_tool_call(support_client, "create_ticket", _create_args())
            if data is None:
                raise HTTPException(status_code=502, detail=f"Не удалось создать тикет: {err}")
            ticket = data
            ticket_number = ticket.get("number")
            created_now = True
        else:
            active = store.get_support_ticket_id(sid)
            if active:
                data, _err = _support_tool_call(support_client, "get_ticket", {"ticket_id": active})
                if data is not None:
                    ticket = data
                    ticket_number = ticket.get("number")
            if ticket is None:
                data, err = _support_tool_call(support_client, "create_ticket", _create_args())
                if data is None:
                    raise HTTPException(status_code=502, detail=f"Не удалось создать тикет: {err}")
                ticket = data
                ticket_number = ticket.get("number")
                created_now = True

        # 3. Привязка «сессия → тикет» + история диалога.
        if store.get_session_meta(sid) is None:
            store.create_session(sid, messages=[])
        store.set_support_ticket_id(sid, ticket_number)

        history = store.load(sid) or []
        window = history[-8:] if history else []

        # 4. Сборка контекста и генерация.
        support_block = build_support_system_block(ticket, rag_chunks)
        messages = [{"role": "system", "content": support_block}]
        messages.extend(window)
        messages.append({"role": "user", "content": question})

        agent_req = AgentRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        data = provider.complete(messages, agent_req, task=None, user_text=question, tools=None)
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") \
            if isinstance(data, dict) else str(data)
        answer = (answer or "").strip()

        response = f"Создан тикет №{ticket_number}.\n\n{answer}" if created_now and ticket_number else answer

        # 5. Сохраняем ход диалога.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        store.save(sid, history)

        return {
            "session_id": sid,
            "response": response,
            "ticket": ticket,
            "ticket_number": ticket_number,
            "ticket_created": created_now,
            "abstained": False,
            "sources": rag_sources,
            "duration": round(time.time() - start, 3),
        }

    @app.post("/files")
    def files_endpoint(request: FilesRequest):
        """Файловый ассистент (День 34).

        Принимает задачу на уровне ЦЕЛИ и автономно работает с файлами через files-mcp:
        разведка (list_dir/search_files/read_file) → анализ → правки (write_file/edit_file).

        apply=False (по умолчанию) — dry-run: правки стейджатся, но на диск НЕ пишутся;
        возвращается только итоговый diff. apply=True — правки применяются на диск.
        Применение/предпросмотр выполняются детерминированно (не зависят от того,
        вызовет ли модель apply_changes сама) — это гарантирует воспроизводимость.
        """
        sid = request.session_id or uuid.uuid4().hex
        start = time.time()

        files_client = _find_files_client(clients)
        if files_client is None:
            raise HTTPException(
                status_code=503,
                detail="files-mcp не подключён: добавьте FilesMcpClient в create_app.",
            )

        goal = (request.message or "").strip()
        if not goal:
            raise HTTPException(
                status_code=400,
                detail="Пустая задача: опишите цель для файлового ассистента.",
            )

        # Сброс стейджинга: каждый запрос начинается с чистого листа (воспроизводимость).
        _files_tool_call(files_client, "discard_changes", {})

        block = build_files_system_block(apply=request.apply)
        messages = [{"role": "system", "content": block}]
        messages.append({"role": "user", "content": goal})

        agent_req = AgentRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        content, tool_calls_info, _data = _run_tool_loop(
            provider, agent_req, messages, [files_client], goal, max_cycles=request.max_cycles
        )

        # Детерминированное применение / предпросмотр изменений.
        applied = False
        diff_text = ""
        changed_files: List[str] = []
        if request.apply:
            apply_data, _err = _files_tool_call(files_client, "apply_changes", {})
            if apply_data:
                applied_list = apply_data.get("applied") or []
                if isinstance(applied_list, list):
                    changed_files = [str(p) for p in applied_list]
                diff_text = apply_data.get("diff") or ""
                applied = bool(changed_files)
        else:
            diff_data, _err = _files_tool_call(files_client, "diff", {})
            diff_text = (diff_data.get("diff") if diff_data else "") or ""
            _files_tool_call(files_client, "discard_changes", {})

        # Если apply не вернул список, восстановим затронутые пути из трейса инструментов.
        if not changed_files:
            seen: List[str] = []
            for tc in tool_calls_info or []:
                if tc.get("ok") and tc.get("name") in ("write_file", "edit_file"):
                    p = (tc.get("arguments") or {}).get("path")
                    if p and p not in seen:
                        seen.append(str(p))
            changed_files = seen

        return {
            "session_id": sid,
            "response": (content or "").strip(),
            "applied": applied,
            "diff": diff_text,
            "changed_files": changed_files,
            "tool_calls": tool_calls_info,
            "duration": round(time.time() - start, 3),
        }

    @app.post("/agent")
    def agent_endpoint(request: AgentRequest):
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

            # 2.5 Классификация запроса (задача / продолжение / подтверждение / чат / help).
            help_question = _is_help_command(user_text)
            is_help = help_question is not None
            is_new_task = _detect_task(user_text)
            is_continue = _is_continue(user_text)

            task = None
            mode = "chat"
            was_paused = False
            resume_instruction = None

            if is_help:
                mode = "help"
            elif is_new_task:
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

            # 4.1 RAG: дополнить контекст найденными чанками (режим «с RAG»).
            rag_context = None
            rag_abstain = False
            rag_sources = []
            help_context = None
            if request.rag and rag_retriever is not None and user_text:
                try:
                    rag_result = rag_retriever.retrieve(
                        user_text,
                        strategy="structural",
                        top_k=request.rag_top_k,
                        mode=request.rag_mode,
                        top_k_candidates=request.rag_top_k_candidates,
                        min_score=request.rag_min_score,
                    )
                    if isinstance(rag_result, dict) and "chunks" in rag_result:
                        rag_chunks = rag_result.get("chunks") or []
                    else:
                        rag_chunks = rag_result or []
                    below = bool(isinstance(rag_result, dict)
                                 and rag_result.get("below_relevance"))
                    if not rag_chunks or below:
                        # Детерминированный режим «не знаю» при слабом контексте.
                        rag_abstain = True
                    else:
                        rag_block = build_rag_system_block(rag_chunks)
                        request_messages.insert(0, {"role": "system", "content": rag_block})
                        context_tokens += estimate_tokens(rag_block)
                    rag_context = {
                        "chunks": rag_chunks,
                        "sources": list(dict.fromkeys(c.get("source") for c in rag_chunks)),
                        "mode": request.rag_mode,
                        "below_relevance": below,
                    }
                    rag_sources = [
                        {
                            "n": i,
                            "source": (c.get("source") or ""),
                            "title": (c.get("title") or ""),
                            "section": (c.get("section") or ""),
                            "chunk_id": (c.get("chunk_id") or ""),
                            "score": c.get("score"),
                        }
                        for i, c in enumerate(rag_chunks or [], 1)
                    ]
                    if isinstance(rag_result, dict):
                        for key in ("rewritten_query", "original_query", "candidates",
                                    "kept", "dropped", "max_score", "relevance_threshold"):
                            if key in rag_result:
                                rag_context[key] = rag_result[key]
                except Exception as e:
                    logger.warning("Ошибка RAG-ретрива: %s", e)
                    rag_context = {"error": str(e)}

            # 4.15 Команда /help: документация проекта (RAG) + контекст git (MCP).
            if is_help:
                help_chunks = []
                help_sources = []
                full_docs = []
                rag_query = help_question or "структура проекта"
                if rag_retriever is not None:
                    try:
                        # Забираем больше кандидатов и оставляем только документацию
                        # проекта (docs/agent_*): общий корпус RAG (NLP-статьи и т.п.)
                        # семантически вытесняет ответы о самом проекте.
                        rag_result = rag_retriever.retrieve(
                            rag_query,
                            strategy="structural",
                            top_k=max(20, request.rag_top_k * 4),
                            mode="baseline",
                        )
                        if isinstance(rag_result, dict) and "chunks" in rag_result:
                            raw_chunks = rag_result.get("chunks") or []
                        else:
                            raw_chunks = rag_result or []
                        project_chunks = [c for c in raw_chunks if _is_project_doc(c)]
                        help_chunks = (project_chunks or raw_chunks)[:request.rag_top_k]
                        full_docs = _expand_project_documents(rag_retriever, project_chunks)
                        help_sources = [
                            {
                                "n": i,
                                "source": (c.get("source") or ""),
                                "title": (c.get("title") or ""),
                                "section": (c.get("section") or ""),
                                "chunk_id": (c.get("chunk_id") or ""),
                            }
                            for i, c in enumerate(help_chunks or [], 1)
                        ]
                    except Exception as e:
                        logger.warning("Ошибка RAG-ретрива для /help: %s", e)
                git_context = _collect_git_context(clients)
                help_block = build_help_system_block(help_chunks, git_context, full_docs)
                request_messages.insert(0, {"role": "system", "content": help_block})
                context_tokens += estimate_tokens(help_block)
                help_context = {
                    "question": rag_query,
                    "sources": help_sources,
                    "git": git_context,
                }

            # 4.2 Guard контекста: не превышать лимит (num_ctx) — обрезаем старый
            #     диалог; если даже текущий запрос не влезает — явная ошибка 413.
            if context_tokens > max_prompt_tokens:
                trimmed = _trim_messages_to_budget(request_messages, max_prompt_tokens)
                if trimmed is None:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Контекст превышает лимит {max_prompt_tokens} токенов "
                            f"(фактически {context_tokens}). Сократите запрос или "
                            f"увеличьте AGENT_MAX_PROMPT_TOKENS / OLLAMA_NUM_CTX."
                        ),
                    )
                request_messages = trimmed
                context_tokens = estimate_messages_tokens(request_messages)

            # 5. Генерация + переход конечного автомата.
            tools_active = _tools_enabled(request, clients)
            tool_calls_info: List[dict] = []
            data: dict = {}
            if refused:
                content = build_refusal_reply(refusal_inv, {
                    "invariant_name": refusal_inv.get("name"),
                    "reason": refusal_reason or refusal_inv.get("rationale"),
                })
            elif rag_abstain:
                content = (
                    "## Ответ\nЯ не знаю ответа на этот вопрос — в базе знаний не "
                    "нашлось достаточно релевантной информации.\n\nПожалуйста, "
                    "уточните вопрос или переформулируйте его."
                )
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
            elif mode == "help":
                data = provider.complete(request_messages, request, None, user_text)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                if tools_active:
                    content, tool_calls_info, data = _run_tool_loop(
                        provider, request, request_messages, clients, user_text
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

            # 6. Сохраняем ответ в историю (с источниками, если был RAG).
            assistant_msg = {"role": "assistant", "content": content}
            if request.rag and not refused:
                assistant_msg["sources"] = rag_sources
            if refused:
                assistant_msg["invariant_refusal"] = True
            conversation.append(assistant_msg)
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
                "rag_context": rag_context,
                "rag_abstained": rag_abstain,
                "sources": rag_sources,
                "help_context": help_context,
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
    def suggest_endpoint(session_id: str, request: SuggestRequest):
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
