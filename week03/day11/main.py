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
    "working — данные ТЕКУЩЕЙ задачи (ключ, значение, kind, state):\n"
    "  kind ∈ {goal, constraint, todo, result, context, note}; "
    "state ∈ {pending, in_progress, done, blocked} или null.\n"
    "long_term — профиль/решения/знания (ключ, значение, kind, tags):\n"
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


class BranchRequest(BaseModel):
    checkpoint: Optional[int] = Field(None, description="Индекс сообщения, от которого ветвимся")


class SwitchRequest(BaseModel):
    branch_id: str = Field(..., description="ID ветки, на которую переключаемся")


class MemoryOpsRequest(BaseModel):
    ops: List[dict] = Field(..., description="Список операций памяти (save/delete/move)")


class SuggestRequest(BaseModel):
    message: Optional[str] = Field(None, description="Сообщение для анализа (по умолчанию — последнее от пользователя)")
    model: str = Field("deepseek-chat", description="Модель для генерации предложений")


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


def mock_deepseek_response(messages, model, temperature, top_k, top_p, stop, max_tokens):
    user_message = messages[-1]["content"] if messages else ""
    content = f"Эхо (mock): {user_message}"
    prompt_tokens = estimate_messages_tokens(messages)
    completion_tokens = estimate_tokens(content)
    return {
        "id": f"mock-{uuid.uuid4()}",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }


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


def build_request_messages(strategy, conversation, working, long_term, window_size):
    """Собирает сообщения модели из трёх слоёв памяти.

    • краткосрочная  — окно последних N (или полная ветка для branching);
    • рабочая        — системный блок (всегда);
    • долговременная — системный блок (для sticky_facts — только факты kind=fact).
    """
    if strategy == STRATEGY_STICKY_FACTS:
        block = build_memory_system_block(working, long_term, long_term_kinds={"fact"})
    else:
        block = build_memory_system_block(working, long_term)

    messages = []
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
        # 1. Восстановление слоёв памяти.
        conversation = storage.load(agent_id)
        working = storage.load_working(agent_id)
        long_term = storage.load_long_term()

        current_user_message = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            None,
        )

        if conversation is None:
            conversation = list(request.messages)
            storage.create_session(agent_id, strategy=strategy, window_size=window_size, messages=conversation)
        else:
            storage.set_session_meta(agent_id, strategy, window_size)
            if current_user_message and (not conversation or conversation[-1] != current_user_message):
                conversation.append(current_user_message)

        # 2. Явные операции памяти — пользователь выбирает, что и куда сохранять.
        memory_ops_applied = []
        for op in request.memory_ops or []:
            try:
                memory_ops_applied.append(storage.apply_memory_op(agent_id, op, source="manual"))
            except Exception as e:
                logger.warning("Ошибка применения memory_op %s: %s", op, e)
        working = storage.load_working(agent_id)
        long_term = storage.load_long_term()

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

        # 4. Формируем контекст из трёх слоёв.
        request_messages = build_request_messages(strategy, conversation, working, long_term, window_size)

        raw_context_tokens = estimate_messages_tokens(conversation)
        context_tokens = estimate_messages_tokens(request_messages)

        # 5. Вызов модели.
        if API_KEY == "sk-1234567890":
            data = mock_deepseek_response(
                request_messages, request.model, request.temperature,
                request.top_k, request.top_p, request.stop, request.max_tokens
            )
        else:
            data = call_deepseek(request_messages, request)

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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "history_tokens": history_tokens,
                "total_tokens": total_tokens,
            },
            "context": context_summary,
            "memory": {"working": working, "long_term": long_term},
            "pending_memory": pending_memory,
            "memory_ops_applied": memory_ops_applied,
            "duration": round(duration, 3),
            "cost": round(cost, 6),
        }

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
    working = storage.load_working(session_id)
    long_term = storage.load_long_term()
    suggestion, s_inp, s_out = suggest_memory(working, long_term, message.get("content", ""), request.model)
    return {
        "pending_memory": flatten_suggestions(suggestion),
        "cost": round(_tokens_money(s_inp, s_out), 6),
    }


# Глобальная долговременная память (общая для всех сессий).
@app.get("/memory/longterm")
async def list_long_term():
    return {"long_term": storage.load_long_term()}


# ============================================================
# История, ветки, сессии
# ============================================================
@app.get("/agent/{session_id}")
async def get_agent_history(session_id: str):
    meta = storage.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    messages = storage.load(session_id)
    working = storage.load_working(session_id)
    long_term = storage.load_long_term()
    branches = storage.list_branches(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
        "strategy": meta["strategy"],
        "window_size": meta["window_size"],
        "working": working,
        "long_term": long_term,
        "current_branch": meta["current_branch"],
        "branches": branches,
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


# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
