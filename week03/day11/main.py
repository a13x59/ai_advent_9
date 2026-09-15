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

from storage import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

app = FastAPI(title="AI Agent Service", description="Обработка запросов к DeepSeek")

# Разрешаем CORS для клиента
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация API DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1234567890")  # Заглушка, если ключ такой – используем мок

# Цены DeepSeek (пример, за 1M токенов) — используются для оценки стоимости.
INPUT_PRICE_PER_M = 0.14    # $0.14 за 1M входных токенов
OUTPUT_PRICE_PER_M = 0.28   # $0.28 за 1M выходных токенов

# ------------------------------------------------------------
# Стратегии управления контекстом
# ------------------------------------------------------------
# sliding_window — в запрос идут только последние N сообщений, остальное отбрасывается.
# sticky_facts    — блок фактов (ключ-значение) + последние N сообщений.
# branching       — диалог ветвится от checkpoint, ветки продолжаются независимо.
STRATEGY_SLIDING_WINDOW = "sliding_window"
STRATEGY_STICKY_FACTS = "sticky_facts"
STRATEGY_BRANCHING = "branching"
STRATEGIES = {STRATEGY_SLIDING_WINDOW, STRATEGY_STICKY_FACTS, STRATEGY_BRANCHING}
DEFAULT_STRATEGY = STRATEGY_SLIDING_WINDOW
DEFAULT_WINDOW_SIZE = 10

FACTS_MAX_TOKENS = 512    # лимит токенов для извлечения фактов
FACTS_TEMPERATURE = 0.2   # низкая температура — предсказуемый результат

FACTS_SYSTEM = "Ты — ассистент, который ведёт память фактов диалога."

FACTS_PROMPT = (
    "Извлеки из сообщения пользователя важные факты и верни их СТРОГО в виде "
    "JSON-объекта (ключ — категория, значение — краткая формулировка). "
    "Категории могут быть такими: цель, ограничения, предпочтения, решения, "
    "договорённости, контекст и т.п. Не выдумывай лишнего. Объедини новые факты "
    "с уже известными, ничего не теряя."
)


# Модель для входных данных
class AgentRequest(BaseModel):
    messages: List[dict] = Field(..., description="История сообщений (role, content)")
    session_id: Optional[str] = Field(None, description="ID сессии для сохранения/восстановления контекста")
    strategy: str = Field(DEFAULT_STRATEGY, description="Стратегия управления контекстом: sliding_window | sticky_facts | branching")
    window_size: int = Field(DEFAULT_WINDOW_SIZE, ge=1, description="Число последних сообщений в окне (N)")
    model: str = Field(..., description="Название модели (например, deepseek-chat)")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(0, ge=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192)


class BranchRequest(BaseModel):
    checkpoint: Optional[int] = Field(None, description="Индекс сообщения, от которого ветвимся (по умолчанию — конец диалога)")


class SwitchRequest(BaseModel):
    branch_id: str = Field(..., description="ID ветки, на которую переключаемся")


# ============================================================
# Подсчёт токенов
# ============================================================
# Точное число токенов зависит от токенизатора конкретной модели (BPE).
# DeepSeek использует словарь ~128K токенов, поэтому без тяжёлых зависимостей
# (transformers / tiktoken) считаем приблизительно:
#   • CJK-иероглифы и слоговая азбука (хирагана/катакана/хангыль) — ~1 токен на символ;
#   • слова (латиница/кириллица/цифры) — ~1.3 токена на слово (из-за BPE-разбиения);
#   • прочие символы (пунктуация, эмодзи и т.п.) — ~1 токен на символ.
# Когда DeepSeek возвращает usage (prompt_tokens / completion_tokens), эти
# значения используются как эталон для истории диалога и ответа модели.

_CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"   # CJK Unified Ideographs
    r"\u3040-\u30FF\uAC00-\uD7AF]"                  # хирагана/катакана/хангыль
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


# Мок-ответ для заглушки
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


# Универсальный вызов DeepSeek (без привязки к AgentRequest)
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
# Стратегия 2: Sticky Facts / Key-Value Memory
# ============================================================
def _call_tokens(input_text, output_text, usage=None):
    """Токены вспомогательного вызова (вход, выход).

    Если API вернул usage — берём его эталонные значения, иначе оцениваем
    по тексту промпта и тексту ответа.
    """
    usage = usage or {}
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    if inp is None:
        inp = estimate_tokens(input_text)
    if out is None:
        out = estimate_tokens(output_text)
    return inp, out


def _tokens_money(input_tokens, output_tokens) -> float:
    """Стоимость вызова в долларах по ценам DeepSeek."""
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
    # Модель может обернуть JSON в пояснительный текст — берём первую {...} скобку.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


# Наивное извлечение фактов для мока (без реального API).
def mock_extract_facts(existing, user_message):
    facts = dict(existing or {})
    t = (user_message or "").lower()
    if any(k in t for k in ("цель", "задача", "нужно", "хочу")):
        facts["цель"] = user_message
    if any(k in t for k in ("нельзя", "огранич", "лимит", "не ")):
        facts["ограничения"] = user_message
    if any(k in t for k in ("предпочит", "нравится", "лучше", "люблю")):
        facts["предпочтения"] = user_message
    if any(k in t for k in ("решил", "решение", "давай", "сделаем")):
        facts["решения"] = user_message
    if any(k in t for k in ("договорились", "согласен", "договор")):
        facts["договорённости"] = user_message
    facts["последний_запрос"] = user_message
    return facts


def extract_facts(existing_facts, user_message, model):
    """Обновляет блок фактов по сообщению пользователя.

    Возвращает (facts_dict, input_tokens, output_tokens).
    """
    if API_KEY == "sk-1234567890":
        facts = mock_extract_facts(existing_facts, user_message)
        return facts, estimate_tokens(user_message), estimate_tokens(json.dumps(facts, ensure_ascii=False))

    prompt_text = (
        FACTS_PROMPT
        + "\n\nУже известные факты (JSON):\n"
        + json.dumps(existing_facts or {}, ensure_ascii=False)
        + "\n\nСообщение пользователя:\n"
        + (user_message or "")
    )
    msgs = [
        {"role": "system", "content": FACTS_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    data = call_deepseek_raw(
        msgs, model=model, temperature=FACTS_TEMPERATURE, max_tokens=FACTS_MAX_TOKENS
    )
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    facts = parse_json_object(content) or dict(existing_facts or {})
    inp, out = _call_tokens(prompt_text, content, data.get("usage"))
    return facts, inp, out


def facts_to_text(facts) -> str:
    """Превращает словарь фактов в системную подсказку для модели."""
    if not facts:
        return ""
    lines = [f"- {k}: {v}" for k, v in facts.items()]
    return "Важные факты из диалога (память «ключ-значение»):\n" + "\n".join(lines)


# ============================================================
# Формирование контекста запроса в зависимости от стратегии
# ============================================================
def build_request_messages(strategy, conversation, facts, window_size):
    """Собирает сообщения, которые будут отправлены модели.

    • sliding_window: последние N сообщений;
    • sticky_facts:    факты (system) + последние N сообщений;
    • branching:       полная история активной ветки.
    """
    if strategy == STRATEGY_STICKY_FACTS:
        messages = []
        facts_text = facts_to_text(facts)
        if facts_text:
            messages.append({"role": "system", "content": facts_text})
        messages.extend(conversation[-window_size:])
        return messages
    if strategy == STRATEGY_BRANCHING:
        return list(conversation)
    # По умолчанию — скользящее окно.
    return conversation[-window_size:]


def build_context_summary(strategy, conversation, facts, window_size, session_id):
    """Сводка по контексту для UI (зависит от выбранной стратегии)."""
    total = len(conversation)
    info = {
        "strategy": strategy,
        "total_messages": total,
        "window_size": window_size,
    }
    if strategy == STRATEGY_STICKY_FACTS:
        info["facts"] = facts or {}
        info["facts_count"] = len(facts or {})
        info["sent_messages"] = min(total, window_size)
        info["discarded_messages"] = max(0, total - window_size)
    elif strategy == STRATEGY_BRANCHING:
        current = storage.get_current_branch_id(session_id)
        info["branch"] = {"id": current}
        info["branches"] = storage.list_branches(session_id)
        info["sent_messages"] = total
        info["discarded_messages"] = 0
    else:
        info["sent_messages"] = min(total, window_size)
        info["discarded_messages"] = max(0, total - window_size)
    return info


# Основной обработчик
@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    agent_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    strategy = request.strategy if request.strategy in STRATEGIES else DEFAULT_STRATEGY
    window_size = request.window_size if request.window_size >= 1 else DEFAULT_WINDOW_SIZE

    try:
        # 1. Восстановление или создание сессии. Полная история всегда лежит в БД.
        conversation = storage.load(agent_id)
        facts = storage.load_facts(agent_id) or {}

        current_user_message = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            None,
        )

        if conversation is None:
            # Новая сессия: берём историю, присланную клиентом.
            conversation = list(request.messages)
            storage.create_session(agent_id, strategy=strategy, window_size=window_size, messages=conversation)
        else:
            # Продолжаем диалог: добавляем новое сообщение пользователя к
            # активной ветке, как будто агент не выключался.
            storage.set_session_meta(agent_id, strategy, window_size)
            if current_user_message and (not conversation or conversation[-1] != current_user_message):
                conversation.append(current_user_message)

        # 2. Стратегия sticky_facts: обновляем блок фактов после каждого
        #    сообщения пользователя.
        facts_cost_money = 0.0
        if strategy == STRATEGY_STICKY_FACTS and current_user_message:
            try:
                facts, f_inp, f_out = extract_facts(facts, current_user_message.get("content", ""), request.model)
                storage.save_facts(agent_id, facts)
                facts_cost_money += _tokens_money(f_inp, f_out)
            except Exception as e:
                logger.warning("Ошибка обновления фактов, оставляю прежние: %s", e)

        # 3. Формируем контекст согласно стратегии.
        request_messages = build_request_messages(strategy, conversation, facts, window_size)

        # Токены: полная история (активной ветки) vs реально отправленный контекст.
        raw_context_tokens = estimate_messages_tokens(conversation)
        context_tokens = estimate_messages_tokens(request_messages)

        # 4. Вызов модели.
        if API_KEY == "sk-1234567890":
            data = mock_deepseek_response(
                request_messages, request.model, request.temperature,
                request.top_k, request.top_p, request.stop, request.max_tokens
            )
        else:
            data = call_deepseek(request_messages, request)

        duration = time.time() - start_time

        # Извлечение ответа
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        # --- Подсчёт токенов ---
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = context_tokens
        # Вся история диалога (активной ветки), откалиброванная реальным счётом.
        if context_tokens > 0:
            history_tokens = round(raw_context_tokens * (prompt_tokens / context_tokens))
        else:
            history_tokens = raw_context_tokens
        total_tokens = prompt_tokens + completion_tokens

        # 5. Сохраняем ответ ассистента в полную историю активной ветки.
        conversation.append({"role": "assistant", "content": content})
        storage.save(agent_id, conversation)
        storage.set_session_meta(agent_id, strategy, window_size)

        # Сводка по контексту для UI (вместо баннера сжатия).
        context_summary = build_context_summary(strategy, conversation, facts, window_size, agent_id)

        # Расчёт стоимости: основной запрос + вызов извлечения фактов.
        cost = (
            (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
            + (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
            + facts_cost_money
        )

        return {
            "session_id": agent_id,
            "response": content,
            "strategy": strategy,
            "usage": {
                "prompt_tokens": prompt_tokens,      # что отправлено модели
                "response_tokens": completion_tokens, # ответ модели
                "history_tokens": history_tokens,    # вся история активной ветки
                "total_tokens": total_tokens,        # prompt + completion
            },
            "context": context_summary,
            "duration": round(duration, 3),
            "cost": round(cost, 6),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Получение сохранённой истории сессии (для восстановления контекста на клиенте)
@app.get("/agent/{session_id}")
async def get_agent_history(session_id: str):
    meta = storage.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    messages = storage.load(session_id)
    facts = storage.load_facts(session_id) or {}
    branches = storage.list_branches(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
        "strategy": meta["strategy"],
        "window_size": meta["window_size"],
        "facts": facts,
        "current_branch": meta["current_branch"],
        "branches": branches,
    }


# Список веток сессии (обзор/переключение)
@app.get("/agent/{session_id}/branches")
async def list_session_branches(session_id: str):
    if storage.get_session_meta(session_id) is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {
        "branches": storage.list_branches(session_id),
        "current_branch": storage.get_current_branch_id(session_id),
    }


# Создание двух веток от checkpoint (стратегия branching)
@app.post("/agent/{session_id}/branch")
async def create_branches(session_id: str, request: BranchRequest):
    messages = storage.load(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    total = len(messages)
    checkpoint = total if request.checkpoint is None else request.checkpoint
    checkpoint = max(0, min(checkpoint, total))
    prefix = messages[:checkpoint]

    # Две независимые ветки, начинающиеся с одной и той же контрольной точки.
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


# Переключение активной ветки
@app.post("/agent/{session_id}/switch")
async def switch_branch(session_id: str, request: SwitchRequest):
    if not storage.switch_branch(session_id, request.branch_id):
        raise HTTPException(status_code=404, detail="Ветка не найдена")
    return {
        "current_branch": request.branch_id,
        "messages": storage.load_branch(request.branch_id),
        "branches": storage.list_branches(session_id),
    }


# Список сохранённых сессий (обзор/отладка)
@app.get("/agents")
async def list_agents():
    return {"sessions": storage.list_sessions()}


# Удаление сессии (сброс контекста)
@app.delete("/agent/{session_id}")
async def delete_agent_history(session_id: str):
    if not storage.delete(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"deleted": True, "session_id": session_id}


# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
