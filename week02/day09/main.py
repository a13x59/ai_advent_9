# main.py
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
API_KEY = "sk-1234567890"  # Заглушка, если ключ такой – используем мок

# Цены DeepSeek (пример, за 1M токенов) — используются для оценки стоимости
# как основного запроса, так и вызовов саммаризации.
INPUT_PRICE_PER_M = 0.14    # $0.14 за 1M входных токенов
OUTPUT_PRICE_PER_M = 0.28   # $0.28 за 1M выходных токенов

# ------------------------------------------------------------
# Настройки компрессии контекста
# ------------------------------------------------------------
# KEEP_LAST_N — сколько самых СВЕЖИХ сообщений держим «как есть».
# Как только несуммаризированных сообщений становится больше KEEP_LAST_N,
# самые старые KEEP_LAST_N из них сворачиваются в резюме (summary),
# которое подставляется в запрос вместо этой части истории.
KEEP_LAST_N = 10
SUMMARY_MAX_TOKENS = 512   # лимит токенов для генерации резюме
SUMMARY_TEMPERATURE = 0.2  # низкая температура — более предсказуемое резюме

SUMMARY_PROMPT = (
    "Сожми следующий фрагмент диалога в краткое резюме. "
    "Сохрани ключевые факты, имена, числа, решения и незавершённые задачи. "
    "Пиши на том же языке, что и диалог, кратко и без воды."
)

MERGE_PROMPT = (
    "Объедини предыдущее резюме диалога и новый фрагмент в одно обновлённое "
    "краткое резюме. Сохрани ключевые факты, имена, числа, решения и "
    "незавершённые задачи. Пиши на том же языке, что и диалог, кратко и без воды."
)


# Модель для входных данных
class AgentRequest(BaseModel):
    messages: List[dict] = Field(..., description="История сообщений (role, content)")
    session_id: Optional[str] = Field(None, description="ID сессии для сохранения/восстановления контекста")
    model: str = Field(..., description="Название модели (например, deepseek-chat)")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(0, ge=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192)

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


# Мок-резюме для заглушки (без реального API).
# Это «наивное» сжатие: ограничиваем размер резюме, чтобы наглядно показать
# экономию токенов в режиме без реального ключа.
def _mock_summary_from(existing, messages):
    fragments = []
    if existing:
        fragments.append(existing.replace("Резюме: ", "").rstrip("…"))
    for m in messages:
        content = str(m.get("content", "")).strip().replace("\n", " ")
        role = "П" if m.get("role") == "user" else "А"
        fragments.append(f"{role}: {content[:40]}")
    text = " | ".join(fragments)
    if len(text) > 220:
        text = text[:220] + "…"
    return "Резюме: " + text


def mock_summarize(messages):
    return _mock_summary_from(None, messages)


def mock_merge_summary(existing, messages):
    return _mock_summary_from(existing, messages)


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
# Компрессия контекста (управление историей)
# ============================================================
def format_messages_for_summary(messages) -> str:
    """Превращает список сообщений в текст для передачи в промпт резюме."""
    lines = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _summary_tokens(input_text, summary_text, usage=None):
    """Считает токены вызова саммаризации отдельно: (вход, выход).

    Если API вернул usage — берём его эталонные значения, иначе оцениваем
    по тексту промпта и тексту резюме.
    """
    usage = usage or {}
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    if inp is None:
        inp = estimate_tokens(input_text)
    if out is None:
        out = estimate_tokens(summary_text)
    return inp, out


def _tokens_money(input_tokens, output_tokens) -> float:
    """Стоимость вызова в долларах по ценам DeepSeek."""
    return (input_tokens / 1_000_000) * INPUT_PRICE_PER_M + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M


def summarize_batch(messages, model):
    """Сжимает список сообщений в одно резюме (без учёта предыдущего).

    Возвращает (summary, input_tokens, output_tokens) — токены, потраченные
    на сам вызов саммаризации (раздельно вход и выход).
    """
    if API_KEY == "sk-1234567890":
        summary = mock_summarize(messages)
        return summary, estimate_messages_tokens(messages), estimate_tokens(summary)
    text = format_messages_for_summary(messages)
    prompt_text = SUMMARY_PROMPT + "\n\n" + text
    msgs = [
        {"role": "system", "content": "Ты — ассистент, который кратко пересказывает историю диалога."},
        {"role": "user", "content": prompt_text},
    ]
    data = call_deepseek_raw(
        msgs, model=model, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
    )
    summary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    inp, out = _summary_tokens(prompt_text, summary, data.get("usage"))
    return summary, inp, out


def merge_summary(existing_summary, messages, model):
    """Объединяет старое резюме и новый фрагмент диалога в одно резюме.

    Возвращает (summary, input_tokens, output_tokens).
    """
    if API_KEY == "sk-1234567890":
        summary = mock_merge_summary(existing_summary, messages)
        inp = estimate_tokens(existing_summary) + estimate_messages_tokens(messages)
        return summary, inp, estimate_tokens(summary)
    text = format_messages_for_summary(messages)
    user_content = (
        MERGE_PROMPT
        + "\n\nПредыдущее резюме:\n" + existing_summary
        + "\n\nНовый фрагмент:\n" + text
    )
    msgs = [
        {"role": "system", "content": "Ты — ассистент, который обновляет резюме диалога."},
        {"role": "user", "content": user_content},
    ]
    data = call_deepseek_raw(
        msgs, model=model, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
    )
    summary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    inp, out = _summary_tokens(user_content, summary, data.get("usage"))
    return summary, inp, out


def ensure_compressed(conversation, summary, summarized_count, model):
    """Поддерживает инвариант: сырых (не свернутых) сообщений не больше KEEP_LAST_N.

    Сырая часть диалога — это самые свежие сообщения conversation[summarized_count:].
    Как только их становится больше KEEP_LAST_N, самые старые KEEP_LAST_N из них
    сворачиваются в резюме (одним батчем). Цикл нужен на случай, если за один
    запрос пришло сразу много сообщений.

    Возвращает (summary, summarized_count, folded_count, input_tokens, output_tokens):
    - folded_count — сколько сообщений свёрнуто в этом вызове (0, если нечего);
    - input_tokens/output_tokens — токены, потраченные на саммаризацию (0, если нечего).
    """
    folded = 0
    inp_total = 0
    out_total = 0

    while len(conversation) - summarized_count > KEEP_LAST_N:
        raw = conversation[summarized_count:]
        to_compress = raw[:KEEP_LAST_N]  # самые старые из сырой части

        if summary:
            summary, inp, out = merge_summary(summary, to_compress, model)
        else:
            summary, inp, out = summarize_batch(to_compress, model)

        summarized_count += len(to_compress)
        folded += len(to_compress)
        inp_total += inp
        out_total += out

    if folded:
        logger.info(
            "Контекст сжат: свёрнуто сообщений — %d (всего в резюме — %d), "
            "затраты на саммаризацию — %d токенов",
            folded, summarized_count, inp_total + out_total,
        )
    return summary, summarized_count, folded, inp_total, out_total


def build_request_messages(conversation, summary, summarized_count):
    """Формирует контекст для модели: резюме + свежие (ещё не свёрнутые) сообщения.

    В резюме свёрнуты самые старые summarized_count сообщений, поэтому «как есть»
    отправляются только сообщения conversation[summarized_count:] (их не больше
    KEEP_LAST_N).
    """
    messages = []
    if summary:
        messages.append({
            "role": "system",
            "content": "Краткое резюме предыдущего диалога (вместо полной истории):\n" + summary,
        })
    messages.extend(conversation[summarized_count:])
    return messages


# Основной обработчик
@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    # Восстанавливаем существующую сессию либо создаём новую
    agent_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        # 1. Загрузка сохранённого контекста (если сессия уже существует):
        #    полная история + отдельное резюме (summary).
        conversation = storage.load(agent_id)
        summary_state = storage.load_summary(agent_id) or {}
        summary = summary_state.get("summary")
        summarized_count = summary_state.get("summarized_count", 0)
        total_saved_tokens = summary_state.get("total_saved_tokens", 0)      # чистая экономия
        total_summary_tokens = summary_state.get("total_summary_tokens", 0)  # затраты на саммаризацию

        # Текущее сообщение пользователя, пришедшее в этом запросе
        current_user_message = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            None,
        )

        if conversation is None:
            # Новая сессия: берём историю, присланную клиентом
            conversation = list(request.messages)
        else:
            # Продолжаем диалог: добавляем новое сообщение пользователя
            # к сохранённой истории, как будто агент не выключался.
            if current_user_message and (not conversation or conversation[-1] != current_user_message):
                conversation.append(current_user_message)

        # 2. Компрессия перед вызовом: если свежих (не свёрнутых) сообщений
        #    стало больше KEEP_LAST_N — сворачиваем самые старые из них в резюме.
        compressed_this_turn = False
        summary_tokens_this_turn = 0       # токены (вход+выход) на саммаризацию за ход
        summary_cost_money_this_turn = 0.0  # деньги за саммаризацию за ход
        use_compression = True
        try:
            summary, summarized_count, folded, s_inp, s_out = ensure_compressed(
                conversation, summary, summarized_count, request.model
            )
            compressed_this_turn = folded > 0
            summary_tokens_this_turn += s_inp + s_out
            summary_cost_money_this_turn += _tokens_money(s_inp, s_out)
        except Exception as e:
            # Если сжатие не удалось — отправляем полную историю, чтобы не терять контекст.
            logger.warning("Ошибка сжатия контекста, отправляю полную историю: %s", e)
            use_compression = False

        if use_compression:
            request_messages = build_request_messages(conversation, summary, summarized_count)
        else:
            request_messages = list(conversation)

        # Токены: полная история (без сжатия) vs то, что реально отправлено.
        raw_context_tokens = estimate_messages_tokens(conversation)
        compressed_context_tokens = estimate_messages_tokens(request_messages)
        gross_saved_tokens = max(0, raw_context_tokens - compressed_context_tokens) if use_compression else 0
        # Чистая экономия за ход = выигрыш на основном запросе − затраты на саммаризацию.
        saved_tokens = gross_saved_tokens - summary_tokens_this_turn
        total_saved_tokens += saved_tokens
        total_summary_tokens += summary_tokens_this_turn

        # 3. Вызов модели
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
        # 1) Токены текущего запроса — новое сообщение пользователя.
        request_tokens = (
            estimate_tokens(current_user_message.get("content", ""))
            if current_user_message else 0
        )
        # 2) Токены ответа модели — эталон из usage API, иначе оценка по тексту.
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
        # 3) Токены контекста, отправленного модели (резюме + последние N сообщений).
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = compressed_context_tokens
        total_tokens = prompt_tokens + completion_tokens

        # 4. Сохраняем ответ ассистента в полную историю сессии
        conversation.append({"role": "assistant", "content": content})

        # 5. Компрессия после ответа: ответ ассистента мог вытеснить ещё одно
        #    сообщение за окно последних N — досворачиваем его в резюме.
        #    NB: затраты этого второго вызова тоже учитываем в экономии.
        if use_compression:
            try:
                summary, summarized_count, folded_after, s_inp, s_out = ensure_compressed(
                    conversation, summary, summarized_count, request.model
                )
                compressed_this_turn = compressed_this_turn or folded_after > 0
                s_cost = s_inp + s_out
                summary_tokens_this_turn += s_cost
                summary_cost_money_this_turn += _tokens_money(s_inp, s_out)
                saved_tokens -= s_cost
                total_saved_tokens -= s_cost
                total_summary_tokens += s_cost
            except Exception as e:
                logger.warning("Ошибка сжатия после ответа: %s", e)

        # 6. Сохраняем полную историю и резюме (отдельно) в БД.
        storage.save(agent_id, conversation)
        if summary:
            storage.save_summary(
                agent_id, summary, summarized_count,
                total_saved_tokens, total_summary_tokens,
            )

        # Расчёт стоимости: основной запрос + вызовы саммаризации.
        cost = (
            (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
            + (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
            + summary_cost_money_this_turn
        )

        return {
            "id": agent_id,
            "session_id": agent_id,
            "response": content,
            "usage": {
                "request_tokens": request_tokens,
                "history_tokens": prompt_tokens,
                "raw_history_tokens": raw_context_tokens,
                "response_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "saved_tokens": saved_tokens,              # чистая экономия за ход
                "gross_saved_tokens": gross_saved_tokens,  # выигрыш на основном запросе
                "summary_tokens": summary_tokens_this_turn, # затраты на саммаризацию за ход
            },
            "compression": {
                "applied": bool(summary) and use_compression,
                "compressed_this_turn": compressed_this_turn,
                "summarized_messages": summarized_count,       # сколько сообщений в резюме
                "kept_messages": len(conversation) - summarized_count,  # свежих «как есть»
                "total_messages": len(conversation),           # полная история в БД
                "saved_tokens": saved_tokens,               # чистая экономия за ход
                "gross_saved_tokens": gross_saved_tokens,
                "summary_tokens": summary_tokens_this_turn,
                "total_saved_tokens": total_saved_tokens,   # чистая экономия за сессию
                "total_summary_tokens": total_summary_tokens, # затраты на саммаризацию за сессию
                "summary": summary,
            },
            "duration": round(duration, 3),
            "cost": round(cost, 6)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Получение сохранённой истории сессии (для восстановления контекста на клиенте)
@app.get("/agent/{session_id}")
async def get_agent_history(session_id: str):
    messages = storage.load(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    summary_state = storage.load_summary(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
        "summary": summary_state["summary"] if summary_state else None,
        "summarized_count": summary_state["summarized_count"] if summary_state else 0,
        "total_saved_tokens": summary_state["total_saved_tokens"] if summary_state else 0,
        "total_summary_tokens": summary_state["total_summary_tokens"] if summary_state else 0,
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
