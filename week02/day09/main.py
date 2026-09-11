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

# ------------------------------------------------------------
# Настройки компрессии контекста
# ------------------------------------------------------------
# Последние N сообщений храним «как есть», всё что старше — сворачиваем
# в резюме (summary), которое подставляется в запрос вместо полной истории.
KEEP_LAST_N = 5
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
def call_deepseek_raw(messages, model, temperature=1.0, top_p=1.0, max_tokens=4096,
                      top_k=0, stop=None):
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


def summarize_batch(messages, model) -> str:
    """Сжимает список сообщений в одно резюме (без учёта предыдущего)."""
    if API_KEY == "sk-1234567890":
        return mock_summarize(messages)
    text = format_messages_for_summary(messages)
    msgs = [
        {"role": "system", "content": "Ты — ассистент, который кратко пересказывает историю диалога."},
        {"role": "user", "content": SUMMARY_PROMPT + "\n\n" + text},
    ]
    data = call_deepseek_raw(
        msgs, model=model, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
    )
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def merge_summary(existing_summary, messages, model) -> str:
    """Объединяет старое резюме и новый фрагмент диалога в одно резюме."""
    if API_KEY == "sk-1234567890":
        return mock_merge_summary(existing_summary, messages)
    text = format_messages_for_summary(messages)
    msgs = [
        {"role": "system", "content": "Ты — ассистент, который обновляет резюме диалога."},
        {
            "role": "user",
            "content": (
                MERGE_PROMPT
                + "\n\nПредыдущее резюме:\n" + existing_summary
                + "\n\nНовый фрагмент:\n" + text
            ),
        },
    ]
    data = call_deepseek_raw(
        msgs, model=model, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
    )
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def ensure_compressed(conversation, summary, summarized_count, model):
    """Сворачивает в резюме всё, что вышло за окно последних KEEP_LAST_N сообщений.

    Возвращает (summary, summarized_count, folded_count), где folded_count —
    сколько сообщений было свёрнуто именно в этом вызове (0, если нечего).
    """
    if len(conversation) <= KEEP_LAST_N:
        return summary, summarized_count, 0

    overflow = conversation[:-KEEP_LAST_N]
    folded = len(overflow) - summarized_count
    if folded <= 0:
        return summary, summarized_count, 0

    new_messages = overflow[summarized_count:]
    if summary:
        summary = merge_summary(summary, new_messages, model)
    else:
        summary = summarize_batch(new_messages, model)

    logger.info(
        "Контекст сжат: свёрнуто сообщений — %d (всего в резюме — %d)",
        folded, len(overflow),
    )
    return summary, len(overflow), folded


def build_request_messages(conversation, summary):
    """Формирует контекст для модели: резюме + последние N сообщений «как есть»."""
    messages = []
    if summary:
        messages.append({
            "role": "system",
            "content": "Краткое резюме предыдущего диалога (вместо полной истории):\n" + summary,
        })
    messages.extend(conversation[-KEEP_LAST_N:])
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
        total_saved_tokens = summary_state.get("total_saved_tokens", 0)

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

        # 2. Компрессия перед вызовом: сворачиваем всё, что вышло за окно
        #    последних KEEP_LAST_N сообщений, в резюме.
        compressed_this_turn = False
        use_compression = True
        try:
            summary, summarized_count, folded = ensure_compressed(
                conversation, summary, summarized_count, request.model
            )
            compressed_this_turn = folded > 0
        except Exception as e:
            # Если сжатие не удалось — отправляем полную историю, чтобы не терять контекст.
            logger.warning("Ошибка сжатия контекста, отправляю полную историю: %s", e)
            use_compression = False

        if use_compression:
            request_messages = build_request_messages(conversation, summary)
        else:
            request_messages = list(conversation)

        # Токены: полная история (без сжатия) vs то, что реально отправлено.
        raw_context_tokens = estimate_messages_tokens(conversation)
        compressed_context_tokens = estimate_messages_tokens(request_messages)
        saved_tokens = max(0, raw_context_tokens - compressed_context_tokens) if use_compression else 0
        total_saved_tokens += saved_tokens

        # 3. Вызов модели
        if API_KEY == "sk-1234567890":
            data = mock_deepseek_response(
                request_messages, request.model, request.temperature,
                request.top_k, request.top_p, request.stop, request.max_tokens
            )
        else:
            data = call_deepseek(request_messages, request)

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
        if use_compression:
            try:
                summary, summarized_count, folded_after = ensure_compressed(
                    conversation, summary, summarized_count, request.model
                )
                compressed_this_turn = compressed_this_turn or folded_after > 0
            except Exception as e:
                logger.warning("Ошибка сжатия после ответа: %s", e)

        # 6. Сохраняем полную историю и резюме (отдельно) в БД.
        storage.save(agent_id, conversation)
        if summary:
            storage.save_summary(agent_id, summary, summarized_count, total_saved_tokens)

        # Расчёт стоимости (пример для deepseek-chat, цены за 1M токенов)
        # Цены могут меняться, для демонстрации используем приблизительные
        input_price_per_m = 0.14   # $0.14 за 1M input токенов
        output_price_per_m = 0.28  # $0.28 за 1M output токенов
        cost = (prompt_tokens / 1_000_000) * input_price_per_m + (completion_tokens / 1_000_000) * output_price_per_m

        duration = time.time() - start_time

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
                "saved_tokens": saved_tokens,
            },
            "compression": {
                "applied": bool(summary) and use_compression,
                "compressed_this_turn": compressed_this_turn,
                "summarized_messages": summarized_count,
                "kept_messages": min(KEEP_LAST_N, len(conversation)),
                "total_messages": len(conversation),
                "saved_tokens": saved_tokens,
                "total_saved_tokens": total_saved_tokens,
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
