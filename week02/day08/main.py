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

# Реальный вызов DeepSeek
def call_deepseek(messages, request: AgentRequest):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": request.model,
        "messages": messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }
    if request.stop is not None:
        payload["stop"] = request.stop
    # top_k не поддерживается DeepSeek официально, но можно передать, если нужно
    if request.top_k:
        payload["top_k"] = request.top_k

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
    return response.json()

# Основной обработчик
@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    # Восстанавливаем существующую сессию либо создаём новую
    agent_id = request.session_id or str(uuid.uuid4())
    print("session_id=" + agent_id)
    start_time = time.time()

    try:
        # 1. Загрузка сохранённого контекста (если сессия уже существует)
        conversation = storage.load(agent_id)

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

        # 2. Вызов модели
        if API_KEY == "sk-1234567890":
            data = mock_deepseek_response(
                conversation, request.model, request.temperature,
                request.top_k, request.top_p, request.stop, request.max_tokens
            )
        else:
            data = call_deepseek(conversation, request)

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
        # 3) Токены всей истории диалога — эталон из usage.prompt_tokens
        #    (вся переписка, отправленная модели), иначе оценка по всем сообщениям.
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = estimate_messages_tokens(conversation)
        total_tokens = prompt_tokens + completion_tokens

        # 3. Сохраняем ответ ассистента в историю сессии
        conversation.append({"role": "assistant", "content": content})
        storage.save(agent_id, conversation)

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
                "response_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
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
    return {"session_id": session_id, "messages": messages}


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