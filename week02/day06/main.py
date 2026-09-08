# main.py
import os
import time
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from fastapi.middleware.cors import CORSMiddleware

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
    model: str = Field(..., description="Название модели (например, deepseek-chat)")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(0, ge=0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192)

# Мок-ответ для заглушки
def mock_deepseek_response(messages, model, temperature, top_k, top_p, stop, max_tokens):
    user_message = messages[-1]["content"] if messages else ""
    return {
        "id": f"mock-{uuid.uuid4()}",
        "choices": [{"message": {"role": "assistant", "content": f"Эхо (mock): {user_message}"}}],
        "usage": {
            "prompt_tokens": len(user_message) // 4,
            "completion_tokens": len(user_message) // 4,
            "total_tokens": len(user_message) // 2,
        }
    }

# Реальный вызов DeepSeek
def call_deepseek(request: AgentRequest):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": request.model,
        "messages": request.messages,
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
    agent_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Если ключ – заглушка, используем мок
        if API_KEY == "sk-1234567890":
            data = mock_deepseek_response(
                request.messages, request.model, request.temperature,
                request.top_k, request.top_p, request.stop, request.max_tokens
            )
        else:
            data = call_deepseek(request)

        # Извлечение ответа
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Расчёт стоимости (пример для deepseek-chat, цены за 1M токенов)
        # Цены могут меняться, для демонстрации используем приблизительные
        input_price_per_m = 0.14   # $0.14 за 1M input токенов
        output_price_per_m = 0.28  # $0.28 за 1M output токенов
        cost = (prompt_tokens / 1_000_000) * input_price_per_m + (completion_tokens / 1_000_000) * output_price_per_m

        duration = time.time() - start_time

        return {
            "id": agent_id,
            "response": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "duration": round(duration, 3),
            "cost": round(cost, 6)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: uvicorn main:app --host 0.0.0.0 --port 8000