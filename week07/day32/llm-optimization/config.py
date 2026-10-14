"""Конфигурация бенчмарка оптимизации локальной LLM (День 29).

Бенчмарк прогоняет контрольные вопросы rag/data/golden.json через локальную
модель Ollama (llama3.1) и оценивает качество независимым судьёй на DeepSeek.
Retrieval (ретрив чанков) выполняется ОДИН раз и кэшируется, поэтому контекст
для всех конфигураций идентичен — различия относятся только к генератору.

Настройки берутся из корневого .env (репозитория), не зависят от рабочей папки.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
load_dotenv(REPO_ROOT / ".env", override=True)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

GOLDEN_PATH = REPO_ROOT / "rag" / "data" / "golden.json"
DATA_DIR = ROOT / "data"
CHUNKS_CACHE = DATA_DIR / "chunks.json"
REPORTS_DIR = ROOT / "reports"

# RAG-ретрив: локальный сервис (rag/src/server.py) или прямой импорт не нужен —
# чанки дёргаем по HTTP и кэшируем.
RETRIEVAL_URL = os.environ.get("RAG_RETRIEVAL_URL", "http://127.0.0.1:8891").rstrip("/")
RETRIEVAL_STRATEGY = "structural"
RETRIEVAL_TOP_K = 5

# Теги моделей (три уровня квантования).
MODEL_Q3 = "llama3.1:8b-instruct-q3_K_M"
MODEL_Q4 = "llama3.1:8b"                # текущий, уже установлен (Q4_K_M)
MODEL_Q8 = "llama3.1:8b-instruct-q8_0"

QUANTS = [
    ("q3_K_M", MODEL_Q3),
    ("q4_K_M", MODEL_Q4),
    ("q8_0", MODEL_Q8),
]

# Базовые параметры инференса (дефолт «до оптимизации»).
# top_k = 0 означает «top-k выключен» (llama.cpp трактует 0 как отсутствие
# ограничения), top_k = 40 — включён на стандартном значении.
BASE_OPTIONS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "max_tokens": 1024,   # маппится в options.num_predict
    "num_ctx": 4096,
}

# Сетки для OFAT-перебора (один параметр за раз).
SWEEP_TEMPERATURE = [0.0, 0.2, 1.0]
SWEEP_TOP_K = [0, 40]
SWEEP_MAX_TOKENS = [256, 512, 1024, 2048]
SWEEP_NUM_CTX = [2048, 4096, 8192, 16384]

# Вопросы для грубого прогона (разнообразные темы) и весь набор (10).
COARSE_QUESTION_IDS = ["q1", "q2", "q3", "q5", "q10"]

TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
