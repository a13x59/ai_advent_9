# Оптимизация локальной LLM (День 29)

Бенчмарк локальной модели **llama3.1 (Ollama)** под кейс **RAG-QA**
(контрольные вопросы `../rag/data/golden.json`). Тестируется **только
локальная модель**; качество оценивает независимый судья на DeepSeek.

Сравниваются: параметры инференса (temperature / top_k / max_tokens /
num_ctx), три кванта (q3_K_M / q4_K_M / q8_0) и два промпт-шаблона.

## Как это устроено

- **Retrieval кэшируется один раз** (`data/chunks.json`) через локальный
  RAG-сервис (`http://127.0.0.1:8891/retrieve`). Контекст у всех конфигураций
  одинаковый — различия относятся только к генератору.
- Вызов модели — нативный `/api/chat` Ollama (серверные тайминги
  `eval_duration`, `prompt_eval_count`, `load_duration`), а не OpenAI-совместимый
  эндпоинт, поэтому скорость честная.
- Ресурсы: RSS процессов Ollama (`psutil`) + загруженная модель (`/api/ps`).
- Качество: LLM-as-judge на DeepSeek (score 0–2 против `expectation`).

## Установка

```bash
cd llm-optimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Требуется: запущенный Ollama (`http://127.0.0.1:11434`), RAG-сервис ретрива
(`uvicorn src.server:app --port 8891` из `../rag`), и `DEEPSEEK_API_KEY` в
корневом `.env`.

## Запуск

```bash
# 1. Кэш чанков (один раз)
python bench.py fetch-chunks

# 2. Baseline (текущая конфигурация)
python bench.py run --label baseline --models llama3.1:8b --template baseline \
    --questions all --runs 3 --out baseline

# 3. Подбор параметров (OFAT + финалисты)
python bench.py sweep --runs 3

# 4. Сравнение квантов
python bench.py quant --runs 3

# 5. A/B промпт-шаблона
python bench.py prompt-ab --runs 3
```

Отчёты — в `reports/` (`.md` + `.json`).

## Важно

Выигрышная комбинация параметров пишется только в отчёт
(`reports/params_sweep.*`) и **не подставляется как дефолт** — применение в
`agent/` и `rag/` выполняется отдельным согласованным шагом.
