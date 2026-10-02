# RAG-индексация документов

Учебный пайплайн индексации документов для курса по AI. Собирает набор
русскоязычных документов, разбивает их на чанки двумя способами, строит
эмбеддинги, сохраняет векторный индекс (FAISS + SQLite) и сравнивает две
стратегии chunking по метрикам поиска.

## Что делает пайплайн

1. **Загрузка** (`src/loader.py`) — читает документы из `data/raw/`
   (статьи `.txt`, README `.md`, код `.py`, PDF `.pdf`) и приводит их к
   единому виду `Document` (текст + метаданные `source`, `title`, `file_type`).
2. **Chunking** (`src/chunkers.py`) — две стратегии:
   - `fixed` — по фиксированному размеру: окно в ~100 слов с перекрытием 20 слов;
   - `structural` — по структуре: заголовки markdown (`#`, `##`), заголовки
     вики (`== Раздел ==`, `=== Подраздел ===`), для кода — границы функций/классов;
     длинные секции дорезаются по абзацам без разрыва абзаца.
   Каждый чанк получает метаданные: `chunk_id`, `source`, `title`, `section`,
   `strategy`, `char_start`, `char_end`.
3. **Эмбеддинги** (`src/embedder.py`) — локально через sentence-transformers,
   модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   (384-мерные векторы, поддержка русского языка, без API-ключа).
4. **Индекс** (`src/indexer.py`) — FAISS (`IndexFlatIP`, cosine-поиск по
   нормализованным векторам) + SQLite (текст и метаданные каждого чанка).
5. **Оценка** (`src/evaluate.py`) — Recall@5, MRR@10, nDCG@10 на тест-сете
   из 14 русскоязычных запросов с релевантностью на уровне документа.

## Структура проекта

```
rag/
├── data/
│   └── raw/              # исходные документы (как скачаны)
│       ├── articles/     # статьи русской Википедии
│       ├── docs/         # README open-source проектов
│       ├── code/         # исходный код
│       └── sources.md    # перечень источников и объём
├── src/
│   ├── models.py         # Document / Chunk
│   ├── loader.py         # загрузка документов
│   ├── chunkers.py       # 2 стратегии chunking + метаданные
│   ├── embedder.py       # sentence-transformers
│   ├── indexer.py        # FAISS + SQLite
│   ├── evaluate.py       # метрики + тест-сет
│   └── pipeline.py       # CLI-точка входа
├── scripts/
│   └── fetch_docs.py     # сбор документов (одноразовый)
├── indexes/              # *.faiss, *.db, *.json (результат)
├── reports/
│   ├── comparison.md     # сравнение стратегий
│   └── metrics.json      # детальные метрики
└── requirements.txt
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Модель эмбеддингов скачивается автоматически при первом запуске и
кэшируется в `models_cache/` внутри проекта.

## Использование

```bash
# собрать документы (опционально, если data/raw/ пуст)
python scripts/fetch_docs.py

# собрать индексы обеих стратегий
python -m src.pipeline run

# посчитать метрики и записать reports/metrics.json
python -m src.pipeline eval

# задать вопрос индексу
python -m src.pipeline query -s structural "Что такое word2vec?"

# список документов
python -m src.pipeline list
```

## Результаты

Корпус: 19 документов (~168 тыс. символов, ~90 страниц). Итог по двум
стратегиям (полное сравнение — [`reports/comparison.md`](reports/comparison.md),
метрики по запросам — `reports/metrics.json`):

| Метрика | fixed | structural |
|---|---|---|
| Чанков | 265 | 396 |
| Средняя длина / std | 783 / 201 симв. | 422 / 554 симв. |
| Recall@5 | 0.222 | 0.232 |
| MRR@10 | 0.810 | 0.827 |
| nDCG@10 | 0.509 | 0.481 |

Качество поиска у стратегий близкое; главное различие — структура чанков:
fixed даёт равномерные чанки (но без `section` и с перекрытием), structural —
смысловые чанки по разделам с заполненным `section`, но с большой дисперсией
длины.

## RAG-ответы (вопрос → чанки → объединение → LLM)

Поверх индекса реализована цепочка генерации ответа:

- `src/retrieval.py` — `retrieve(query, strategy, top_k)`: поиск релевантных
  чанков (переиспользует `load_index` + `Embedder`).
- `src/qa.py` — `answer_plain(question)` (без RAG) и `answer_rag(question)`
  (ретрив → промпт с контекстом → DeepSeek). Требует `DEEPSEEK_API_KEY`.

```bash
python -m src.compare --strategy structural --top-k 5
```

`compare.py` прогоняет 10 контрольных вопросов из `data/golden.json` (у каждого —
`question`, `expectation`, `sources`, `relevant`) и для каждого генерирует два
ответа. Затем `judge.py` (LLM-as-judge) оценивает каждый ответ 0–2 против
`expectation` и пишет:

- `reports/rag_answers.json` — сырые ответы и найденные чанки;
- `reports/rag_comparison.json` — те же данные с оценками;
- `reports/rag_comparison.md` — сводная таблица и детали.

Оценку можно пересчитать отдельно: `python -m src.judge`.

## Реранкинг и фильтрация (День 23)

Второй этап после поиска + query rewrite + сравнение режимов. Параметры —
в `src/config.py` (`MIN_SCORE`, `TOP_K_CANDIDATES`, `TOP_K_FINAL`, `LEX_WEIGHT`,
`MMR_LAMBDA`, `MODES`).

### Режимы

`src/compare.py` прогоняет контрольные вопросы в нескольких режимах:

| Режим | Что делает |
|---|---|
| `baseline` | одностадийный поиск, без второго этапа |
| `rewrite` | LLM-переформулировка запроса (DeepSeek) → поиск |
| `filter` | топ-K кандидатов → отсечение по порогу similarity |
| `rewrite+filter` | rewrite + фильтр по порогу |
| `rerank` | топ-K кандидатов → лексическая эвристика + fusion + MMR |

### Механика второго этапа

- **filter** — абсолютный порог `MIN_SCORE` (по умолчанию 0.5, подбирается
  sweep-ом) + опциональный относительный запас `MAX_SCORE_MARGIN` для
  устойчивости к масштабу score на коротких запросах.
- **rerank** — лексическое пересечение слов запроса и чанка (`lexical_score`),
  fusion с косинусом (`LEX_WEIGHT`) + MMR-разнообразие (`MMR_LAMBDA`), чтобы
  убрать дубли из одного документа и пересортировать выдачу.

### Команды

```bash
# метрики ретрива + precision@5 / precision@10
python -m src.pipeline eval

# сетка порогов (precision/recall) → reports/threshold_sweep.json
python -m src.pipeline sweep

# матрица режимов + LLM-judge → reports/rerank_comparison.md, reports/rag_modes.json
python -m src.compare --strategy structural --top-k 5
```

### Отчёты

- `reports/threshold_sweep.json` — precision/recall по сетке порогов;
- `reports/rag_modes_answers.json` / `reports/rag_modes.json` — сырые ответы и оценки по режимам;
- `reports/rerank_comparison.md` — сводная таблица режимов (score, лучше/хуже baseline, hit@k, precision, kept).

## HTTP-сервис ретрива

Для агента (`agent/rag_client.py`) ретрив вынесен в отдельный сервис, чтобы не
тянуть faiss/sentence-transformers в venv агента:

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8891
```

- `GET /health` — статус;
- `POST /retrieve` — `{"query": "...", "strategy": "structural", "top_k": 5,
  "top_k_candidates": 20, "min_score": 0.5, "mode": "baseline"}` →
  `{"chunks": [...], "mode": ..., "candidates": ..., "kept": ...,
  "dropped": [...], "rewritten_query": ...}`. `mode` ∈
  `baseline|rewrite|filter|rewrite+filter|rerank`; режимы `rewrite*`
  переформулируют запрос через DeepSeek.

Агент включает режим RAG флагом `rag: true` в теле `/agent` (или чекбоксом
«📚 RAG» на веб-странице). В теле запроса можно задать `rag_mode`,
`rag_top_k`, `rag_top_k_candidates`, `rag_min_score`; на веб-странице этим
управляет группа «📚 Параметры RAG (реранкинг и фильтрация)». Найденные чанки
подставляются в контекст, а в ответе приходит поле `rag_context` с
использованными источниками и метаданными второго этапа (mode, kept, dropped,
rewritten_query).
