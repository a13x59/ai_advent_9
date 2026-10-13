#!/usr/bin/env python3
"""Нагрузочный тест приватного AI-сервиса: стабильность + rate limit + max context.

Проверяет требования задания:
  1) доступ к модели по сети (/healthz, /limits);
  2) стабильность при нескольких запросах (N параллельных /agent → p50/p95/p99);
  3) базовые ограничения (rate limit → 429; max context → 413/успешный ответ).

Запуск:
    python scripts/load_test.py --base http://127.0.0.1:8002 --n 8 --concurrency 3
    python scripts/load_test.py --base http://<IP> --api-key KEY --n 8 --concurrency 3

Примечание про rate limit: лимит per-IP. При тесте НАПРЯМУЮ против uvicorn
(127.0.0.1:8002) скрипт подставляет заголовок X-Forwarded-For для изоляции фаз
(стабильность/rate-limit). За nginx заголовок перезаписывается реальным IP,
поэтому изоляция не работает — там лимит применяется к вашему IP честно.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import time

import requests

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


def _headers(api_key, xff=None):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-Api-Key"] = api_key
    if xff:
        h["X-Forwarded-For"] = xff
    return h


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def one_chat(base: str, api_key: str, i: int, xff: str) -> tuple[int, float, dict]:
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": f"Ответь одним коротким предложением на число {i}"}],
        "max_tokens": 128,
    }
    t0 = time.perf_counter()
    r = requests.post(base + "/agent", json=payload, headers=_headers(api_key, xff), timeout=300)
    dt = time.perf_counter() - t0
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r.status_code, dt, body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE_URL", "http://127.0.0.1:8002"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--n", type=int, default=8, help="число запросов в фазе стабильности")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"==> Базовый URL: {base}")
    print(f"==> Модель: {DEFAULT_MODEL}")

    # 0) Здоровье и лимиты.
    for path in ("/healthz", "/limits"):
        r = requests.get(base + path, headers=_headers(args.api_key), timeout=30)
        print(f"[GET {path}] {r.status_code}: {r.text[:240]}")

    # 1) Стабильность под параллельной нагрузкой (изолируем IP, чтобы не
    #    пересекаться с проверкой rate limit ниже).
    print(f"\n==> Стабильность: {args.n} запросов, concurrency={args.concurrency}")
    lat, statuses, errors = [], {}, 0
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_chat, base, args.api_key, i, "10.0.0.1") for i in range(args.n)]
        for f in concurrent.futures.as_completed(futs):
            try:
                code, dt, body = f.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  [error] {type(e).__name__}: {e}")
                continue
            statuses[code] = statuses.get(code, 0) + 1
            lat.append(dt)
    wall = time.perf_counter() - t0
    if lat:
        print(f"  wall={wall:.1f}s  throughput={args.n / wall:.2f} req/s")
        print(f"  p50={percentile(lat, 50):.2f}s  p95={percentile(lat, 95):.2f}s  "
              f"p99={percentile(lat, 99):.2f}s  max={max(lat):.2f}s")
    print(f"  статусы={statuses}  ошибки={errors}")

    # 2) Rate limit: шлём сверх лимита с отдельного IP и ждём 429.
    print("\n==> Rate limit (ожидаем 429)")
    limits = requests.get(base + "/limits", headers=_headers(args.api_key), timeout=30).json()
    per_min = int(str(limits.get("rate_limit", "20/minute")).split("/")[0])
    got_429 = False
    for i in range(per_min + 3):
        payload = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "ping"}]}
        r = requests.post(base + "/agent", json=payload, headers=_headers(args.api_key, "10.0.0.2"), timeout=300)
        if r.status_code == 429:
            got_429 = True
            print(f"  429 получен после {i + 1}-го запроса "
                  f"(Retry-After={r.headers.get('Retry-After')}, X-RateLimit-Limit={r.headers.get('X-RateLimit-Limit')})")
            break
    if not got_429:
        print("  ВНИМАНИЕ: 429 не получен — проверьте AGENT_RATE_LIMIT_PER_MINUTE")

    # 3) Max context: длинный промпт (изолированный IP).
    print("\n==> Max context (длинный промпт)")
    long_text = ("повтори слово тест " * 1500).strip()
    r = requests.post(
        base + "/agent",
        json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": long_text}], "max_tokens": 64},
        headers=_headers(args.api_key, "10.0.0.3"),
        timeout=300,
    )
    print(f"  длинный промпт -> HTTP {r.status_code}")


if __name__ == "__main__":
    main()
