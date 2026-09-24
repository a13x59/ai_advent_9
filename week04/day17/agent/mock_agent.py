# mock_agent.py
"""Тестовый агент с детерминированным провайдером (мок), БЕЗ сети и ключа.

Повторяет тот же пайплайн, что и main.py (тот же agent_core.create_app), но
вместо реального DeepSeek использует детерминированные заглушки:

  • complete        — эмулирует ответ модели + маркер [STATE] (жёсткая логика
                      конечного автомата, как раньше жила в _mock_reply/_task_state_update);
  • suggest_memory  — наивное извлечение предложений памяти;
  • check_invariants — сверка с deny_examples инвариантов.

Запуск вручную: uvicorn mock_agent:app --port 8001
"""
import json
import re
import uuid
from typing import List, Optional

from agent_core import (
    AgentProvider,
    AgentRequest,
    create_app,
    estimate_tokens,
    _is_continue,
)
from storage import HistoryStorage

DEFAULT_MOCK_DB = None  # None => HistoryStorage() возьмёт DB_PATH (agent_history.db)


class MockProvider(AgentProvider):
    """Детерминированный провайдер: то, что раньше было мок-ветками в main.py."""

    def complete(self, messages: List[dict], request: AgentRequest,
                 task: Optional[dict], user_text: str) -> dict:
        if task is None:
            return self._response(_mock_chat_reply(user_text))
        reply, marker = _mock_turn(task, user_text)
        content = reply
        if marker is not None:
            content += "\n[STATE] " + json.dumps(marker, ensure_ascii=False)
        return self._response(content)

    @staticmethod
    def _response(content: str) -> dict:
        return {
            "id": "mock-" + uuid.uuid4().hex,
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }

    def suggest_memory(self, working, long_term, user_message, model):
        result = mock_suggest_memory(working, long_term, user_message)
        return result, estimate_tokens(user_message), estimate_tokens(json.dumps(result, ensure_ascii=False))

    def check_invariants(self, invariants, user_text, model) -> dict:
        return mock_check_invariants(invariants, user_text)


def _mock_chat_reply(user_text: str) -> str:
    if _is_continue(user_text):
        return "Эхо (mock). Нет активной задачи — опишите, что нужно сделать."
    return f"Эхо (mock): {user_text}"


def _mock_plan(user_text: str) -> List[dict]:
    """Наивный план шагов для мока (без реального API)."""
    t = (user_text or "").lower()
    if any(k in t for k in ("калькулятор", "calc")):
        labels = [
            "Спроектировать интерфейс калькулятора",
            "Написать код арифметики",
            "Проверить на тестовых примерах",
        ]
    elif any(k in t for k in ("тест", "test")):
        labels = [
            "Определить сценарии тестов",
            "Написать тесты",
            "Прогнать и зафиксировать результат",
        ]
    else:
        labels = [
            "Уточнить требования и ограничения",
            "Выполнить основную работу",
            "Проверить результат",
        ]
    return [{"index": i + 1, "label": label, "state": "pending"} for i, label in enumerate(labels)]


def _mock_turn(task: dict, user_text: str):
    """Детерминированный «ответ модели»: (reply, marker_or_None).

    Маркер проходит ТУ ЖЕ валидацию конечного автомата, что и реальный ответ:
    единственный способ продвинуться по этапам. Гварды здесь не выставляются —
    их выставляет пользователь через подтверждение (режим approve).
    """
    stage = task["stage"]
    title = (task.get("title") or "Задача").strip()

    if stage == "planning":
        plan = task.get("plan") or _mock_plan(user_text)
        total = len(plan) or 1
        plan_text = "\n".join(f"{s['index']}. {s['label']}" for s in plan)
        reply = (
            f"Эхо (mock). Составил план задачи «{title}»:\n{plan_text}\n\n"
            "Утвердите план (скажите «да»)."
        )
        marker = {
            "stage": "planning",
            "step_index": 0,
            "step_total": total,
            "step_label": "план готов",
            "expected_action": "confirm",
            "plan": plan,
            "reason": "план составлен, жду утверждения",
        }
        return reply, marker

    if stage == "execution":
        plan = task.get("plan") or []
        total = task.get("step_total") or len(plan) or 1
        idx = task.get("step_index") or 1
        if idx >= total:
            reply = (
                f"Эхо (mock). Шаг {total}/{total} выполнен. Проверяю результат "
                "(этап validation). Подтвердите завершение (скажите «да»)."
            )
            marker = {
                "stage": "validation",
                "step_index": total,
                "step_total": total,
                "step_label": "проверка результата",
                "expected_action": "confirm",
                "reason": "выполнение завершено",
            }
            return reply, marker
        done_label = plan[idx - 1]["label"] if idx - 1 < len(plan) else f"шаг {idx}"
        nxt = idx + 1
        next_label = plan[nxt - 1]["label"] if nxt - 1 < len(plan) else ""
        reply = f"Эхо (mock). Выполнил шаг {idx}/{total}: {done_label}."
        if next_label:
            reply += f" Дальше — шаг {nxt}/{total}: {next_label}."
        marker = {
            "stage": "execution",
            "step_index": nxt,
            "step_total": total,
            "step_label": next_label,
            "expected_action": "wait_user",
            "reason": f"шаг {idx} выполнен",
        }
        return reply, marker

    if stage == "validation":
        reply = (
            "Эхо (mock). Результат проверен — всё в порядке. "
            "Подтвердите завершение задачи (скажите «да»)."
        )
        marker = {
            "stage": "validation",
            "step_index": task.get("step_total") or 0,
            "step_total": task.get("step_total") or 0,
            "step_label": "проверка результата",
            "expected_action": "confirm",
            "reason": "ожидание подтверждения валидации",
        }
        return reply, marker

    if stage == "done":
        return "Эхо (mock). Валидация подтверждена. Задача завершена (этап done).", None

    return f"Эхо (mock). Этап {stage}.", None


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


def mock_check_invariants(invariants, user_text) -> dict:
    """Сверяет запрос с явными примерами нарушений (deny_examples) инвариантов."""
    text = (user_text or "").lower()
    for inv in invariants:
        for token in inv.get("deny_examples") or []:
            token = str(token).strip().lower()
            if not token:
                continue
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                return {
                    "violation": True,
                    "invariant_name": inv.get("name"),
                    "invariant": inv,
                    "reason": f"запрос содержит запрещённое для этого инварианта: «{token}»",
                }
    return {"violation": False, "invariant_name": None, "reason": ""}


def create_mock_app(db_path: Optional[str] = None) -> "object":
    """Собирает приложение с мок-провайдером и отдельным хранилищем.

    db_path — путь к SQLite-файлу; None => стандартный agent_history.db.
    В тестах передавайте временный файл (SQLite :memory: не подходит — у хранилища
    несколько соединений).
    """
    store = HistoryStorage(db_path) if db_path else HistoryStorage()
    return create_app(MockProvider(), store)


app = create_mock_app()
