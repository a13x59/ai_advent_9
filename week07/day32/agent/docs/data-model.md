# Схемы данных (SQLite, `agent/storage.py`)

Хранилище — SQLite (`agent_history.db`), по одной таблице на сущность.

## `conversations` — сессии

| Поле                        | Тип     | Назначение             |
|-----------------------------|---------|------------------------|
| `session_id`                | TEXT PK | ID сессии              |
| `current_branch`            | TEXT    | Активная ветка диалога |
| `strategy`                  | TEXT    | Стратегия контекста    |
| `window_size`               | INT     | Размер окна            |
| `profile_id`                | TEXT    | Привязанный профиль    |
| `created_at` / `updated_at` | TEXT    | Временные метки        |

## `branches` — ветки диалога

`branch_id` (PK), `session_id`, `name`, `messages` (JSON), `checkpoint`, метки.

## `working_memory` — рабочая память (текущая задача)

`entry_id` (PK), `session_id`, `profile_id`, `key`, `value`, `kind`, `state`,
`importance`, метки. Уникальность: `(session_id, profile_id, key)`.

`kind ∈ {goal, constraint, todo, result, context, note, clarification, term}`;
`state ∈ {pending, in_progress, done, blocked}`.

## `long_term_memory` — постоянная память

`entry_id` (PK), `key` (UNIQUE), `profile_id`, `value`, `kind`, `tags` (JSON),
`importance`, метки.

`kind ∈ {profile, decision, knowledge, preference, agreement, fact}`.

## `user_profiles` — профили

`profile_id` (PK), `name` (UNIQUE), `display_name`, `style`, `format`,
`constraints`, `notes`, `is_active`, метки.

## `memory_log` — журнал действий над памятью

`id`, `session_id`, `layer`, `action`, `key`, `source`, `created_at`.

## `tasks` — задачи конечного автомата

`task_id` (PK), `session_id`, `title`, `stage`, `status`, `step_index`,
`step_total`, `step_label`, `expected_action`, `plan` (JSON), `resume_note`,
`guards` (JSON), `messages` (JSON), метки.

Частичный уникальный индекс `idx_one_active_task` гарантирует не более одной активной задачи на сессию.

## `task_transitions` — журнал переходов

`id`, `task_id`, `session_id`, `from_stage`, `to_stage`, `from_status`,
`to_status`, `step_label`, `expected_action`, `source` (`model`/`user`/`manual`),
`reason`, `accepted`, `created_at`.

## `invariants` — инварианты

`invariant_id` (PK), `profile_id`, `session_id`, `task_id` (скоуп), `category`,
`name`, `statement`, `rationale`, `deny_examples` (JSON), `is_active`, метки. Уникальность:
`(profile_id, session_id, task_id, name)`.

## Конечный автомат (FSM), `task_state.py`

- Этапы: `planning → execution → validation → done`.
- Статусы жизни задачи: `active | paused | done`.
- `expected_action`: `model_call | wait_user | confirm | done`.
- Гварды: `execution` требует `plan_approved`, `done` требует `validation_passed`.
- Переходы: `planning→{planning,execution}`, `execution→{execution,validation}`,
  `validation→{execution,validation,done}`, `done→{done}`.
- Прогресс модель сообщает маркером `[STATE] {"stage": ..., "step_index": ...}`, который валидируется и применяется
  `validate_and_apply`.
