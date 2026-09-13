# Отчёт: персистентный агент с human-in-the-loop (ДЗ 6.4)

> Инкремент к базовому ReAct-графу (ДЗ 6.3): чекпоинтер, `interrupt()`/`Command(resume=...)`,
> SSE-стриминг и time-travel. Реализация — `app/services/agent_persistent.py`,
> `app/routers/agent.py`, `scripts/time_travel_demo.py`, `tests/test_agent_persistent.py`.

## 1. Выбор backend чекпоинтера

Переключение — через env `AGENT_CHECKPOINTER` (`memory` / `sqlite` / `postgres`), дефолт `sqlite`.

| Режим | Чекпоинтер | Когда | Почему |
| :--- | :--- | :--- | :--- |
| `sqlite` | `AsyncSqliteSaver` | локальная разработка | ноль конфигурации, файл `var/agent_checkpoints.sqlite`, один процесс — SQLite достаточно |
| `postgres` | `AsyncPostgresSaver` | docker-compose / production | переживает рестарт контейнера, тот же Postgres, что у FastAPI (multi-tenant, чек-пойнты персистентны) |
| `memory` | `InMemorySaver` | unit-тесты и демо | всё в RAM, пропадает при рестарте — подходит только для тестов |

Ключевой момент: чекпоинтер хранит **операционную память прогона** (состояние графа, паузы HIL,
replay), а не «систему записи» диалога — доменная таблица `chat_messages` остаётся источником
правды для показа/экспорта истории. Таблицы чекпоинтера (`checkpoints`, `checkpoint_writes`,
`checkpoint_blobs`, `checkpoint_migrations`) ведёт `setup()`, доменную схему — Alembic.

Версии: `langgraph 1.2.2`, `langgraph-checkpoint-sqlite 3.1.1`,
`langgraph-checkpoint-postgres 3.1.2`, `aiosqlite 0.22.1`, `psycopg 3.3.5`.

## 2. Конфигурация Postgres в docker-compose

В сервис `app` (`compose.yaml`, секция `environment`) добавлены две переменные — тот же Postgres,
что уже используется FastAPI (сервис `db`, пользователь `chat`, БД `intellibase`):

```yaml
AGENT_CHECKPOINTER: postgres
AGENT_CHECKPOINTER_POSTGRES_URI: postgresql://chat:${POSTGRES_PASSWORD:-pswd}@db:5432/intellibase
```

Важный нюанс драйвера: `AsyncPostgresSaver` работает только на **psycopg v3** (`postgresql://`),
поэтому для чекпоинтера заведён **отдельный URI**, а не переиспользуется `DATABASE_URL`
(`postgresql+asyncpg://…`), который нужен SQLAlchemy. Отсюда два URI в проекте:

- `DATABASE_URL=postgresql+asyncpg://…` — SQLAlchemy/домен (Alembic);
- `AGENT_CHECKPOINTER_POSTGRES_URI=postgresql://…` — чекпоинтер LangGraph (psycopg).

`await checkpointer.setup()` вызывается ровно один раз на старте (внутри `agent_lifespan`, который
подключается в `lifespan` приложения через `AsyncExitStack`) — он создаёт служебные таблицы
чекпоинтера. На каждый HTTP-запрос `setup()` не вызывается.

Проверка таблиц (после `docker compose up -d db`):

```bash
docker compose exec db psql -U chat -d intellibase -c '\dt'
```

Среди таблиц БД `intellibase` (наряду с доменными `chats`, `chat_messages`, `kb_categories`,
`rag_queries` и др.) появились четыре таблицы чекпоинтера:

```
 public | checkpoint_blobs      | table | chat
 public | checkpoint_migrations | table | chat
 public | checkpoint_writes     | table | chat
 public | checkpoints           | table | chat
```

То есть чекпоинтер переиспользует ту же базу, что и FastAPI, но живёт в своих таблицах —
доменная схема и схема чекпоинтера не пересекаются.

Чтобы Alembic не вписал `drop_table` для таблиц LangGraph (их нет в SQLAlchemy-моделях), в
`migrations/env.py` добавлен `include_name`, исключающий `checkpoints` / `checkpoint_writes` /
`checkpoint_blobs` / `checkpoint_migrations`.

## 3. Опасный tool и границы interrupt()

Опасный инструмент — **`send_telegram_message`** (отправка сообщения клиенту в Telegram). Это
единственный из трёх инструментов с побочным эффектом (уходит наружу), поэтому именно он закрыт
human-in-the-loop. `search_knowledge_base` и `get_current_time` — read-only, HIL им не нужен.

Граф разбит на два узла с обязательным ребром `prepare_send → confirm_and_send`:

- **до interrupt()** (`prepare_send`, идемпотентно): из `tool_call` рендерится payload-словарь
  `draft = {chat_id, text, tool_call_id}`. Никакого side-effect — только чтение аргументов.
- **после interrupt()** (`confirm_and_send`): `decision = interrupt({"type": "approve_send", "preview": draft})`,
  и только при одобрении (`resume=True`) вызывается реальная отправка `await send_telegram_fn(draft)`.

Почему это критично: при resume узел перезапускается **с начала**, а не со строки `interrupt()`.
Если side-effect поставить до `interrupt()`, он выполнится дважды. Поэтому до прерывания — только
идемпотентная подготовка, а сам side-effect — после.

Используется актуальный API LangGraph 1.0: `interrupt()` + `Command(resume=...)`. Устаревшие
`interrupt_before` / `interrupt_after` не задействованы.

## 4. Лог: __interrupt__ payload + Command(resume=True)

Вывод `scripts/time_travel_demo.py` (sqlite in-memory, фейковая модель):

```
1) INTERRUPT payload: {'type': 'approve_send', 'preview': {'chat_id': '34564444', 'text': 'Вот информация из базы знаний...', 'tool_call_id': 'call-1'}}
4) две ветки: отказ → sent=False, одобрение → sent=True, отправок=1
Итог: один и тот же вход дал две ветки — отказ (sent=False) и одобрение (sent=True).
```

В момент `interrupt()` граф останавливается и наружу уходит payload с превью действия
(`approve_send` + текст сообщения). После `Command(resume=True)` узел `confirm_and_send`
выполняет `send_telegram_fn(draft)` ровно один раз (`отправок=1` — эффект сработал только в
ветке одобрения). В ветке отказа `sent=False`, side-effect не вызван.

## 5. Time travel: история чек-пойнтов и replay

Вывод `aget_state_history` из того же демо (свежие → старые):

```
1f1a8740-1b03-…  next=('confirm_and_send',)
1f1a8740-1b01-…  next=('prepare_send',)
1f1a8740-1af5-…  next=('call_model',)
1f1a8740-1af0-…  next=('__start__',)
```

Чтение состояния на **прошлом** чек-пойнте (`aget_state` с `checkpoint_id` снимка, где
`next=('confirm_and_send',)`):

```
3) чтение прошлого чек-пойнта: sent=False, draft_готов=True, next=('confirm_and_send',)
```

То есть в прошлом состоянии черновик уже подготовлен (`draft` непустой), отправка ещё не
выполнена (`sent=False`), а `next` указывает на узел подтверждения.

Replay с противоположным решением показан как **две ветки из одинакового входа**. Важный нюанс:
значение `Command(resume=...)` сохраняется в чекпоинтере как pending-write и **детерминировано на
весь thread-lineage** — повторный resume того же interrupt-чекпойнта с другим значением вернёт
первое зафиксированное. Поэтому две ветки строятся на **двух разных `thread_id`**
(`demo` и `demo-alt`) с одинаковым входом, а не повторным resume одного треда.

## 6. Streaming mode и обоснование

Выбран `graph.astream(input, config, stream_mode=["updates", "messages"])`:

- `updates` — прогресс по узлам («выполнен узел X»), для прогресс-бара;
- `messages` — токены LLM по мере генерации, для chat-стиля.

`astream_events(version="v2")` даёт больше деталей (`on_chat_model_stream` и т.п.), но дороже по
объёму событий; для базового SSE достаточно `updates` + `messages`. Эндпоинт `POST /agent/stream`
оборачивает их в `StreamingResponse(media_type="text/event-stream")`, каждое событие — `data: {json}\n\n`.

Проверка одного потока (до паузы):

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","input":{"messages":[{"role":"user","content":"отправь клиенту 34564444 сообщение Привет"}]}}'
```

Вывод:

```
data: {"type": "update", "nodes": ["call_model"]}
data: {"type": "update", "nodes": ["prepare_send"]}
data: {"type": "interrupt", "payload": {"type": "approve_send", "preview": {"chat_id": "34564444", "text": "Привет", "tool_call_id": "call_00_xeXxdy3M9Ks00FcYc7sL8143"}}}
data: {"type": "done"}
```

Момент паузы распознаётся по `__interrupt__` в payload режима `updates` (в LangGraph 1.2.2
interrupt приходит именно так). После подтверждения поток возобновляется тем же `thread_id`:

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","resume":true}'
```

Вывод (продолжение потока; токены финального ответа идут по частям, здесь частично сокращены):

```
data: {"type": "token", "text": "сообщение отправлено: Привет"}     # ToolMessage от confirm_and_send
data: {"type": "update", "nodes": ["confirm_and_send"]}
data: {"type": "token", "text": "Со"}
data: {"type": "token", "text": "об"}
data: {"type": "token", "text": "щение"}
...                                                                 # «Сообщение «Привет» отправлено клиенту в чат 34564444. ✅»
data: {"type": "update", "nodes": ["call_model"]}
data: {"type": "update", "nodes": ["force_finish"]}
data: {"type": "done"}
```

Видны оба режима: `updates` (узлы `confirm_and_send` → `call_model` → `force_finish`) и `messages`
(токены LLM + ToolMessage). После `Command(resume=True)` поток дошёл до финального ответа.

## 7. Permission policy

Роль пользователя передаётся в `config["configurable"]["user_role"]` и принимает одно из трёх
значений: `read-only` / `write-with-approve` (дефолт) / `full`. В узле `confirm_and_send` роль
проверяется: для `full` `interrupt()` пропускается (действие выполняется без подтверждения), для
`write-with-approve` — обязательная пауза. Политика выбора уровня = выбор набора `tools` для
конкретного треда/пользователя; промпт «не делай опасных вещей» границей безопасности не является.

## 8. Что осталось хрупким и что хочется доделать

- **Реальная отправка — заглушка.** `_send_telegram_impl` только печатает в лог; интеграция с
  реальным ботом (через `bot/notify`) — вне рамок этого ДЗ.
- **Демо на фейковой модели.** `FakeChat` в `time_travel_demo.py` и тестах эмитит tool-call
  безусловно, минуя `bind_tools`, поэтому сам по себе не доказывает, что реальная модель «сама
  решает» отправить сообщение. Этот путь проверяется через `curl` на `/agent/stream`.
- **`_psycopg_uri(database_url)`** принимает аргумент, но игнорирует его и читает настройку —
  либо использовать аргумент (замена `+asyncpg` → `postgresql://`), либо убрать параметр.
- **Интеграция стриминга в Telegram-бот** — отдельная задача (в рамках ДЗ не требовалась).
