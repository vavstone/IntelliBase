# Инструкция по локальному запуску IntelliBase в режиме дебага

## Архитектура запуска

Python-код работает на хосте Windows (можно дебажить в PyCharm), а сторонние сервисы
(Redis, PostgreSQL, Arize Phoenix, Qdrant) — в Docker-контейнерах.

```
┌──────────────────────────────────────────────────────┐
│  Docker (compose.infra.yaml)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Redis   │ │ Postgres │ │ Phoenix  │ │  Qdrant  │ │
│  │  :6379   │ │  :5432   │ │  :6006   │ │  :6333   │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└──────────────────────────────────────────────────────┘
         ↑ localhost          ↑ localhost

┌─────────────────────────────────────────────┐
│  PyCharm Debugger / uv run (твой код)        │
│  ┌──────────────────────────────────┐        │
│  │  uvicorn app.main:app --reload   │        │
│  │  Порт: 8000                      │        │
│  │  Можно ставить breakpoints        │        │
│  └──────────────────────────────────┘        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Ollama (отдельно, если нужен локальный LLM) │
│  http://localhost:11434                      │
└─────────────────────────────────────────────┘
```

`.env` уже настроен на `localhost` для всех сервисов — ничего менять не нужно.

---

## Файлы для разработки

| Файл | Назначение |
|-------|------------|
| `compose.infra.yaml` | Docker Compose **только** с redis, db, phoenix, qdrant (без app) |
| `dev-infra-up.bat` | Запуск инфраструктуры (двойной клик) |
| `dev-infra-down.bat` | Остановка инфраструктуры |
| `dev.bat` | Полный запуск: инфра + uvicorn (двойной клик) |
| `.idea/runConfigurations/IntelliBase_Debug.xml` | Конфигурация дебага FastAPI в PyCharm |
| `.idea/runConfigurations/IntelliBase_Bot.xml` | Конфигурация дебага Telegram-бота |

---

## Пошаговая настройка

### 1. Настройка интерпретатора в PyCharm (однократно)

1. PyCharm → **Settings** → **Project: IntelliBase** → **Python Interpreter**
2. **Add Interpreter** → **Add Local Interpreter** → выбери **uv** (PyCharm 2024.2+)
   или укажи путь к `.venv/Scripts/python.exe`
3. Если через uv — PyCharm сам подхватит зависимости из `pyproject.toml`

### 2. Запуск инфраструктуры (Redis, PostgreSQL, Phoenix, Qdrant)

**Самый простой способ — двойной клик по `dev-infra-up.bat`.**

Или из терминала:

```bash
docker compose -f compose.infra.yaml up -d
```

Проверка, что всё поднялось:

```bash
docker compose -f compose.infra.yaml ps
```

Ожидаемый вывод: четыре контейнера со статусом `Up` (healthy).

### 3. Дебаг приложения в PyCharm

1. Убедись, что Docker-инфраструктура запущена (шаг 2)
2. В PyCharm: выпадающий список конфигураций (справа сверху) → **IntelliBase Debug**
3. Нажми кнопку **Debug** (зелёный жук) — приложение запустится с `--reload`

После первого запуска может потребоваться:

- **Run** → **Edit Configurations** → выбрать **IntelliBase Debug**
- Проверить, что **Python interpreter** указывает на `.venv`
- На вкладке **EnvFile** должен быть включён `.env`

### 4. Альтернативный запуск из терминала

```bash
# Только приложение (инфраструктура уже должна быть запущена):
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Или напрямую через .venv (как делают .bat-файлы):
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Или всё вместе одной командой:
dev.bat
```

> **Почему в `.bat`-файлах используется `.venv\Scripts\python.exe`, а не `uv run`?**
> При двойном клике `cmd.exe` может не найти `uv` в PATH. Прямой вызов `.venv`-шного
> python работает всегда, независимо от переменных окружения.

### 5. Остановка

```bash
# Остановить инфраструктуру (двойной клик по dev-infra-down.bat):
docker compose -f compose.infra.yaml down
```

Data Redis и Postgres сохраняются в Docker-volumes между перезапусками.
Чтобы сбросить данные:

```bash
docker compose -f compose.infra.yaml down -v
```

---

## Что ещё может понадобиться

### PostgreSQL и миграции

По умолчанию `CHAT_REPOSITORY=json` — чаты хранятся в JSON-файлах, PostgreSQL
не требуется. Если хочешь переключиться на Postgres, измени в `.env`:

```
CHAT_REPOSITORY=postgres
```

И выполни миграции:

```bash
uv run alembic upgrade head
```

### Ollama (локальный LLM)

Если используешь `LLM__DEFAULT_PROVIDER=ollama` (текущая настройка), Ollama
должна быть установлена и запущена отдельно. Модели: `qwen2.5:3b` (чат) и
`gemma3:4b` (RAG):

```bash
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

### Phoenix UI

Интерфейс для просмотра трейсов LLM-вызовов доступен по адресу:
[http://localhost:6006](http://localhost:6006)

Работает только если в `.env` задан `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317`.

### Production-сборка через Docker

Оригинальный `compose.yaml` (с 4 сервисами, включая app) остался без изменений:

```bash
docker compose up -d
```

---

## Шпаргалка

| Что делаешь | Чем |
|---|---|
| Запустить Redis + PG + Phoenix + Qdrant | Двойной клик по `dev-infra-up.bat` |
| Дебажить код в PyCharm | Run Configuration → **IntelliBase Debug** → Debug |
| Всё одной командой в терминале | `dev.bat` |
| Остановить инфраструктуру | Двойной клик по `dev-infra-down.bat` |
| Сбросить все данные Docker | `docker compose -f compose.infra.yaml down -v` |
| Миграции БД | `uv run alembic upgrade head` |
| Phoenix UI (трейсы) | http://localhost:6006 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Swagger API | http://localhost:8000/docs |
