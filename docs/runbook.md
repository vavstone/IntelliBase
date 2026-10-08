# Runbook — операционные команды

Копипаст-команды для запуска, проверки и переиндексации. Все — из корня проекта.
Порты и сервисы — см. [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Быстрый старт (Makefile)

Основные операции собраны в `Makefile` — одна команда вместо цепочки шагов:

```bash
make up         # поднять стек и дождаться готовности (docker compose up -d --build --wait)
make smoke      # проверить: контейнеры, /health, /ready, коллекция Qdrant, Phoenix
make smoke-rag  # то же + сквозной вопрос к RAG (нужны LLM и наполненный корпус)
make metrics    # метрики: p95 задержек, cache hit rate, последние числа RAGAS
make down       # остановить стек (данные в томах сохраняются)
make help       # список всех целей
```

Вспомогательные: `make logs`, `make ps`, `make restart`, `make shell`,
`make test`, `make test-all`, `make ingest`, `make reindex`, `make eval`,
`make clean` (удаляет тома — осторожно).

### Метрики (`make metrics`)

Печатает две группы цифр:

* из admin-API `/chats/admin/stats` — задержки за окно (по умолчанию 24 ч,
  `METRICS_ARGS="--window-hours 168"` для недели), cache hit rate и доля отказов
  RAG. Задержки считаются **без** служебных `/health` и `/ready`: пробы
  healthcheck'ов идут каждые 15 секунд и без этого фильтра p95 уезжает в миллисекунды;
* из последнего файла `tests/eval/results/*.csv` — метрики RAGAS
  (faithfulness, answer_relevancy, context_precision, context_recall, has_citation).

Нужны поднятый стек и `ADMIN_TOKEN` в `.env`. Цель запускается через `uv` на
хосте (не в контейнере): файлы результатов RAGAS лежат в `tests/`, а этот каталог
в образ не копируется.

`make smoke` запускается изнутри контейнера `app`, если стек поднят (там гарантированно
есть python), иначе — локально через `uv`. Код возврата 1 — есть провал (используется
в проверках «готов к защите»).

Первый `make up` долгий: приложение скачивает embedding-модель E5 (~2 ГБ) и
индексирует корпус; в `compose.yaml` для этого задан `start_period: 600s`.

Если модель уже скачана на хосте — укажите её кэш в `.env`, и она смонтируется
в контейнер (старт вместо скачивания):

```bash
HF_CACHE_DIR=/home/user/.cache/huggingface          # Linux/macOS
HF_CACHE_DIR=C:/Users/<user>/.cache/huggingface     # Windows
```

Каталог весов для LlamaIndex задаётся отдельно (`EMBEDDING_MODEL_CACHE_DIR`,
в compose это `/home/appuser/.cache/huggingface/hub`): без него LlamaIndex ищет
модель в собственном кэше внутри контейнера, теряет её при пересоздании и
`/rag/query` отвечает 503. Проверка — `make smoke-rag`.

### Если контейнер app уходит в рестарт-луп

`docker logs llm-service` показывает `exec /app/entrypoint.sh: no such file or
directory` — значит файлы выгружены с переводами строк CRLF, и шебанг в скрипте
превратился в `#!/bin/sh\r`. Так бывает на Windows при `core.autocrlf=true`
(значение по умолчанию), если в репозитории нет `.gitattributes` с `eol=lf`.
Лечится переклонированием после обновления репозитория; локальная проверка:

```bash
head -c 20 entrypoint.sh | od -c   # ожидается "#!/bin/sh\n", без \r
git config --get core.autocrlf     # true — не страшно, .gitattributes перекрывает
```

### `make up` падает на «dependency failed to start: container llm-service is unhealthy»

`docker logs llm-service` показывает `ValueError: Unknown scheme for proxy URL` и
`ERROR: Application startup failed. Exiting.` — значит `PROXY_URL` содержит не
URL. Пустое значение из `.env.example` (`PROXY_URL=`) — тоже не URL: httpx
принимает `None`, но не пустую строку, и падает прямо в lifespan на
`app/main.py` (`httpx.AsyncClient(proxy=settings.proxy_url)`), до старта uvicorn.
С 08.10 пустая строка приводится к `None` валидатором `proxy_url`
(`app/core/config.py`, `bot/config.py`). В образе, собранном раньше, лечится
удалением строки `PROXY_URL` из `.env` целиком — пересборка не нужна.

### Если команда «висит», а сервис жив (Windows + Docker Desktop)

Запрос к `localhost:<порт>` не отвечает до таймаута, при этом `127.0.0.1:<порт>`
работает и контейнер healthy. Причина — IPv6-проброс портов Docker Desktop:
`localhost` резолвится в `::1`, а этот путь идёт через `wslrelay.exe`, который
изредка залипает (несколько минут). Проверка и лечение:

```bash
curl -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health   # 200
curl -m 5 -o /dev/null -w "%{http_code}\n" http://localhost:8000/health   # 000/виснет
```

Само отпускает за 1–2 минуты; радикально — рестарт Docker Desktop
(`wsl --shutdown`, затем запуск Docker). В `.env` этого репозитория адреса
docker-сервисов указаны через `127.0.0.1` именно поэтому; `scripts/smoke.py`
по умолчанию тоже обращается к IPv4.

## Запуск и остановка

### Docker-инфраструктура (Redis, Postgres, Qdrant, Phoenix)

```bash
docker compose -f compose.infra.yaml up -d      # поднять
docker compose -f compose.infra.yaml down       # остановить
```

### Полный стек (app + bot + инфраструктура)

```bash
docker compose up -d --build
```

### Локальный запуск приложения (вне Docker)

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
uv run python -m bot                            # бот (отдельный процесс)
```

> Ollama не входит в compose — запускается локально (`ollama serve`). Модели:
> `gemma3:4b` (чат), `qwen3:8b` (RAG, на CPU ~2–4 мин на ответ), `qwen2.5:3b` (резерв).

## Проверка живости

```bash
# FastAPI
curl -s -m 3 http://localhost:8000/health
curl -s -m 3 http://localhost:8000/ready

# Ollama (список моделей)
curl -s -m 3 http://localhost:11434/api/tags

# Qdrant (список коллекций)
curl -s -m 5 http://localhost:6333/collections

# Phoenix UI
# http://localhost:6006
```

## Индексация корпуса (офлайн-контур)

По умолчанию индексируется демонстрационный корпус `data/demo_kb` (18
синтетических документов, лежит в репозитории) в коллекцию `rag_demo`.
Рабочий корпус подключается через `RAG_DATA_DIR`/`RAG_COLLECTION` в `.env`.

```bash
# Инкрементально (UPSERTS пропустит неизменённое)
uv run python scripts/ingest.py data/demo_kb

# Полная переиндексация (вычистить и заново)
uv run python scripts/ingest.py data/demo_kb --full

# Точечно — только перечисленные файлы
uv run python scripts/ingest.py --files data/demo_kb/tarify/a.pdf data/demo_kb/malahit/b.docx

# Рабочий корпус (не хранится в git)
uv run python scripts/ingest.py data/kb          # RAG_COLLECTION=rag_block_05 в .env
```

## Демонстрационный корпус

Синтетические документы (7 категорий-ПС, PDF + DOCX) генерируются скриптом —
их безопасно держать в публичном репозитории и удобно использовать для демо
и для наполнения коллекции на чистом клоне:

```bash
# Пересобрать корпус (PDF собираются шрифтом DejaVu Sans)
uv run python scripts/generate_demo_corpus.py

# Только проверка состава, без записи файлов
uv run python scripts/generate_demo_corpus.py --check

# Очистить каталог и собрать заново
uv run python scripts/generate_demo_corpus.py --clean
```

Содержимое документов описано декларативно в `scripts/demo_corpus/content_*.py`.

## Внешние LLM: прокси и TLS

`PROXY_URL` применяется только к OpenAI/OpenRouter. DeepSeek доступен напрямую —
под него создаётся отдельный HTTP-клиент **без** прокси: через прокси запрос
падает на проверке сертификата (`SSL: CERTIFICATE_VERIFY_FAILED ... self-signed
certificate in certificate chain`), потому что прокси подменяет TLS-цепочку.

Проверка связи из контейнера:

```bash
docker compose exec app python -c "
import os, httpx
key = os.environ['LLM__DEEPSEEK_API_KEY']
r = httpx.post('https://api.deepseek.com/chat/completions',
    headers={'Authorization': f'Bearer {key}'},
    json={'model': 'deepseek-v4-flash',
          'messages': [{'role': 'user', 'content': 'ping'}], 'max_tokens': 2},
    timeout=20)
print(r.status_code)"
```

Ожидаемый ответ — `200`. Разовый сетевой сбой на стороне провайдера виден в логах
app как `APIConnectionError` после нескольких ретраев — повторный запрос обычно
проходит.

Альтернатива через API:

```bash
curl -s -X POST http://localhost:8000/documents/reindex \
  -H 'Content-Type: application/json' \
  -d '{"mode":"full"}'   # или "incremental" / "files"
```

## Сброс RAG-коллекции

Нужен после смены embed-модели или схемы метаданных (когда инкрементальный
UPSERTS уже не отражает реальность):

```bash
curl -s -X DELETE http://localhost:6333/collections/rag_demo
rm -f var/rag_docstore.json
uv run python scripts/ingest.py data/demo_kb --full
```

Для рабочего корпуса — то же с `rag_block_05` и `data/kb`.

## Smoke-тест RAG

```bash
curl -s -m 120 -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Что входит в состав КПС Тарифы?","category":"tarify"}'
```

Ожидается JSON с `answer`, `sources` (нумерованные цитаты), `confident`.
Off-topic вопрос → `confident=false` + честный отказ без вызова LLM.

## Оценка качества RAG (RAGAS, Б5.6)

Зависимости ставятся опциональными группами: `uv sync --extra eval --extra tracing`.
Судья — DeepSeek (`EVAL_JUDGE_MODEL=deepseek-v4-flash`, без VPN), эмбеддинги судьи —
локальная E5. Production-RAG в `/rag/query` при этом не меняется.

```bash
# Генерация golden dataset (сырой CSV, дальше ручная вычитка)
uv run --extra eval python scripts/generate_testset.py --size 40

# Прогон метрик (пишет tests/eval/results/{timestamp}_{label}.csv + .json)
uv run --extra eval python scripts/run_eval.py --label baseline

# A/B-варианты (override коллекции / top-K / re-ranker)
uv run --extra eval python scripts/run_eval.py --collection rag_block_05_chunk1024 --label chunk_1024
uv run --extra eval python scripts/run_eval.py --top-k 5 --label top_k_5

# Самопроверка критериев Б5.6
uv run --extra eval --extra tracing python dev_tasks/verify_5_6.py
```

### Прогон на демо-корпусе

По умолчанию `make eval` считает метрики по демо-корпусу: golden —
`tests/eval/golden_dataset_demo.json`, метка — `demo`, коллекция — `rag_demo`
(из `.env`). Рабочий корпус: `make eval GOLDEN=tests/eval/golden_dataset.json LABEL=block_05`.

```bash
make eval          # RAGAS на демо-корпусе (нужны стек и ключ DeepSeek)
make thresholds    # сверить последний прогон с порогами eval/thresholds.yaml
make metrics       # те же числа + p95 и cache hit rate
```

Скрипты `generate_testset.py` и `run_eval.py` сами подхватывают `.env`
(`HF_HUB_OFFLINE=1` — модель E5 из кэша, без обращений к huggingface.co) и включают
`truststore` — без него TLS-инспекция Kaspersky подменяет сертификат и часть запросов
к DeepSeek падает с `CERTIFICATE_VERIFY_FAILED`. При запуске из контейнера это не нужно:
там переменные приходят из compose.

### A/B по чанкингу: отдельная коллекция

Смена chunk_size — это переиндексация (офлайн-контур). Нужна **отдельная коллекция и
отдельный docstore**, иначе UPSERTS пропустит всё по хешу (chunk_size в хеш не входит):

```bash
RAG_COLLECTION=rag_block_05_chunk1024 \
RAG_CHUNK_SIZE=1024 \
RAG_DOCSTORE_PATH=var/rag_docstore_chunk1024.json \
uv run python scripts/ingest.py data/kb
```

### Трейсы в Phoenix (RAG, чат, агент)

Phoenix поднимается вместе со стеком (`compose.yaml`, сервис `phoenix`), UI —
[http://localhost:6006](http://localhost:6006) → проект **`diploma-fastapi`**.

- **В контейнере** трейсинг включён всегда: `compose.yaml` задаёт
  `PHOENIX_ENABLED=true` и gRPC-эндпоинт `http://phoenix:4317`, а образ собирается
  с `--extra tracing` (инструментеры OpenAI, LangChain/LangGraph, LlamaIndex).
- **На хосте** нужен `uv sync --extra tracing` и `PHOENIX_ENABLED=true` в `.env`
  (HTTP-эндпоинт `http://localhost:6006/v1/traces`).
- Имя проекта задано в коде (`app/observability/tracing.py`, `diploma-fastapi`).

Один HTTP-запрос = один трейс: корень — серверный спан (`POST /rag/query`) от
инструментера FastAPI, внутри — спаны библиотек. Без серверного спана они
расходятся по отдельным трейсам (инструментеры LangChain/LlamaIndex намеренно не
привязывают спаны к OTel-контексту), и один вопрос выглядит как 4–6 записей.

| Операция | Спаны внутри трейса |
|---|---|
| `POST /rag/query`, RAG в `/chats` | `VectorIndexRetriever.aretrieve` (+ scores) → `HuggingFaceEmbedding.*`; `OllamaLLM.acomplete` → `ChatCompletion` |
| `POST /chat`, `/chats`, модерация | `ChatCompletion` (OpenAI SDK) |
| `POST /agent/chat`, `/agent/resume` | `LangGraph` → узлы (`call_model`, `execute_tool`, `prepare_send`, `confirm_and_send`) → `ChatOpenAI`; инструменты — `search_knowledge_base` с вложенными RAG-спанами |

Служебные пробы (`/health`, `/ready`, `/docs`, `/openapi.json`), проверка доступа
(`GET /access/{chat_id}` — бот дёргает её на каждом апдейте) и создание чата
(`POST /chats`) исключены из трейсинга переменной `PHOENIX_EXCLUDED_URLS`, спаны
`http receive/send` не пишутся. Шаблон для создания чата — с якорями
(`^https?://[^/]+/chats$`): список сопоставляется регулярками через `re.search`
по полному URL, и шаблон `/chats` без якорей выключил бы заодно
`/chats/{id}/messages` — основной путь чата со спанами LLM.

**Пустые колонки у корневого спана — это норма.** У HTTP-спана `kind = unknown`,
`status` — прочерк (`UNSET`), `input`/`output`/`metadata` пусты: тип спана и
вход/выход проставляют только инструментеры LLM-библиотек (OpenInference), а
FastAPI-инструментер тел запросов и ответов не перехватывает. Заполненные данные
(промпт, ответ, токены, `thread_id`) — на дочерних спанах `llm`/`chain`/`tool`:
в списке нужно снять фильтр **Root Spans** или раскрыть трейс. Токены у корня при
этом не нулевые — колонка кумулятивная (сумма по потомкам), а стоимость нулевая
потому, что `deepseek-v4-flash` нет во встроенном прайс-листе Phoenix.

Однострочная проверка, что спаны доехали (без UI):

```bash
curl -s "http://localhost:6006/v1/projects/UHJvamVjdDoy/spans?limit=5" | head -c 400
```

Отдельный демо-скрипт по RAG (без FastAPI), из хоста:

```bash
PHOENIX_ENABLED=true uv run --extra tracing python scripts/trace_demo.py
```

Если спанов нет: проверить, что `PHOENIX_ENABLED=true`, что в логе старта есть
строка `Phoenix-трейсинг включён (OpenAI, LangChain, LlamaIndex)`, и что версия
образа содержит `--extra tracing` (в старом образе инструментеров RAG/агента нет —
будет только `ChatCompletion` от OpenAI-инструментера).

## Доступ к боту: кому выдавать и как отзывать

Бот закрыт: список пользователей — в таблице `bot_users`, управление через админ-API
(под `X-Admin-Token`). Пустая таблица = бот не отвечает никому; `BOT_ALLOWED_CHAT_IDS`
в `.env` — bootstrap для чистого клона и аварийный доступ.

```bash
# Основной путь — скрипт (адрес бэкенда из BACKEND_URL, токен из ADMIN_TOKEN)
uv run python scripts/bot_users.py list            # кто имеет доступ
uv run python scripts/bot_users.py list --all      # включая отозванные
uv run python scripts/bot_users.py add 111222333 "Иванов Пётр, руководитель отдела"
uv run python scripts/bot_users.py remove 111222333
uv run python scripts/bot_users.py check 111222333 # что ответит бот (GET /access)

# То же через Makefile
make users ARGS='list'
make users ARGS='add 111222333 "Иванов Пётр, руководитель отдела"'
```

Тот же API напрямую (если нужен curl):

```bash
ADMIN='change-me-admin'   # значение ADMIN_TOKEN из .env

curl -s http://127.0.0.1:8000/chats/admin/bot-users -H "X-Admin-Token: $ADMIN"

curl -s -X POST http://127.0.0.1:8000/chats/admin/bot-users \
  -H "X-Admin-Token: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"chat_id":"111222333","title":"Иванов Пётр, руководитель отдела"}'

curl -s -X DELETE http://127.0.0.1:8000/chats/admin/bot-users/111222333 \
  -H "X-Admin-Token: $ADMIN"
```

Список хранится в Postgres (таблица `bot_users`, том `postgres_data`), поэтому
переживает `make down`/`make up`; `make clean` (удаление томов) его стирает —
тогда доступ снова только у id из `BOT_ALLOWED_CHAT_IDS`.

Прямой SQL на случай, когда сервис лежит:

```bash
docker compose exec db psql -U chat -d intellibase -c "
  INSERT INTO bot_users (chat_id, title, created_by)
  VALUES ('111222333', 'Иванов Пётр', 'manual')
  ON CONFLICT (chat_id) DO UPDATE
    SET title = EXCLUDED.title, is_active = TRUE;"
```

Проверка гейта без Telegram (тот же вызов, что делает бот на каждом апдейте):

```bash
curl -s http://127.0.0.1:8000/access/111222333 \
  -H "X-Internal-Token: $(grep '^INTERNAL_TOKEN=' .env | cut -d= -f2)"
# → {"chat_id":"111222333","allowed":true}
```

Пользователь, которому отказано, видит свой id — его и передаёт администратору.
Проверка кэшируется в боте на 60 с, отзыв доступа действует в пределах этого окна.

## Демо: «покажи документы → отправь файл коллеге»

Сценарий, ради которого сделаны `list_documents` и отправка вложением:

1. В боте: **«Какие ФТ есть в базе знаний?»** — вопрос уходит в обычный чат (RAG).
   Чтобы получить точный список файлов, спросите агентом: `/agent какие документы
   есть в базе` — тогда ответит `list_documents` по файлам корпуса, а не выборка
   чанков из векторного поиска.
2. **«Отправь ФТ по Тарифам мне и Иванову»** — команда не нужна: сообщение с
   просьбой отправить/переслать уходит агенту автоматически (ответ помечен
   «🤖 Обработано агентом»). Агент берёт точное имя файла, находит получателей
   через `find_recipient` и уходит на подтверждение. Имя файла можно писать без
   расширения («… Версия 1.9») — сервер найдёт `… Версия 1.9.docx`.
3. В превью видно и файл, и всех получателей: `Иванов Пётр (111222333), …`.
4. После «✅ Отправить» файл уходит вложением (`sendDocument`) каждому получателю;
   в подписи — краткое описание из задачи.

Ограничения, о которых стоит знать: файл должен лежать в корпусе (`RAG_DATA_DIR`),
быть документом (pdf/docx/txt/md/xlsx/csv) и не больше 50 МБ (лимит Telegram).
Пути и `..` отбрасываются — модель называет имя файла, путь собирает сервер.

## Агент: HIL-цепочка без бота (запасной план для шага 4 демо)

Если на защите не нажимаются кнопки в Telegram, тот же сценарий проходится curl'ом.
`chat_id` — обязательное поле: это чат-инициатор, ему отправка разрешена всегда
(остальным — только если они есть в `bot_users`).

```bash
# 1. Старт задачи: агент дойдёт до опасного действия и вернёт status="interrupted"
curl -s -X POST http://localhost:8000/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Найди регламент по обмену и отправь выжимку","thread_id":"demo-1","chat_id":"100000001"}'
# → {"status":"interrupted","interrupt":{"type":"approve_send","preview":{...}}}

# 2a. Подтвердить (реальная отправка через бота) — decision=false отменяет
curl -s -X POST http://localhost:8000/agent/resume \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo-1","decision":true,"chat_id":"100000001"}'
```

Проверка политики получателя: тот же запрос с `"chat_id":"999999"` (чужой адрес в
тексте задачи) не отправит ничего — агент получит ошибку инструмента и переспросит
с разрешённым получателем; в логе app появится `Отклонена отправка: запрошен chat_id=...`.

## Эксперимент Б6.5: мультиагент против single-agent

Нужны поднятый Qdrant (коллекция `rag_block_05`) и доступ к DeepSeek. Скрипты пишут в один
файл `experiments/results.json` (merge по `(impl, qid)`), поэтому порядок запуска любой.

```bash
rm experiments/results.json                          # чистовой прогон: убрать прошлые данные

uv run python -m experiments.multi_agent_langgraph   # 5 вопросов мультиагентом + схема графа
uv run python -m experiments.single_agent_baseline   # те же 5 вопросов одним агентом
uv run python -m experiments.judge                   # LLM-судья: quality_score в results.json

uv run python dev_tasks/verify_6_5.py                # самопроверка ДЗ (15 критериев)
```

Признаки корректного прогона мультиагента: в консоли на каждый вопрос четыре узла
(`supervisor → researcher → supervisor → writer → supervisor`), `handoff_count = 2`,
`llm_calls ≈ 6`, а в ответе есть ссылки `[1]`, `[2]`. Итоговые документы —
[docs/multi-agent-report.md](multi-agent-report.md) и
[docs/architecture-multi-agent.md](architecture-multi-agent.md).

## Тесты

```bash
# Быстрый прогон — без интеграционных тестов, требующих PG/Qdrant/Ollama
uv run pytest tests/ -v \
  --ignore=tests/chat/test_routes.py \
  --ignore=tests/chat/test_service_context.py

# Полный прогон — нужна поднятая инфраструктура
uv run pytest tests/ -v
```

Подробнее о том, какие тесты требуют инфраструктуру — [testing.md](testing.md).

## Миграции БД

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic revision --autogenerate -m "описание"
```
