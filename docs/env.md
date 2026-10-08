# Переменные окружения

Источник истины — `app/core/config.py` (pydantic-settings, префиксы и `__` для
вложенных групп). Шаблон — `.env.example` (реальные ключи — в `.env`, в git не попадает).

## Служебные токены (сменить перед продом)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `ADMIN_TOKEN` | `change-me-admin` ⚠️ | заголовок `X-Admin-Token` для `/chats/admin/*` |
| `INTERNAL_TOKEN` | `change-me-internal` ⚠️ | заголовок `X-Internal-Token` (backend↔bot) |
| `LLM__OPENAI_API_KEY` | `sk-test-placeholder` | ключ OpenAI |
| `LLM__OPENROUTER_API_KEY` | `sk-test-placeholder` | ключ OpenRouter |

Генерация надёжных токенов: `openssl rand -hex 32`.

## LLM (`LLM__*`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `LLM__DEFAULT_PROVIDER` | `ollama` | `ollama` / `openai` / `openrouter` / `deepseek` |
| `LLM__DEFAULT_MODEL` | `gemma3:4b` | модель чата по умолчанию (Ollama; облако — `deepseek-v4-flash`) |
| `LLM__OLLAMA_BASE_URL` | `http://localhost:11434/v1` | эндпоинт Ollama |
| `LLM__OPENAI_BASE_URL` | `https://api.openai.com/v1` | эндпоинт OpenAI |
| `LLM__OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | эндпоинт OpenRouter |
| `LLM__DEEPSEEK_API_KEY` | `sk-test-placeholder` | ключ DeepSeek (OpenAI-совместимый, без VPN) |
| `LLM__DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | эндпоинт DeepSeek |
| `LLM__REQUEST_TIMEOUT` | `30.0` | таймаут LLM-вызовов (сек) |
| `LLM__MAX_RETRIES` | `3` | число ретраев |
| `LLM__FALLBACK_PROVIDER` | `ollama` | резерв при недоступности основного провайдера (чат, синтез RAG, агент); пусто — выключить |
| `LLM__FALLBACK_MODEL` | `qwen2.5:3b` | модель резерва (у провайдеров разные имена моделей) |

**Резерв (`LLM__FALLBACK_*`).** Подключается, когда основной провайдер
недоступен: нет сети, таймаут, 429, отклонённый ключ. Ретраи (3 попытки с
backoff) при этом отрабатывают первыми — резерв включается после них. Ошибки
фильтра контента и 400 на резерв не переводятся: это не сбой доступности
(см. `app/services/llm_fallback.py`). Если резерв совпадает с основным
провайдером, подмена не срабатывает.

## Эмбеддинги (`EMBEDDING__*`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` (локально) / `openai` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | модель (dim 1024) |
| `EMBEDDING_BATCH_SIZE` | `32` | размер батча (16–32 на CPU) |
| `EMBEDDING_CACHE_DIR` | `./var/embedding_cache` | diskcache между рестартами |
| `EMBEDDING_MODEL_CACHE_DIR` | — | каталог весов в формате HF-кэша; в Docker — смонтированный кэш (иначе модель ищется в кэше LlamaIndex внутри контейнера и RAG не поднимается) |
| `EMBEDDING_MAX_RETRIES` | `5` | ретраи |

## Хранилище чата и кэш

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `DATABASE_URL` | `…localhost:5432/intellibase` | Postgres (asyncpg) |
| `CHAT_REPOSITORY` | `json` | `json` (файлы) / `postgres` |
| `CHAT_STORAGE_DIR` | `./var/chats` | путь для JSONL при `json` |
| `CHAT_CONTEXT_WINDOW` | `10` | sliding window (сообщений) |
| `REDIS_URL` | `redis://localhost:6379/0` | кэш LLM |
| `CACHE_TTL_SECONDS` | `3600` | TTL кэша (сек) |
| `PROXY_URL` | — | прокси для внешних API (OpenAI/OpenRouter); DeepSeek ходит напрямую; пусто = без прокси (пустая строка приводится к `None`) |

## Qdrant

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `QDRANT_URL` | `http://localhost:6333` | адрес Qdrant |
| `QDRANT_API_KEY` | — | ключ (в dev не требуется) |
| `QDRANT_COLLECTION` | `documents` | коллекция `VectorStore` |
| `EMBEDDING_DIM` | `1024` | размерность векторов |

## RAG (`RAG_*`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `RAG_DATA_DIR` | `data/demo_kb` | корпус для индексации (демо-набор в git; рабочий корпус — `data/kb`) |
| `RAG_COLLECTION` | `rag_demo` | рабочая коллекция LlamaIndex (под демо-корпус; под рабочий — `rag_block_05`) |
| `RAG_COLLECTION_BARE` | `rag_block_03_bare` | bare-metal сравнение (Б5.3) |
| `RAG_DOCSTORE_PATH` | `var/rag_docstore.json` | состояние инкрементальной индексации |
| `RAG_LLM_MODEL` | `qwen3:8b` | LLM генерации RAG-ответа (выбор Б5.6) |
| `RAG_LLM_TIMEOUT` | `600` | таймаут генерации (сек; qwen3:8b на CPU до ~5 мин) |
| `RAG_LLM_CONTEXT_WINDOW` | `8192` | контекстное окно LLM |
| `RAG_TOP_K` | `10` | ширина retrieval (similarity_top_k) |
| `RAG_CHUNK_SIZE` | `512` | размер чанка |
| `RAG_CHUNK_OVERLAP` | `64` | перекрытие чанков |
| `RAG_PDF_PARSER` | `inspector` | движок извлечения PDF: `inspector` (pdf-inspector, Rust) или `pymupdf` (legacy-откат) |
| `RAG_SCORE_THRESHOLD` | `0.80` | порог score-guard |
| `RAG_RERANK_ENABLED` | `false` | включить реранкер |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | модель реранкера |
| `RAG_RERANK_TOP_N` | `5` | top-N в промпт |
| `RAG_ENABLE_CHAT` | `true` | диалоговый RAG в `/chats` |
| `RAG_CONDENSE_ENABLED` | `true` | condense follow-up |

## Оценка качества RAGAS (`EVAL_JUDGE_*`, группа `eval`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `EVAL_JUDGE_PROVIDER` | `deepseek` | провайдер судьи метрик: `deepseek` / `openai` / `anthropic` |
| `EVAL_JUDGE_MODEL` | `deepseek-v4-flash` | модель судьи (RAGAS + TestsetGenerator) |
| `ANTHROPIC_API_KEY` | — | ключ, если `EVAL_JUDGE_PROVIDER=anthropic` |

Эмбеддинги для `AnswerRelevancy` — локальная `EMBEDDING_MODEL` (E5), не облако.

## Phoenix-трейсинг (группа `tracing`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `PHOENIX_ENABLED` | `false` | включить инструментеры OpenAI, LangChain/LangGraph, LlamaIndex и FastAPI |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP-эндпоинт Phoenix (HTTP; порт `:4317` — gRPC, протокол выводится из URL) |
| `PHOENIX_EXCLUDED_URLS` | `/health,/ready,/openapi.json,/docs,/redoc,/access,^https?://[^/]+/chats$` | пути, исключённые из трейсинга (healthcheck'и раз в 30 с вытесняют осмысленные трейсы, `/access` бот дёргает на каждом апдейте). Шаблоны — regex, ищутся в полном URL через `re.search`: чтобы исключить только `POST /chats`, нужны якоря `^…$` — без них шаблон `/chats` выключит и `/chats/{id}/messages` |

В контейнере флаг включён в `compose.yaml` (`PHOENIX_ENABLED=true`, эндпоинт
`http://phoenix:4317` — gRPC внутри сети compose): образ собирается с
`--extra tracing`, а сервис `phoenix` поднимается в том же compose. Для запуска
на хосте нужен `uv sync --extra tracing` и поднятый Phoenix на `:6006`.

## Агентный слой (`AGENT_*`, LangGraph)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `AGENT_CHECKPOINTER` | `sqlite` | хранилище чек-пойнтов: `memory` / `sqlite` / `postgres` |
| `AGENT_CHECKPOINTER_POSTGRES_URI` | `postgresql://chat:pswd@localhost:5432/intellibase` | URI для `AsyncPostgresSaver` (psycopg v3, НЕ asyncpg) |
| `AGENT_SQLITE_PATH` | `var/agent_checkpoints.sqlite` | файл SQLite-чекпоинтера при `AGENT_CHECKPOINTER=sqlite` |
| `BOT_ALLOWED_CHAT_IDS` | — | bootstrap-список разрешённых пользователей бота (id через запятую) |

## Кто имеет доступ к боту (`bot_users`)

Корпоративный бот закрыт: список тех, кому он отвечает, лежит в БД (таблица
`bot_users`, админ-API `/chats/admin/bot-users`). **Пустая таблица = бот не отвечает
никому**; исключение — id из `BOT_ALLOWED_CHAT_IDS` (bootstrap для чистого клона и
аварийный доступ, если список в БД испорчен).

Один и тот же список решает две задачи:

- **кому бот отвечает** — гейт на входе (`bot/middlewares/access.py` → `GET /access/{chat_id}
  `): посторонний не доходит ни до `/ask`, ни до `/agent`; в отказе виден его id,
  чтобы передать администратору;
- **кому агент может отправлять** — получателя выбирает не модель: chat_id из tool_call
  сверяется с `bot_users` ∪ чат-инициатор запроса; чужой адрес приводит не к отправке,
  а к ошибке инструмента. Имя из `title` уходит в превью подтверждения, и по нему же
  работает «отправь Иванову» (инструмент `find_recipient`).

Чат-инициатор разрешён всегда и от БД не зависит — «отправь мне» переживает
недоступность Postgres.

## Внутренний канал app → бот (`BOT_URL`, `INTERNAL_TOKEN`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `BOT_URL` | `http://bot:9000` | HTTP-API бота для `POST /notify` (handoff, рассылки, отправка агентом); в контейнере compose задаётся как `http://bot:9000`, в `.env` — `http://localhost:9000` для запуска на хосте |
| `INTERNAL_TOKEN` | `change-me-internal` | общий секрет app ↔ bot (`X-Internal-Token`) |

## Бот (`bot/config.py`, префикс `BOT_`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `BOT_TOKEN` | — | токен @BotFather |
| `BACKEND_URL` | `http://app:8000` | адрес бэкенда |
| `BOT_ADMIN_IDS` | — | ID админов (через запятую) |
| `BOT_API_PORT` | `9000` | порт HTTP-сервера бота (`/notify`) |
| `ADMIN_CHAT_ID` | — | чат для alert drain |
| `BOT_URL` | `http://bot:9000` | адрес бота для вызовов `/notify` |

## Прочее

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `MODERATION_USE_OPENAI` | `false` | второй слой модерации — OpenAI Moderation API; выключен по умолчанию: внешний вызов, нужен ключ и выход в сеть. Regex-слой работает всегда |
| `RATE_LIMIT_MESSAGES_PER_MIN` | `15` | лимит сообщений/мин на владельца |
| `APP_NAME` | `llm-service-example` | имя приложения |
| `CORS_ORIGINS` | `["*"]` | список origin |
| `DEBUG` | `false` | отладка |
| `SSL_CERT_FILE` | — | CA-бандл для `httpx`/OpenSSL; нужен, если антивирус инспектирует TLS и из контейнера (`CERTIFICATE_VERIFY_FAILED`) — команды в `docs/runbook.md`, «Антивирус с TLS-инспекцией» |
| `REQUESTS_CA_BUNDLE` | — | то же для `requests` (через него качает `huggingface_hub`); указывать на тот же комбинированный бандл, что и `SSL_CERT_FILE` |

## Тихий режим HuggingFace (`app/core/hf_env.py`)

Ставятся кодом (`os.environ.setdefault`) на старте любого процесса, который импортирует
`app.*`, и до того, как `huggingface_hub` подтянет langchain/transformers. В `.env` их
писать не нужно — они уже выставлены; переопределять есть смысл только для отладки
загрузки моделей.

| Переменная | Значение по умолчанию | Назначение |
|-----------|--------|------------|
| `HF_HUB_VERBOSITY` | `error` | убирает предупреждение «You are sending unauthenticated requests to the HF Hub» |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | убирает прогресс-бар загрузки |
| `TRANSFORMERS_VERBOSITY` | `error` | убирает прогресс-бар «Loading weights» при загрузке embedding-модели |
| `HF_CACHE_DIR` | — | путь к уже скачанному кэшу моделей на хосте: compose монтирует его в контейнер (`/home/appuser/.cache/huggingface`), и E5 (2.2 ГБ) не скачивается заново |
| `HF_HUB_OFFLINE` | — | `1` запрещает любые сетевые обращения к huggingface.co: модель берётся только из локального кэша. Обход для машин с нестабильным каналом до HuggingFace (см. `docs/runbook.md`, «Антивирус с TLS-инспекцией») |
