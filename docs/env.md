# Переменные окружения

Источник истины — `app/core/config.py` (pydantic-settings, префиксы и `__` для
вложенных групп). Шаблон — `.env.example` (реальные ключи — в `.env`, в git не попадает).

> ⚠️ Известное расхождение (сверить при настройке):
> - `LLM__DEFAULT_MODEL`: дефолт `qwen2.5:3b` (config.py) vs `gemma3:4b` (.env.example).

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
| `LLM__DEFAULT_PROVIDER` | `ollama` | `ollama` / `openai` / `openrouter` |
| `LLM__DEFAULT_MODEL` | `qwen2.5:3b` | модель чата по умолчанию |
| `LLM__OLLAMA_BASE_URL` | `http://localhost:11434/v1` | эндпоинт Ollama |
| `LLM__OPENAI_BASE_URL` | `https://api.openai.com/v1` | эндпоинт OpenAI |
| `LLM__OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | эндпоинт OpenRouter |
| `LLM__REQUEST_TIMEOUT` | `30.0` | таймаут LLM-вызовов (сек) |
| `LLM__MAX_RETRIES` | `3` | число ретраев |

## Эмбеддинги (`EMBEDDING__*`)

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `EMBEDDING__PROVIDER` | `sentence_transformers` | `sentence_transformers` (локально) / `openai` |
| `EMBEDDING__MODEL` | `intfloat/multilingual-e5-large` | модель (dim 1024) |
| `EMBEDDING__BATCH_SIZE` | `32` | размер батча (16–32 на CPU) |
| `EMBEDDING__CACHE_DIR` | `./var/embedding_cache` | diskcache между рестартами |
| `EMBEDDING__MAX_RETRIES` | `5` | ретраи |

## Хранилище чата и кэш

| Переменная | Дефолт | Назначение |
|-----------|--------|------------|
| `DATABASE_URL` | `…localhost:5432/intellibase` | Postgres (asyncpg) |
| `CHAT_REPOSITORY` | `json` | `json` (файлы) / `postgres` |
| `CHAT_STORAGE_DIR` | `./var/chats` | путь для JSONL при `json` |
| `CHAT_CONTEXT_WINDOW` | `10` | sliding window (сообщений) |
| `REDIS_URL` | `redis://localhost:6379/0` | кэш LLM |
| `CACHE_TTL_SECONDS` | `3600` | TTL кэша (сек) |
| `PROXY_URL` | — | прокси для внешних API |

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
| `RAG_DATA_DIR` | `data/kb` | корпус для индексации |
| `RAG_COLLECTION` | `rag_block_05` | рабочая коллекция LlamaIndex |
| `RAG_COLLECTION_BARE` | `rag_block_03_bare` | bare-metal сравнение (Б5.3) |
| `RAG_DOCSTORE_PATH` | `var/rag_docstore.json` | состояние инкрементальной индексации |
| `RAG_LLM_MODEL` | `gemma3:4b` | LLM генерации RAG-ответа |
| `RAG_LLM_TIMEOUT` | `120` | таймаут генерации (сек) |
| `RAG_LLM_CONTEXT_WINDOW` | `8192` | контекстное окно LLM |
| `RAG_TOP_K` | `10` | ширина retrieval (similarity_top_k) |
| `RAG_CHUNK_SIZE` | `512` | размер чанка |
| `RAG_CHUNK_OVERLAP` | `64` | перекрытие чанков |
| `RAG_SCORE_THRESHOLD` | `0.80` | порог score-guard |
| `RAG_RERANK_ENABLED` | `false` | включить реранкер |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | модель реранкера |
| `RAG_RERANK_TOP_N` | `5` | top-N в промпт |
| `RAG_ENABLE_CHAT` | `true` | диалоговый RAG в `/chats` |
| `RAG_CONDENSE_ENABLED` | `true` | condense follow-up |

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
| `MODERATION_USE_OPENAI` | `true` | включить OpenAI Moderation API |
| `RATE_LIMIT_MESSAGES_PER_MIN` | `15` | лимит сообщений/мин на владельца |
| `APP_NAME` | `llm-service-example` | имя приложения |
| `CORS_ORIGINS` | `["*"]` | список origin |
| `DEBUG` | `false` | отладка |
