# Архитектура IntelliBase

> Корпоративная база знаний с ИИ-ассистентом для поиска и анализа документов (PDF, DOCX, HTML, MD).
> Этот документ описывает **текущее** состояние системы: модули, потоки данных, внешние сервисы.
> История решений (ADR) — в [docs/architecture.md](docs/architecture.md).

## Обзор

Система состоит из двух независимых сервисов:

- **FastAPI-бэкенд** (порт 8000) — REST API + SSE, векторный поиск (RAG) по базе знаний.
- **Telegram-бот** (aiogram, порт 9000) — клиент бэкенда; поднимает собственный HTTP-сервер для обратных вызовов (`POST /notify`).

```
┌──────────────┐     HTTP/SSE       ┌────────────────┐
│  aiogram Bot │ ────────────────→  │  FastAPI App   │
│  (port 9000) │ ←─── POST /notify ─│  (port 8000)   │
└──────────────┘                    └───────┬────────┘
                                            │
      ┌─────────────┬──────────────┬────────┼────────┬──────────┐
      ↓             ↓              ↓        ↓        ↓          ↓
 PostgreSQL 16   Redis 7.4   Arize Phoenix  Qdrant  Ollama  Embed-модель
 (чаты, кат-ии) (кэш LLM)   (трейсы LLM)  (вектора) (LLM)  (ST, локально)
```

## Карта модулей

### `app/` — бэкенд

| Модуль | Ответственность |
|--------|-----------------|
| `app/main.py` | Точка входа: lifespan (клиенты, фоновые таски, инициализация RAG), middleware, обработчики ошибок |
| `app/core/` | Конфигурация (pydantic-settings), доменные исключения, PII/security-паттерны |
| `app/deps/providers.py` | DI-зависимости (Settings, LLM-клиенты, Session, Cache, RAG/Ingestion) |
| `app/routers/` | Тонкие роутеры: chat (legacy), health, models, categories, documents, rag |
| `app/chat/` | Stateful-чат: домен, оркестратор (с RAG-путём), repository (JSON/PG), A/B промптов |
| `app/kb/` | Каталог категорий знаний (ПС): домен + репозиторий `kb_categories` |
| `app/services/rag.py` | Онлайн-контур RAG: retrieve → score-guard → генерация с цитатами |
| `app/services/rag_baremetal.py` | RAG без фреймворка (сравнение с LlamaIndex) |
| `app/services/ingestion.py` | Офлайн-контур: парсинг → чанкинг → эмбеддинг → UPSERTS в Qdrant |
| `app/services/vector_store.py` | Async-обёртка над Qdrant (коллекция `documents`) |
| `app/services/reranker.py` | Cross-encoder реранкер (опционально) |
| `app/services/chunking.py` | Стратегии чанкинга fixed/recursive/semantic (эксперимент) |
| `app/services/embeddings.py` | Embed-сервис (OpenAI API + sentence-transformers) |
| `app/moderation/` | Двухслойная модерация (regex → OpenAI API) |
| `app/ratelimit/` | Rate limiting (Redis, per owner) |
| `app/admin/` | Админ-API: статистика, рассылки, экспорт, handoff, алерты |
| `app/observability/` | structlog, Phoenix tracing, PII redaction |

### `bot/` — Telegram-бот

| Модуль | Ответственность |
|--------|-----------------|
| `bot/__main__.py` | Точка входа: aiogram polling + HTTP `/notify` + alert drain |
| `bot/handlers/` | commands, text, media, fsm (сценарий `/ask`), admin, feedback, handoff |
| `bot/services/` | backend_client (вызовы бэкенда), streaming (SSE), alert_drain, error_handling |
| `bot/keyboards/inline.py` | Inline-клавиатуры (в т.ч. выбор категории ПС) |

### `scripts/` — утилиты индексации

| Скрипт | Назначение |
|--------|------------|
| `ingest.py` | Полная индексация корпуса `data/kb` в Qdrant |
| `prepare_corpus.py` | Сборка корпуса из исходных документов |
| `load_to_qdrant.py` | Загрузка готового JSONL в Qdrant |
| `compare_metrics.py` | Сравнение cosine vs dot в Qdrant |
| `run_chunking_experiment.py` | Прогон эксперимента по чанкингу |

## Поток данных RAG

Два контура, разделённых по времени жизни запроса.

### Офлайн-контур (индексация) — `app/services/ingestion.py`

1. Чтение корпуса `data/kb/<category>/…` (PDF/DOCX/HTML/MD) — ридеры PyMuPDF (PDF постранично), Docx, HTML, Markdown.
2. Обогащение метаданными из пути и файла: `category` (папка-ПС), `version`, `visibility`, `last_modified` (стабильный, для идемпотентности).
3. Чанкинг `SentenceSplitter` (chunk 512 / overlap 64).
4. Эмбеддинг `intfloat/multilingual-e5-large` (dim 1024, локально, E5-префиксы `query:` / `passage:`).
5. Запись в коллекцию Qdrant `rag_block_05` через `IngestionPipeline` + `DocstoreStrategy.UPSERTS` — инкрементально, дедуп по детерминированному doc-id и hash.

### Онлайн-контур (запрос) — `app/services/rag.py`

1. **Retrieve** — top-K (по умолчанию 10) кандидатов из Qdrant, опционально со строгим фильтром по категории (ПС).
2. **Re-rank** (опционально) — cross-encoder `bge-reranker-v2-m3` пересортировывает кандидатов, остаётся top-N (по умолчанию 5).
3. **Score-guard** — если лучший score ниже порога `rag_score_threshold` (0.80) → честный отказ без вызова LLM.
4. **Synthesis** — генерация ответа по пронумерованному контексту с цитатами `[1]`, `[2]` через Ollama (`gemma3:4b`).

Диалоговый RAG в `/chats/{id}/messages` использует тот же онлайн-контур + опциональный **condense**: follow-up переписывается в самодостаточный поисковый запрос (один LLM-вызов), после чего retrieval идёт по переписанному тексту.

### Коллекции Qdrant

| Коллекция | Кто пишет | Назначение |
|-----------|-----------|------------|
| `rag_block_05` | `IngestionService` / `RAGService` | Рабочий RAG-индекс (LlamaIndex) |
| `documents` | `VectorStore` | Векторный поиск без LlamaIndex (Б5.2) |

## Внешние сервисы и порты

| Сервис | Порт | Назначение |
|--------|------|------------|
| FastAPI app | 8000 | REST + SSE |
| Bot HTTP | 9000 | `/notify` (обратные вызовы) |
| PostgreSQL | 5432 | чаты, категории, метрики |
| Redis | 6379 | кэш LLM |
| Qdrant | 6333 | векторный поиск |
| Ollama | 11434 | локальные LLM (chat: `qwen2.5:3b`, RAG: `gemma3:4b`) |
| Arize Phoenix | 6006 | трассировка LLM |

## Схема БД (PostgreSQL)

| Таблица | Назначение |
|---------|------------|
| `chats` / `chat_messages` | Чаты и сообщения (soft-delete, `sources` JSONB) |
| `system_prompts` | A/B-промпты |
| `message_feedback` | Оценки ответов |
| `request_metrics` | Метрики запросов |
| `broadcasts` / `alerts` | Рассылки / алерты |
| `rate_limits` | Счётчики rate-limit |
| `rag_queries` | Лог RAG-запросов (refusal_rate, пробелы в знаниях) |
| `kb_categories` | Категории знаний = ПС |

Миграции: Alembic, 8 версий в `migrations/versions/`. Переключение хранилища чата — `CHAT_REPOSITORY=json|postgres`.

## Ключевые настройки

Полный список — `app/core/config.py` и `.env.example`. Основные группы:

| Префикс | Назначение |
|---------|------------|
| `LLM__*` | LLM-провайдеры (Ollama/OpenAI/OpenRouter) |
| `EMBEDDING__*` | Embed-модель (provider, model, batch_size) |
| `RAG_*` | RAG: коллекция, top_k, score_threshold, rerank, condense |
| `QDRANT_*` / `EMBEDDING_DIM` | Векторное хранилище |
| `CHAT_REPOSITORY` | `json` (файлы) vs `postgres` |

## Безопасность

- Входной фильтр (`app/services/security/input_validator.py`): regex-паттерны инъекций, детект base64, проверка длины.
- Выходной фильтр (`app/services/security/output_filter.py`): canary-токен, forbidden phrases, маскирование PII.
- Модерация (`app/moderation/`): regex → OpenAI Moderation API (fail-open).
- Аутентификация: `X-Admin-Token` (админка), `X-Internal-Token` (bot↔backend), `X-Owner-External-Id` (пользователи).
