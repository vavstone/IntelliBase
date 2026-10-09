# Тесты

Структура зеркалит исходный код (`app/chat/` → `tests/chat/`). Запуск — из корня
проекта. Асинхронные тесты помечены `@pytest.mark.asyncio`.

## Канонические команды

```bash
# Быстрый прогон — без интеграционных (external API); тесты, которым нужны
# PG/Qdrant, скипаются сами, если сервисы не подняты
uv run pytest tests/ -v -m "not integration"

# Полный прогон — включая интеграционные (нужны стек и сеть)
uv run pytest tests/ -v
```

Те же команды под Makefile: `make test` (быстрый) и `make test-all` (полный).

## Группы тестов

| Группа | Путь | Что покрывает | Инфраструктура |
|--------|------|---------------|----------------|
| unit | `tests/unit/` | LLM-сервис (моки), фильтры безопасности (`test_security_filters.py`), схемы, чанкинг, ingestion (чистые функции), reranker, конфиг (пустой `PROXY_URL` → `None`), HTTP-слой агента (`test_agent_routes.py`) | в основном нет |
| chat | `tests/chat/` | роуты `/chats` (SSE, история, feedback), контекст и ошибки сервиса, промпты, RAG-диалог, контракт репозитория | PG-часть контракта скипается без Postgres |
| bot | `tests/bot/` | админ, backend_client, FSM, streaming, HIL-кнопки агента (`test_agent_hil.py`) | нет (моки) |
| admin | `tests/admin/` | админ-роуты, rag-репозиторий | частично |
| moderation | `tests/moderation/` | сервис модерации | нет |
| ratelimit | `tests/ratelimit/` | rate limiting | нет |
| app/chat | `tests/app/chat/` | обработка медиа (whisper и т.п.) | нет |
| корневые | `tests/test_*.py` | category, documents, rag, agent (HIL + устойчивость цикла), embeddings, vector_store, token_count | частично |

## Тесты, требующие инфраструктуры

Запускать отдельно или с поднятыми сервисами (Postgres / Qdrant):

- `tests/chat/test_repository_contract.py` — контракт репозитория; PG-ветка параметризации скипается без живого Postgres, JSON-ветка идёт всегда.
- `tests/test_vector_store.py` — Qdrant (скипается без живого Qdrant).
- `tests/test_categories.py`, `tests/test_documents.py` — Postgres (`kb_categories`, индексация).
- `tests/test_embeddings.py` — тяжёлые проверки E5 за гейтом `MULTILINGUAL_MODEL=1` (плюс один вечный `skip`); базовые — без модели.
- `tests/test_token_count.py` — реальные вызовы OpenAI API, помечен `@pytest.mark.integration` (исключён из быстрого прогона); без ключа или сети — `skip`.

## Примечания

- Общие фикстуры — `tests/conftest.py`; модульные — свои `conftest.py`.
- `tests/unit/test_ingestion.py` покрывает чистые функции (`clean`, `category_from_path`, …) без внешних сервисов, включая PDF-движок: `pages_from_inspector` (заглушка модуля `pdf_inspector` через `monkeypatch`) — постраничные Document'ы, пропуск страниц под OCR, откат на PyMuPDF при отсутствии пакета.
- Полный список тестов и маркеры — через `uv run pytest tests/ --collect-only -q`.
