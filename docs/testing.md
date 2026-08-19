# Тесты

Структура зеркалит исходный код (`app/chat/` → `tests/chat/`). Запуск — из корня
проекта. Асинхронные тесты помечены `@pytest.mark.asyncio`.

## Канонические команды

```bash
# Быстрый прогон — без интеграционных тестов, требующих PG/Qdrant/Ollama
uv run pytest tests/ -v \
  --ignore=tests/chat/test_routes.py \
  --ignore=tests/chat/test_service_context.py

# Полный прогон — нужна поднятая инфраструктура
uv run pytest tests/ -v
```

## Группы тестов

| Группа | Путь | Что покрывает | Инфраструктура |
|--------|------|---------------|----------------|
| unit | `tests/unit/` | LLM-сервис (моки), PII, схемы, чанкинг, ingestion (чистые функции), reranker | в основном нет |
| chat | `tests/chat/` | роуты, контекст, промпты, RAG-диалог, контракт репозитория | частично да |
| bot | `tests/bot/` | админ, backend_client, FSM, streaming | нет (моки) |
| admin | `tests/admin/` | админ-роуты, rag-репозиторий | частично |
| moderation | `tests/moderation/` | сервис модерации | нет |
| ratelimit | `tests/ratelimit/` | rate limiting | нет |
| app/chat | `tests/app/chat/` | обработка медиа (whisper и т.п.) | нет |
| корневые | `tests/test_*.py` | categories, documents, rag, embeddings, vector_store, token_count | частично да |

## Тесты, требующие инфраструктуры

Запускать отдельно или с поднятыми сервисами (Postgres / Qdrant / Ollama):

- `tests/chat/test_routes.py`, `tests/chat/test_service_context.py` — Postgres/lifespan (исключены из «быстрого» прогона).
- `tests/chat/test_repository_contract.py` — контракт репозитория (Postgres).
- `tests/chat/test_rag_chat.py` — диалоговый RAG (Ollama + Qdrant, тяжёлый).
- `tests/test_vector_store.py` — Qdrant.
- `tests/test_categories.py`, `tests/test_documents.py` — Postgres (`kb_categories`, индексация).
- `tests/test_embeddings.py`, `tests/unit/test_reranker.py` — грузят sentence-transformers модели (~ГБ при первом запуске).

## Примечания

- Общие фикстуры — `tests/conftest.py`; модульные — свои `conftest.py`.
- `tests/unit/test_ingestion.py` покрывает чистые функции (`clean`, `category_from_path`, …) без внешних сервисов.
- Полный список тестов и маркеры — через `uv run pytest tests/ --collect-only -q`.
