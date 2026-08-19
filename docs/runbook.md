# Runbook — операционные команды

Копипаст-команды для запуска, проверки и переиндексации. Все — из корня проекта.
Порты и сервисы — см. [../ARCHITECTURE.md](../ARCHITECTURE.md).

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
> `qwen2.5:3b` (чат), `gemma3:4b` (RAG).

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

```bash
# Инкрементально (UPSERTS пропустит неизменённое)
uv run python scripts/ingest.py data/kb

# Полная переиндексация (вычистить и заново)
uv run python scripts/ingest.py data/kb --full

# Точечно — только перечисленные файлы
uv run python scripts/ingest.py --files data/kb/tarify/a.pdf data/kb/malahit/b.docx
```

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
curl -s -X DELETE http://localhost:6333/collections/rag_block_05
rm -f var/rag_docstore.json
uv run python scripts/ingest.py data/kb --full
```

## Smoke-тест RAG

```bash
curl -s -m 120 -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Что входит в состав КПС Тарифы?","category":"tarify"}'
```

Ожидается JSON с `answer`, `sources` (нумерованные цитаты), `confident`.
Off-topic вопрос → `confident=false` + честный отказ без вызова LLM.

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
