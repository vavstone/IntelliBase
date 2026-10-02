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

### A/B по чанкингу: отдельная коллекция

Смена chunk_size — это переиндексация (офлайн-контур). Нужна **отдельная коллекция и
отдельный docstore**, иначе UPSERTS пропустит всё по хешу (chunk_size в хеш не входит):

```bash
RAG_COLLECTION=rag_block_05_chunk1024 \
RAG_CHUNK_SIZE=1024 \
RAG_DOCSTORE_PATH=var/rag_docstore_chunk1024.json \
uv run python scripts/ingest.py data/kb
```

### Трейсинг LlamaIndex в Phoenix

```bash
PHOENIX_ENABLED=true uv run --extra tracing python scripts/trace_demo.py
# → http://localhost:6006 → Traces (retriever scores, LLM prompt/usage)
```

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
