# IntelliBase — единая точка входа: запуск, проверка, индексация, тесты.
#
# Требуется Docker с плагином compose. Локальные цели (test, ingest, eval)
# дополнительно требуют uv с установленным окружением.
#
#   make up      поднять стек одной командой
#   make smoke   проверить, что система жива и отвечает
#   make down    остановить стек
#   make help    список всех целей

# Рецепты — одиночные команды без shell-логики (if/then, awk, grep): на Windows
# make нередко исполняет их через cmd.exe, и любая POSIX-конструкция валит цель
# целиком. Находка проверки чистого клона 08.10: `make up` работал, а `make smoke`
# (там была логика с if/awk) падал, не дойдя до самой проверки. Поэтому вся
# логика переехала в scripts/smoke.py, а список целей для `help` продублирован
# текстом, а не собирается грепом из `##`-комментариев.
#
# Тексты, которые печатают сами цели, — латиницей: русские строки в UTF-8-файле
# make на Windows перекодирует в кодировку консоли и они выходят кракозябрами
# (вывод самих скриптов печатает python — он с этим справляется, поэтому там
# русский остался). Комментарии можно писать по-русски: они не печатаются.
SHELL := /bin/sh
.DEFAULT_GOAL := help

COMPOSE ?= docker compose
UV ?= uv
# Корпус для индексации (переопределяется: make ingest CORPUS=data/kb).
CORPUS ?= data/demo_kb
# Дополнительные аргументы smoke (например: make smoke SMOKE_ARGS=--with-rag).
SMOKE_ARGS ?=
# Дополнительные аргументы metrics (например, METRICS_ARGS="--window-hours 168").
METRICS_ARGS ?=
# Дополнительные аргументы eval (например, EVAL_ARGS="--concurrency 4").
EVAL_ARGS ?=
# Golden dataset для `make eval`: по умолчанию демо-корпус под стать RAG_DATA_DIR.
# Рабочий корпус: make eval GOLDEN=tests/eval/golden_dataset.json LABEL=block_05
GOLDEN ?= tests/eval/golden_dataset_demo.json
LABEL ?= demo

.PHONY: help up down restart ps logs smoke smoke-rag smoke-host test test-all ingest reindex eval metrics thresholds shell clean

help: ## Список целей
	@echo IntelliBase targets:
	@echo   up          start the stack (app + bot + infra), wait until ready
	@echo   smoke       liveness: app, bot, /health, /ready, Qdrant, Phoenix
	@echo   smoke-rag   smoke + end-to-end RAG question (needs LLM and corpus)
	@echo   smoke-host  smoke when the app runs on the host (uvicorn outside Docker)
	@echo   down        stop the stack (data in volumes is kept)
	@echo   restart     restart app and bot without rebuild
	@echo   ps          service status
	@echo   logs        app and bot logs (Ctrl+C to exit)
	@echo   test        fast tests: no external API, infra-less skips
	@echo   test-all    full run incl. integration (needs stack and network)
	@echo   ingest      incremental corpus indexing
	@echo   reindex     full reindex (wipes collection and docstore)
	@echo   eval        RAG quality (RAGAS, needs uv)
	@echo   metrics     p95, cache hit rate, last RAGAS run
	@echo   thresholds  check the last RAGAS run against thresholds
	@echo   users       bot access: make users ARGS=list
	@echo   shell       bash inside the app container
	@echo   clean       stop the stack and delete volumes (CAREFUL)

up: ## Поднять стек (app + bot + инфраструктура) и дождаться готовности
	$(COMPOSE) up -d --build --wait
	@echo Stack is up. Check: make smoke

down: ## Остановить стек (данные в томах сохраняются)
	$(COMPOSE) down

restart: ## Перезапустить app и bot без пересборки
	$(COMPOSE) restart app bot

ps: ## Статус сервисов
	$(COMPOSE) ps

logs: ## Логи app и bot (Ctrl+C — выйти)
	$(COMPOSE) logs -f app bot

smoke: ## Проверить живость стека: сервисы, health, Qdrant, Phoenix
	$(COMPOSE) exec -T -e SMOKE_PHOENIX_URL=http://phoenix:6006 app python scripts/smoke.py $(SMOKE_ARGS)

smoke-rag: ## Smoke + сквозной вопрос к RAG (нужны LLM и наполненный корпус)
	@$(MAKE) smoke SMOKE_ARGS=--with-rag

smoke-host: ## То же, когда приложение запущено на хосте (uvicorn вне Docker)
	$(UV) run python scripts/smoke.py $(SMOKE_ARGS)

test: ## Быстрые тесты (без интеграционных; PG/Qdrant сами скипаются)
	$(UV) run pytest tests/ -q -m "not integration"

test-all: ## Полный прогон тестов, включая интеграционные (нужны стек и сеть)
	$(UV) run pytest tests/ -q

ingest: ## Инкрементальная индексация корпуса (CORPUS=data/demo_kb)
	$(UV) run python scripts/ingest.py $(CORPUS)

reindex: ## Полная переиндексация: чистит коллекцию Qdrant и docstore
	$(UV) run python scripts/ingest.py $(CORPUS) --full

eval: ## Оценка качества RAG (RAGAS) на демо-корпусе: нужны LLM и стек
	$(UV) run --extra eval python scripts/run_eval.py --golden $(GOLDEN) --label $(LABEL) $(EVAL_ARGS)

metrics: ## Метрики для демо: p95, cache hit rate, последний RAGAS (нужен стек)
	$(UV) run python scripts/metrics.py $(METRICS_ARGS)

thresholds: ## Пороги качества по последнему прогону RAGAS (гейт eval/check_thresholds.py)
	$(UV) run python eval/check_thresholds.py

users: ## Доступ к боту: make users ARGS='list' | ARGS='add 123456789 "Иванов Пётр"'
	$(UV) run python scripts/bot_users.py $(ARGS)

shell: ## Bash внутри контейнера app
	$(COMPOSE) exec app bash

clean: ## ОСТОРОЖНО: остановить стек и удалить тома (БД, Qdrant, docstore)
	$(COMPOSE) down -v
