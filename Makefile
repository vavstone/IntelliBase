# IntelliBase — единая точка входа: запуск, проверка, индексация, тесты.
#
# Требуется Docker с плагином compose. Локальные цели (test, ingest, eval)
# дополнительно требуют uv с установленным окружением.
#
#   make up      поднять стек одной командой
#   make smoke   проверить, что система жива и отвечает
#   make down    остановить стек
#   make help    список всех целей

# Рецепты написаны на POSIX sh: работает и в минимальных образах, где bash нет.
SHELL := /bin/sh
.DEFAULT_GOAL := help

COMPOSE ?= docker compose
UV ?= uv
# Корпус для индексации (переопределяется: make ingest CORPUS=data/kb).
CORPUS ?= data/demo_kb
# Дополнительные аргументы smoke (например, SMOKE_ARGS="--with-rag").
SMOKE_ARGS ?=
# Дополнительные аргументы metrics (например, METRICS_ARGS="--window-hours 168").
METRICS_ARGS ?=
# Дополнительные аргументы eval (например, EVAL_ARGS="--concurrency 4").
EVAL_ARGS ?=
# Golden dataset для `make eval`: по умолчанию демо-корпус под стать RAG_DATA_DIR.
# Рабочий корпус: make eval GOLDEN=tests/eval/golden_dataset.json LABEL=block_05
GOLDEN ?= tests/eval/golden_dataset_demo.json
LABEL ?= demo

.PHONY: help up down restart ps logs smoke smoke-rag test test-all ingest reindex eval metrics thresholds shell clean

help: ## Список целей
	@echo "IntelliBase — доступные команды:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Поднять стек (app + bot + инфраструктура) и дождаться готовности
	$(COMPOSE) up -d --build --wait
	@echo "Стек поднят. Проверка: make smoke"

down: ## Остановить стек (данные в томах сохраняются)
	$(COMPOSE) down

restart: ## Перезапустить app и bot без пересборки
	$(COMPOSE) restart app bot

ps: ## Статус сервисов
	$(COMPOSE) ps

logs: ## Логи app и bot (Ctrl+C — выйти)
	$(COMPOSE) logs -f app bot

smoke: ## Проверить живость стека: контейнеры, health, Qdrant, Phoenix
	@if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx app; then \
		unhealthy=$$($(COMPOSE) ps --format '{{.Service}} {{.State}} {{.Health}}' \
			| awk '$$2 != "running" || ($$3 != "" && $$3 != "healthy") \
				{ print $$1 ":" $$2 "/" $$3 }'); \
		$(COMPOSE) exec -T -e SMOKE_PHOENIX_URL=http://phoenix:6006 \
			-e SMOKE_CONTAINERS="$$unhealthy" \
			app python scripts/smoke.py $(SMOKE_ARGS); \
	else \
		$(UV) run python scripts/smoke.py $(SMOKE_ARGS); \
	fi

smoke-rag: ## Smoke + сквозной вопрос к RAG (нужны LLM и наполненный корпус)
	@$(MAKE) smoke SMOKE_ARGS="--with-rag"

test: ## Быстрые тесты (без интеграционных, требующих PG/lifespan)
	$(UV) run pytest tests/ -q \
		--ignore=tests/chat/test_routes.py \
		--ignore=tests/chat/test_service_context.py

test-all: ## Полный прогон тестов (нужна поднятая инфраструктура)
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

shell: ## Bash внутри контейнера app
	$(COMPOSE) exec app bash

clean: ## ОСТОРОЖНО: остановить стек и удалить тома (БД, Qdrant, docstore)
	$(COMPOSE) down -v
