# syntax=docker/dockerfile:1.7

# ========== STAGE 1: BUILDER — только зависимости ==========
# Код приложения сюда намеренно НЕ копируется.
#
# Было: `COPY app/` стоял до `uv sync` без --no-install-project, то есть
# установки самого проекта в окружение. Правка любого файла в app/ перезапускала
# эту установку, uv перезаписывал в .venv файлы editable-установки (`.pth`,
# finder, `RECORD`), содержимое .venv менялось — и Docker заново собирал слой
# на ~6 ГБ. Разделения COPY для этого не хватало: инвалидировался источник.
FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

# `--extra tracing` — инструментеры LlamaIndex и LangChain/LangGraph: без них
# Phoenix получает только вызовы OpenAI-SDK, а трейсов RAG и агента нет.
# `--no-install-project` — ставим только сторонние пакеты: сам проект в образе
# не нужен, `app` и `bot` импортируются из /app (см. PYTHONPATH ниже).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra tracing

# ========== STAGE 2: RUNTIME ==========
FROM python:3.13-slim-bookworm

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Окружение — отдельным слоем и только из stage зависимостей. Правка кода его
# больше не инвалидирует: в builder код не заходит, а в этот COPY попадает
# лишь .venv, который от кода не зависит.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Код и данные приложения — прямо из контекста сборки, мелкими слоями: они
# меняются чаще всего и теперь пересобирают только себя.
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser bot/ ./bot/
# Скрипты (ingest, smoke, eval) нужны внутри образа: их запускают через
# `docker compose exec app python scripts/...` — пути к корпусу те же (/app/data).
COPY --chown=appuser:appuser scripts/ ./scripts/
# alembic.ini и migrations нужны entrypoint'у (`alembic upgrade head`).
# migrations — отдельной строкой: `COPY migrations/ ./` скопировал бы
# СОДЕРЖИМОЕ каталога в /app (env.py и versions/ легли бы в корень), а нужен
# сам каталог /app/migrations. Собирать в один COPY с файлами нельзя.
COPY --chown=appuser:appuser alembic.ini entrypoint.sh pyproject.toml uv.lock ./
COPY --chown=appuser:appuser migrations/ ./migrations/

# Каталоги для named-томов (docstore /var, кэш HuggingFace) создаём заранее
# с владельцем appuser: иначе том примонтируется с root-владельцем, и не-root
# процесс получит PermissionError при скачивании E5 / записи docstore.
RUN mkdir -p /app/var /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app/var /home/appuser/.cache && \
    chmod +x /app/entrypoint.sh

# PYTHONPATH=/app нужен из-за отказа от editable-установки проекта: uvicorn и
# `python -m bot` кладут рабочий каталог в sys.path сами, а скрипты запускаются
# как `python scripts/ingest.py` — там sys.path[0] это /app/scripts, и без
# PYTHONPATH импорт `app.*` в `make ingest` / `make eval` сломается.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/ready', timeout=3).status == 200 else sys.exit(1)" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
