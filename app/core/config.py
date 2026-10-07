from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    openai_api_key: SecretStr = SecretStr("sk-test-placeholder")
    openrouter_api_key: SecretStr = SecretStr("sk-test-placeholder")
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # DeepSeek — OpenAI-совместимый эндпоинт, работает без VPN (альтернатива OpenAI).
    # Используется как провайдер чата/RAG (быстрее локальной Ollama на CPU),
    # судья RAGAS (см. EVAL_JUDGE_*) и модель агента (см. main.py).
    deepseek_api_key: SecretStr = SecretStr("sk-test-placeholder")
    deepseek_base_url: str = "https://api.deepseek.com"
    default_provider: Literal["openai", "ollama", "openrouter", "deepseek"] = "ollama"
    default_model: str = "qwen2.5:3b"
    request_timeout: float = 30.0
    max_retries: int = 3
    # Резерв на случай недоступности основного провайдера (нет сети, таймаут,
    # 429, протухший ключ) — см. app/services/llm_fallback.py. Пустая строка
    # выключает подмену. Модель задаётся отдельно: имена у провайдеров разные
    # (deepseek-v4-flash против qwen2.5:3b), одним полем не обойтись.
    fallback_provider: Literal["", "openai", "ollama", "openrouter", "deepseek"] = "ollama"
    fallback_model: str = "qwen2.5:3b"


class EmbeddingSettings(BaseSettings):
    """Настройки embedding-сервиса. Переключаются через EMBEDDING_PROVIDER и EMBEDDING_MODEL."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["openai", "sentence_transformers"] = "sentence_transformers"
    model: str = "intfloat/multilingual-e5-large"
    batch_size: int = 32  # Для ST на CPU — 16–32
    cache_dir: str = "./var/embedding_cache"
    # Каталог, откуда LlamaIndex грузит веса модели (в формате HF-кэша:
    # <dir>/models--<org>--<name>/snapshots/...). По умолчанию — собственный кэш
    # LlamaIndex (get_cache_dir(): ~/.cache/llama_index или AppData), который в
    # контейнере живёт в слое образа и теряется при пересоздании — тогда модель
    # пробует скачаться с huggingface.co. В Docker указываем смонтированный
    # HF-кэш (EMBEDDING_MODEL_CACHE_DIR=/home/appuser/.cache/huggingface/hub).
    model_cache_dir: str | None = None
    max_retries: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "llm-service-example"
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    proxy_url: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    database_url: str = "postgresql+asyncpg://chat:pswd@localhost:5432/intellibase"
    chat_repository: Literal["json", "postgres"] = "json"
    chat_storage_dir: Path = Path("./var/chats")
    chat_context_window: int = 10

    # Production ---------------------------------------------------------
    # X-Admin-Token для /chats/admin/*. Сменить на 32+ hex-байт через
    # `openssl rand -hex 32` в проде.
    admin_token: SecretStr = SecretStr("change-me-admin")
    # Service-to-service: backend ↔ bot (общий с bot /notify).
    internal_token: SecretStr = SecretStr("change-me-internal")
    # Базовый URL bot-сервиса (для broadcast и notify-вызовов из backend).
    bot_url: str = "http://bot:9000"
    # Включить OpenAI Moderation API (layer 2 каскада). Если False —
    # только regex-блоклист.
    # Второй слой модерации — внешний OpenAI Moderation API. По умолчанию
    # ВЫКЛЮЧЕН: требует выхода в интернет (VPN или PROXY_URL), рабочего ключа
    # OpenAI и отправляет текст пользователя во внешний сервис — для внутренней
    # базы знаний это осознанный выбор, а не техническая необходимость.
    # Локальный regex-блоклист работает всегда и от флага не зависит.
    moderation_use_openai: bool = False
    # Rate limit: сколько сообщений на одного owner_external_id в минуту.
    rate_limit_messages_per_min: int = 15

    # Qdrant ---------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    embedding_dim: int = 1024   # intfloat/multilingual-e5-large

    # RAG (LlamaIndex) ------------------------------------------------------
    # Корпус для индексации (data/<корпус>/<category>/...) и отдельные коллекции
    # под LlamaIndex (IngestionPipeline + запросы) и bare-metal сравнение.
    # По умолчанию — демонстрационный корпус из репозитория (синтетические
    # документы, см. scripts/generate_demo_corpus.py): на чистом клоне база
    # знаний наполняется без доступа к реальным документам. Для рабочего корпуса
    # задайте RAG_DATA_DIR=data/kb и отдельную коллекцию (RAG_COLLECTION).
    rag_data_dir: Path = Path("data/demo_kb")
    rag_collection: str = "rag_demo"
    rag_collection_bare: str = "rag_block_03_bare"
    # Дочерний docstore на диске — состояние инкрементальной индексации (UPSERTS).
    rag_docstore_path: Path = Path("var/rag_docstore.json")
    # LLM для генерации RAG-ответа (Ollama через OpenAI-совместимый эндпоинт).
    rag_llm_provider: Literal["ollama", "deepseek", "openai"] = "ollama"
    # Финальный выбор по итогам Б5.6: qwen3:8b (faithfulness 0.775 против 0.666 у
    # gemma3:4b). 8B на CPU медленнее — см. rag_llm_timeout.
    rag_llm_model: str = "qwen3:8b"
    # Таймаут генерации RAG-ответа (сек). qwen3:8b на CPU медленная: ~2–4 мин на
    # ответ, редкие вопросы до 5+ мин. 600 с покрывает с запасом.
    rag_llm_timeout: float = 600.0
    # Контекстное окно LLM (токенов), заявляемое в LLMMetadata. qwen3:8b поддерживает
    # больше, но для RAG-контекста (top-5 чанков) 8192 достаточно.
    rag_llm_context_window: int = 8192
    # Ширина retrieval (similarity_top_k): достаём широко, реранкер/обрезка
    # оставляет rag_rerank_top_n лучших. Из ДЗ 5.4: top-K=10 даёт полный recall.
    rag_top_k: int = 10
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    # Движок извлечения текста из PDF в офлайн-контуре (парсинг корпуса).
    # 'inspector' — pdf-inspector (Rust): классификация text_based/scanned
    # постранично + markdown-извлечение (в замерах ×35 быстрее pymupdf4llm и
    # аккуратнее с таблицами); 'pymupdf' — PyMuPDFReader (legacy-откат).
    # Сканы не читает ни один из них: страницы без текстового слоя пропускаются
    # (нужен внешний OCR).
    rag_pdf_parser: Literal["inspector", "pymupdf"] = "inspector"
    # Если top-1 score ниже порога — ответа в корпусе нет, отдаём честный отказ
    # БЕЗ вызова LLM (score-guard). Двухслойная защита: код + промпт. Порог
    # калибруется под embed-модель: для E5 косинус сжат (релевантное ~0.84,
    # off-topic ~0.76), поэтому 0.80 — зазор между ними (см. docs/rag.md).
    rag_score_threshold: float = 0.80

    # Re-ranker (ДЗ 5.4): cross-encoder поверх bi-encoder поиска. Включается
    # опционально (RAG_RERANK_ENABLED) — при включении пересортировывает
    # top-K кандидатов (rag_top_k) и оставляет top-N в промпт.
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_top_n: int = 5

    # Диалоговый RAG: включать retrieval + цитаты в /chats/{id}/messages при
    # доступном RAG-сервисе. False — /chats остаётся чистым LLM-чатом (M4).
    rag_enable_chat: bool = True
    # Condense: переписывать follow-up в самодостаточный поисковый запрос
    # (один LLM-вызов) при наличии истории — чинит поиск на коротких follow-up
    # вроде «а для них?». Переписанный запрос идёт только в retrieval.
    rag_condense_enabled: bool = True

    # Phoenix-трейсинг (LlamaIndex) -------------------------------------------
    # Инструментирование LlamaIndex в Phoenix — опциональный runtime-путь, группа
    # зависимостей `tracing` (uv sync --extra tracing). По умолчанию выключено;
    # при включении нужен поднятый сервис phoenix (compose.yaml / :6006).
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    # Пути, исключённые из трейсинга (через запятую). Healthcheck'и стучатся в
    # /health и /ready каждые 30 с — без исключения они забивают список трейсов
    # и вытесняют из него осмысленные запросы.
    phoenix_excluded_urls: str = "/health,/ready,/openapi.json,/docs,/redoc"

    # Оценка качества (RAGAS) -------------------------------------------------
    # Судья и эмбеддинги для офлайн-оценки (scripts/run_eval.py,
    # generate_testset.py) — группа зависимостей `eval`. Судья отделён от
    # production-LLM в /rag/query (rag_llm_model): роли разные, путать нельзя.
    anthropic_api_key: SecretStr | None = None
    eval_judge_provider: Literal["anthropic", "openai", "deepseek"] = "deepseek"
    eval_judge_model: str = "deepseek-v4-flash"
    
    agent_checkpointer: Literal["memory", "sqlite", "postgres"] = "sqlite"
    # Bootstrap-список разрешённых пользователей бота (chat_id через запятую).
    # Основной список живёт в БД (таблица bot_users, админ-API
    # /chats/admin/bot-users): в нём же имена для «отправь Иванову». Эта переменная
    # нужна для чистого клона (таблица пуста → бот закрыт для всех, и первый
    # администратор должен откуда-то взяться) и как аварийный доступ.
    # Пусто + пустая таблица = бот не отвечает никому.
    bot_allowed_chat_ids: str = ""
    # URI для чекпоинтера (psycopg v3, НЕ asyncpg):
    agent_checkpointer_postgres_uri: str = "postgresql://chat:pswd@localhost:5432/intellibase"
    agent_sqlite_path: str = "var/agent_checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
