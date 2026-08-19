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
    default_provider: Literal["openai", "ollama", "openrouter"] = "ollama"
    default_model: str = "qwen2.5:3b"
    request_timeout: float = 30.0
    max_retries: int = 3


class EmbeddingSettings(BaseSettings):
    """Настройки embedding-сервиса. Переключаются через EMBEDDING_PROVIDER и EMBEDDING_MODEL."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["openai", "sentence_transformers"] = "sentence_transformers"
    model: str = "intfloat/multilingual-e5-large"
    batch_size: int = 32  # Для ST на CPU — 16–32
    cache_dir: str = "./var/embedding_cache"
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
    moderation_use_openai: bool = True
    # Rate limit: сколько сообщений на одного owner_external_id в минуту.
    rate_limit_messages_per_min: int = 15

    # Qdrant ---------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    embedding_dim: int = 1024   # intfloat/multilingual-e5-large

    # RAG (LlamaIndex) ------------------------------------------------------
    # Корпус для индексации (data/kb/<category>/...) и отдельные коллекции под
    # LlamaIndex (IngestionPipeline + запросы) и bare-metal сравнение.
    rag_data_dir: Path = Path("data/kb")
    rag_collection: str = "rag_block_05"
    rag_collection_bare: str = "rag_block_03_bare"
    # Дочерний docstore на диске — состояние инкрементальной индексации (UPSERTS).
    rag_docstore_path: Path = Path("var/rag_docstore.json")
    # LLM для генерации RAG-ответа (Ollama через OpenAI-совместимый эндпоинт).
    rag_llm_model: str = "gemma3:4b"
    # Таймаут генерации RAG-ответа (сек). gemma3:4b на CPU медленная: прогретый
    # ответ ~45–50 с, холодный старт (загрузка модели после простоя) добавляет
    # ещё ~30 с. Дефолт llama_index (60 с) не покрывает холодный старт.
    rag_llm_timeout: float = 120.0
    # Контекстное окно LLM (токенов), заявляемое в LLMMetadata. Совпадает с
    # gemma3:4b (8192); при смене RAG_LLM_MODEL — поменять (напр. qwen2.5:3b = 32768).
    rag_llm_context_window: int = 8192
    # Ширина retrieval (similarity_top_k): достаём широко, реранкер/обрезка
    # оставляет rag_rerank_top_n лучших. Из ДЗ 5.4: top-K=10 даёт полный recall.
    rag_top_k: int = 10
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
