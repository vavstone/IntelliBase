"""ORM-модели для Postgres-хранения чата.

Лежат рядом с pg_repo.py и не пробрасываются наружу модуля app.chat.
Граница с доменными моделями — `ChatMessage.model_validate(row, from_attributes=True)`.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

TimestampTZ = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_external_id: Mapped[str]
    interface: Mapped[str]
    provider: Mapped[str]
    model: Mapped[str]
    system_prompt: Mapped[str | None]
    handoff_status: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE")
    )
    role: Mapped[str]
    content: Mapped[str]
    media_refs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None]
    prompt_id: Mapped[UUID | None]
    # Показанные источники RAG-ответа (id/file_name/page/score/snippet) — кладутся
    # рядом с assistant-сообщением, чтобы фидбек связывался с источниками по message_id.
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )


class SystemPromptRow(Base):
    __tablename__ = "system_prompts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version: Mapped[str]
    body: Mapped[str]
    active: Mapped[bool] = mapped_column(default=False)
    traffic_pct: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class RequestMetricRow(Base):
    """Метрики запросов для admin-статистики (avg_latency_ms, moderation_block_rate).

    Заполняется observability-middleware (app/main.py) на каждом запросе.
    Схема минимальна — только поля, нужные для агрегации.
    """

    __tablename__ = "request_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(default="")
    status_code: Mapped[int] = mapped_column(default=0)
    duration_ms: Mapped[float] = mapped_column(default=0.0)
    detail_code: Mapped[str | None] = mapped_column(nullable=True)
    owner_external_id: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class BroadcastRow(Base):
    """Очередь массовых рассылок.

    POST /chats/admin/broadcast создаёт запись pending.
    Фоновая таска (broadcast_worker) разгребает очередь.
    """

    __tablename__ = "broadcasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str]
    interface_filter: Mapped[str] = mapped_column(default="telegram")
    # pending → sending → done | partial_fail | failed
    status: Mapped[str] = mapped_column(default="pending")
    sent: Mapped[int] = mapped_column(default=0)
    failed: Mapped[int] = mapped_column(default=0)
    total: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )


class MessageFeedbackRow(Base):
    """Оценки ответов ассистента (feedback up/down)."""

    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint("owner_external_id", "message_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    message_id: Mapped[UUID]
    owner_external_id: Mapped[str]
    value: Mapped[str]  # 'up' | 'down'
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class RateLimitRow(Base):
    """Счётчики rate-limit: minute-bucket'ы на owner/kind."""

    __tablename__ = "rate_limits"

    owner_external_id: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(primary_key=True)
    bucket: Mapped[str] = mapped_column(primary_key=True)
    count: Mapped[int] = mapped_column(default=0)


class AlertRow(Base):
    """Алерты — упрощённая БД-очередь для bot-drain."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kind: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
    acked_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )


class RagQueryRow(Base):
    """Лог RAG-запросов для аналитики (refusal_rate, пробелы в знаниях).

    Каждый ответ /rag/query (и диалогового RAG) пишет строку: нормализованный
    вопрос, флаг confident и top_score. Пробелы считаются group_by по
    question_normalized среди строк с confident=false.
    """

    __tablename__ = "rag_queries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_normalized: Mapped[str]
    confident: Mapped[bool] = mapped_column(default=False)
    top_score: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class KbCategoryRow(Base):
    """Категория знаний = одно ПС (подсистема ФТС).

    `slug` — канонический ключ везде (папка data/kb, метаданные Qdrant, фильтр
    RAG, callback бота); `title` — русское отображение для UI.
    """

    __tablename__ = "kb_categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(unique=True)
    title: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
