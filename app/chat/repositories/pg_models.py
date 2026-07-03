"""ORM-модели для Postgres-хранения чата.

Лежат рядом с pg_repo.py и не пробрасываются наружу модуля app.chat.
Граница с доменными моделями — `ChatMessage.model_validate(row, from_attributes=True)`.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4
from sqlalchemy import DateTime, ForeignKey
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
    tokens: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )