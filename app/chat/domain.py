from datetime import datetime, UTC
from typing import Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class Chat(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    provider: Literal["openai", "ollama", "openrouter"]
    model: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    tokens: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))