from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from uuid import UUID

from app.chat.domain import ChatMessage, Chat
from app.chat.repositories.pg_models import ChatMessageRow, ChatRow


class PostgresChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_chat(
            self,
            owner_external_id: str,
            interface: str,
            provider: Literal["openai", "ollama", "openrouter"],
            model: str,
            system_prompt: str | None
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            provider=provider,
            model=model,
            system_prompt=system_prompt
        )
        row = ChatRow(
            id=chat.id,
            owner_external_id=chat.owner_external_id,
            interface=chat.interface,
            provider=chat.provider,
            model=chat.model,
            system_prompt=chat.system_prompt,
            created_at = chat.created_at
        )
        self.session.add(row)
        await self.session.commit()
        return chat

    async def get_chat(
            self,
            chat_id: UUID
    ) -> Chat | None:
        stmt = select(ChatRow).where(ChatRow.id == chat_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return Chat.model_validate(row, from_attributes=True)

    async def get_or_create_chat(
            self,
            owner_external_id: str,
            interface: str,
            provider: Literal["openai", "ollama", "openrouter"],
            model: str,
            system_prompt: str | None
    ) -> Chat:
        stmt = (
            select(ChatRow)
            .where(
                ChatRow.owner_external_id == owner_external_id,
                ChatRow.interface == interface,
            )
            .order_by(ChatRow.created_at.desc())
            .limit(1)
        )
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is not None:
            return Chat.model_validate(row, from_attributes=True)
        return await self.create_chat(
            owner_external_id=owner_external_id,
            interface = interface,
            provider = provider,
            model = model,
            system_prompt = system_prompt)

    async def append_message(
            self,
            chat_id: UUID,
            message: ChatMessage
    ) -> ChatMessage:
        row = ChatMessageRow(
            id=message.id,
            chat_id=chat_id,
            role=message.role,
            content=message.content,
            tokens = message.tokens
        )
        self.session.add(row)
        await self.session.commit()
        return ChatMessage.model_validate(row, from_attributes=True)

    async def list_messages(
            self,
            chat_id: UUID,
            limit: int = 50
    ) -> list[ChatMessage]:
        stmt = (
            select(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        # Переворачиваем, чтобы получить хронологический порядок (от старых к новым)
        return [
            ChatMessage.model_validate(row, from_attributes=True)
            for row in reversed(rows)
        ]

    async def soft_delete_messages(
            self,
            chat_id: UUID
    ) -> None:
        stmt = (
            update(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None)
            ).values(deleted_at=datetime.now())
        )
        await self.session.execute(stmt)
        await self.session.commit()