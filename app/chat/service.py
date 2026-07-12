import logging
from collections.abc import AsyncIterator
from typing import Literal
from uuid import UUID

from fastapi import UploadFile

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.chat.media import media_to_part

logger = logging.getLogger("llm-service.chat")


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_ollama,
        llm_openai,
        llm_openrouter,
        context_window: int = 10,
        default_provider: str = "ollama",
        default_model: str = "qwen2.5:3b"
    ):
        self.repository = repository
        self.llm_ollama = llm_ollama
        self.llm_openai = llm_openai
        self.llm_openrouter = llm_openrouter
        self.context_window = context_window
        self.default_provider = default_provider
        self.default_model = default_model

    def get_llm(self, provider: str):
        if provider == "openai":
            return self.llm_openai
        elif provider == "openrouter":
            return  self.llm_openrouter
        return  self.llm_ollama

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        provider: Literal["openai", "ollama", "openrouter"],
        model: str,
        system_prompt: str | None = None,
    ) -> Chat:
        return await self.repository.create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
        )

    async def get_or_create_chat(
            self,
            owner_external_id: str,
            interface: str,
            provider: Literal["openai", "ollama", "openrouter"],
            model: str,
            system_prompt: str | None = None,
    ) -> Chat:

        return await self.repository.get_or_create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            provider=provider,
            model=model,
            system_prompt=system_prompt
        )


    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def list_messages(
            self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        return await self.repository.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)
    @staticmethod
    def _message_content_for_llm(m: ChatMessage) -> str | list[dict]:
        """Возвращает content в формате OpenAI Chat Completions.

        Если у сообщения есть `media_refs.part`, content становится
        `[text-part, media-part]` (text-part только если content не пустой).
        Иначе — обычная строка.
        """
        media_part = None
        if m.media_refs and isinstance(m.media_refs, dict):
            media_part = m.media_refs.get("part")
        if media_part is None:
            return m.content
        parts: list[dict] = []
        if m.content:
            parts.append({"type": "text", "text": m.content})
        parts.append(media_part)
        return parts

    def _build_context(
            self,
            chat: Chat,
            history: list[ChatMessage]
    ) -> list[dict]:
        """Sliding window: system_prompt + последние N сообщений.
        system_prompt берётся в порядке приоритета:
        1) chat.system_prompt (исторический per-chat), либо
        2) нет system-сообщения вовсе.
        """
        messages: list[dict] = []
        effective_prompt = chat.system_prompt
        if effective_prompt:
            messages.append({"role": "system", "content": effective_prompt})
        for m in history:
            messages.append(
                {"role": m.role, "content": self._message_content_for_llm(m)}
            )
        return messages


    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media: UploadFile | None = None,
    ) -> AsyncIterator[dict]:
        """Полный цикл обработки сообщения пользователя.

        Yields структурированные события:
        - `{"type":"token","delta":"<chunk>"}` — каждый LLM-фрагмент.
        - `{"type":"message_saved","message_id":"<uuid>"}` — ОДИН раз после
          успешного сохранения assistant-сообщения. Нужно клиенту, чтобы
          навесить feedback-кнопки с реальным id ответа.

        Финальный `{"type":"done"}` добавляется в route handler, чтобы оба
        источника (упавший стрим vs нормальный) одинаково завершались.

        
        """

        # 1. Загружаем чат
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ValueError(f"chat {chat_id} not found")


        llm = self.get_llm(chat.provider)

        # 2. media → part
        media_refs: dict | None = None
        if media is not None:
            mime = media.content_type or ""
            filename = media.filename
            size = getattr(media, "size", None)
            part = await media_to_part(media, llm)
            media_refs = {
                "mime": mime,
                "size": size,
                "filename": filename,
                "part": part,
            }


        # 3. Сохраняем user-сообщение
        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs=media_refs
        )
        await self.repository.append_message(chat_id, user_message)

        # 4. История + контекст
        history = await self.repository.list_messages(
            chat_id, limit=self.context_window
        )
        messages = self._build_context(chat, history)

        # 5. Стримим
        buffer = ""
        usage = None

        # stream_options — только для OpenAI-совместимых провайдеров
        extra = {}
        if chat.provider in ("openai", "openrouter"):
            extra["stream_options"] = {"include_usage": True}

        try:
            stream = await llm.chat.completions.create(
                model=chat.model or self.default_model,
                messages=messages,
                stream=True,
                **extra,
            )
            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage

                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    buffer += content
                    yield {"type": "token", "delta": content}
        except Exception as exc:
            logger.warning(
                "stream interrupted chat_id=%s err=%s saved_chars=%d",
                chat_id,
                exc,
                len(buffer),
            )
            if buffer:
                await self.repository.append_message(
                    chat_id,
                    ChatMessage(
                        chat_id=chat_id,
                        role="assistant",
                        content=buffer,
                        tokens=usage.total_tokens if usage else None,
                    ),
                )
            # Отдаём ошибку как токен, чтобы бот показал пользователю
            yield {"type": "token", "delta": f"\n\n[Ошибка: {exc}]"}
            return

        # 6. Успешное завершение — сохраняем накопленный ответ
        if buffer:
            saved = await self.repository.append_message(
                chat_id,
                ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=buffer,
                    tokens=usage.total_tokens if usage else None,
                ),
            )
            yield {
                "type": "message_saved",
                "message_id": str(saved.id),
            }


