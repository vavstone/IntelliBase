"""ChatService — оркестратор: history -> context strategy -> LLM -> save.

`send_message` принимает опциональное `media: UploadFile`; опциональные
`moderation` (каскад regex+OpenAI) и `prompt_repo` (A/B traffic split
системных промптов) подключаются через DI.

Важно: модерация НЕ вызывается внутри `send_message` — она бы запустилась
уже из stream-генератора, и `raise HTTPException(403)` не вернул бы
корректный HTTP-статус (response к этому моменту уже летит как 200
text/event-stream). Поэтому в route handler вызывается отдельно
`await service.check_input(...)`, и только после прохода —
`StreamingResponse(service.send_message(...))`.

Диалоговый RAG (Б5.5): если в сервис передан `rag_service` и включён
`rag_enable_chat`, `send_message` идёт по RAG-пути — retrieval (с опциональным
condense по истории) + генерация по пронумерованному контексту с цитатами [1][2]
и финальным SSE-событием `sources`. Иначе — прежний чистый LLM-чат (M4).
"""

import logging
from collections.abc import AsyncIterator
from typing import Literal
from uuid import UUID

from fastapi import UploadFile

from app.chat.domain import Chat, ChatMessage
from app.chat.media import media_to_part
from app.chat.prompt_selection import choose_by_split
from app.chat.repository import ChatRepository, SystemPromptRepository
from app.moderation.domain import ModerationResult
from app.moderation.service import ModerationService

logger = logging.getLogger("llm-service.chat")

CONDENSE_TEMPLATE = (
    "Перепиши последний вопрос в самодостаточный поисковый запрос, учитывая "
    "историю диалога (замени местоимения и опущенные слова на явные сущности). "
    "Верни ТОЛЬКО переписанный вопрос, без пояснений.\n\n"
    "История:\n{history}\n\n"
    "Последний вопрос: {question}\n"
    "Переписанный вопрос:"
)


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_ollama,
        llm_openai,
        llm_openrouter,
        context_window: int = 10,
        default_provider: str = "ollama",
        default_model: str = "qwen2.5:3b",
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
        moderation: ModerationService | None = None,
        prompt_repo: SystemPromptRepository | None = None,
        rag_service=None,
        rag_enable_chat: bool = False,
        rag_condense_enabled: bool = False,
        rag_score_threshold: float = 0.5,
    ):
        self.repository = repository
        self.llm_ollama = llm_ollama
        self.llm_openai = llm_openai
        self.llm_openrouter = llm_openrouter
        self.context_window = context_window
        self.default_provider = default_provider
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.moderation = moderation
        self.prompt_repo = prompt_repo
        # Диалоговый RAG (Б5.5) — опционален; без rag_service остаётся M4-чат.
        self.rag_service = rag_service
        self.rag_enable_chat = rag_enable_chat
        self.rag_condense_enabled = rag_condense_enabled
        self.rag_score_threshold = rag_score_threshold

    def get_llm(self, provider: str):
        if provider == "openai":
            return self.llm_openai
        elif provider == "openrouter":
            return  self.llm_openrouter
        return  self.llm_ollama

    def _rag_active(self) -> bool:
        return self.rag_service is not None and self.rag_enable_chat

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
        """Идемпотентная фабрика чата. system_prompt применяется только при
        создании; если чат уже существует — параметр игнорируется."""
        return await self.repository.get_or_create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
        )


    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def list_messages(
            self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        return await self.repository.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)

    async def check_input(
        self, content: str, owner_external_id: str | None = None
    ) -> ModerationResult:
        """Проверка модерации. Вызывается ДО StreamingResponse в route.

        Возвращает ModerationResult; route смотрит на `.allowed` и при
        блокировке возвращает 403 ДО старта streaming. Если moderation
        не сконфигурирована — пропускает (allowed=True).

        `owner_external_id` нужен только для контекста в alert payload —
        чтобы админ в чате видел, кому именно мы заблокировали запрос.
        """
        if self.moderation is None:
            return ModerationResult(allowed=True, layer="passed")
        return await self.moderation.check_input(
            content, owner_external_id=owner_external_id
        )

    async def check_output(
        self, content: str, owner_external_id: str | None = None
    ) -> ModerationResult:
        """Модерация ответа LLM. Вызывается после накопления полного ответа.

        Если ответ заблокирован — вместо него пользователю показывается
        сообщение-заглушка через SSE-событие moderation_notice.
        """
        if self.moderation is None:
            return ModerationResult(allowed=True, layer="passed")
        return await self.moderation.check_output(
            content, owner_external_id=owner_external_id
        )

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
        history: list[ChatMessage],
        system_prompt_body: str | None = None,
    ) -> list[dict]:
        """Sliding window: system_prompt + последние N сообщений.

        system_prompt берётся в порядке приоритета:
        1) явный параметр (A/B выбранный вариант), либо
        2) chat.system_prompt (исторический per-chat), либо
        3) нет system-сообщения вовсе.
        """
        messages: list[dict] = []
        effective_prompt = system_prompt_body or chat.system_prompt
        if effective_prompt:
            messages.append({"role": "system", "content": effective_prompt})
        for m in history:
            messages.append(
                {"role": m.role, "content": self._message_content_for_llm(m)}
            )
        return messages

    async def _pick_prompt(self, owner_external_id: str):
        """A/B traffic-split: возвращает (prompt_id, body) или (None, None)."""
        if self.prompt_repo is None:
            return None, None
        active = await self.prompt_repo.list_active()
        chosen = choose_by_split(owner_external_id, active)
        if chosen is None:
            return None, None
        return chosen.id, chosen.body

    async def _condense(
        self, chat: Chat, user_content: str, history: list[ChatMessage], llm
    ) -> str:
        """Переписывает follow-up в самодостаточный поисковый запрос (1 LLM-вызов).

        Нужен только для retrieval: вектор-поиск видит одну строку и на коротких
        follow-up'ах («а для них?») возвращает мусор. В генерацию идёт полная
        история, поэтому ответ модель всё равно поймёт — condense чинит поиск.
        """
        prior = [m for m in history[:-1] if m.content.strip()]
        if not prior:
            return user_content
        history_str = "\n".join(f"{m.role}: {m.content}" for m in prior[-6:])
        prompt = CONDENSE_TEMPLATE.format(history=history_str, question=user_content)
        resp = await llm.chat.completions.create(
            model=chat.model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or user_content

    def _build_rag_messages(
        self,
        history: list[ChatMessage],
        user_content: str,
        context_str: str,
    ) -> list[dict]:
        """Сообщения для RAG-генерации: предыстория + последний вопрос с контекстом.

        Полная предыстория уходит в LLM целиком — поэтому «а для них?» модель
        понимает из контекста; пронумерованный контекст — в финальном user-сообщении.
        """
        from app.services.rag import CITATION_QA_TEMPLATE

        messages: list[dict] = [
            {"role": "system", "content": "Ты корпоративный ассистент базы знаний."}
        ]
        for m in history[:-1]:
            messages.append({"role": m.role, "content": self._message_content_for_llm(m)})
        messages.append(
            {
                "role": "user",
                "content": CITATION_QA_TEMPLATE.format(
                    context_str=context_str, query_str=user_content
                ),
            }
        )
        return messages

    async def _send_rag(
        self, chat: Chat, user_content: str, llm, prompt_id
    ) -> AsyncIterator[dict]:
        """RAG-путь: condense -> retrieve -> score-guard -> генерация с цитатами."""
        from app.services.rag import (
            REFUSAL_TEXT,
            build_sources,
            numbered_context,
        )

        history = await self.repository.list_messages(chat.id, limit=self.context_window)

        # 1. Поисковый запрос: опционально переписываем follow-up (condense).
        search_query = user_content
        if self.rag_condense_enabled and len(history) > 1:
            try:
                search_query = await self._condense(chat, user_content, history, llm)
            except Exception:
                logger.warning("condense упал, использую сырой вопрос", exc_info=True)

        # 2. Retrieval + score-guard.
        nodes = await self.rag_service.retrieve(search_query)
        top_score = max((n.score or 0.0 for n in nodes), default=0.0)
        sources = build_sources(nodes)
        if not nodes or top_score < self.rag_score_threshold:
            logger.info(
                "RAG score-guard: отказ chat_id=%s top_score=%.3f", chat.id, top_score
            )
            saved = await self.repository.append_message(
                chat.id,
                ChatMessage(
                    chat_id=chat.id,
                    role="assistant",
                    content=REFUSAL_TEXT,
                    sources=[],
                    prompt_id=prompt_id,
                ),
            )
            yield {"type": "token", "delta": REFUSAL_TEXT}
            yield {"type": "sources", "sources": []}
            yield {"type": "message_saved", "message_id": str(saved.id)}
            return

        # 3. Генерация по пронумерованному контексту.
        messages = self._build_rag_messages(
            history, user_content, numbered_context(nodes)
        )
        buffer = ""
        usage = None
        extra = {}
        if chat.provider in ("openai", "openrouter"):
            extra["stream_options"] = {"include_usage": True}

        try:
            stream = await llm.chat.completions.create(
                model=chat.model or self.default_model,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
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
                "RAG stream interrupted chat_id=%s err=%s saved_chars=%d",
                chat.id, exc, len(buffer),
            )
            if buffer:
                await self.repository.append_message(
                    chat.id,
                    ChatMessage(
                        chat_id=chat.id,
                        role="assistant",
                        content=buffer,
                        tokens=usage.total_tokens if usage else None,
                        sources=sources,
                        prompt_id=prompt_id,
                    ),
                )
            yield {"type": "token", "delta": f"\n\n[Ошибка: {exc}]"}
            return

        if buffer:
            mod_result = await self.check_output(
                buffer, owner_external_id=chat.owner_external_id
            )
            if not mod_result.allowed:
                buffer = (
                    "Извините, ответ не может быть показан — "
                    "он нарушает правила сервиса. "
                    "Попробуйте переформулировать вопрос."
                )
                sources = []
                yield {
                    "type": "moderation_notice",
                    "categories": mod_result.categories,
                    "reasons": mod_result.reasons,
                }
            saved = await self.repository.append_message(
                chat.id,
                ChatMessage(
                    chat_id=chat.id,
                    role="assistant",
                    content=buffer,
                    tokens=usage.total_tokens if usage else None,
                    sources=sources,
                    prompt_id=prompt_id,
                ),
            )
            yield {"type": "sources", "sources": sources}
            yield {"type": "message_saved", "message_id": str(saved.id)}

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media: UploadFile | None = None,
    ) -> AsyncIterator[dict]:
        """Полный цикл обработки сообщения пользователя.

        Yields структурированные события:
        - `{"type":"token","delta":"<chunk>"}` — каждый LLM-фрагмент.
        - `{"type":"sources","sources":[...]}` — показанные источники (только RAG).
        - `{"type":"message_saved","message_id":"<uuid>"}` — ОДИН раз после
          успешного сохранения assistant-сообщения.

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

        # 3. Выбираем prompt (A/B). Возвращает (id, body) или (None, None).
        prompt_id, prompt_body = await self._pick_prompt(
            chat.owner_external_id
        )

        # 4. Сохраняем user-сообщение
        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs=media_refs,
            prompt_id=prompt_id,
        )
        await self.repository.append_message(chat_id, user_message)

        # 4.1 Диалоговый RAG: retrieval + цитаты вместо чистого LLM-чата.
        if self._rag_active():
            async for event in self._send_rag(chat, user_content, llm, prompt_id):
                yield event
            return

        # 5. История + контекст
        history = await self.repository.list_messages(
            chat_id, limit=self.context_window
        )
        messages = self._build_context(chat, history, prompt_body)

        # 6. Стримим
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
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
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

        # 7. Успешное завершение — модерация ответа и сохранение
        if buffer:
            # Модерация ответа LLM (check_output).
            # Если ответ заблокирован — заменяем на сообщение-заглушку
            # и шлём moderation_notice в SSE-стрим.
            mod_result = await self.check_output(
                buffer, owner_external_id=chat.owner_external_id
            )
            if not mod_result.allowed:
                logger.info(
                    "output moderation blocked chat_id=%s categories=%s",
                    chat_id,
                    mod_result.categories,
                )
                buffer = (
                    "Извините, ответ не может быть показан — "
                    "он нарушает правила сервиса. "
                    "Попробуйте переформулировать вопрос."
                )
                yield {
                    "type": "moderation_notice",
                    "categories": mod_result.categories,
                    "reasons": mod_result.reasons,
                }

            saved = await self.repository.append_message(
                chat_id,
                ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=buffer,
                    tokens=usage.total_tokens if usage else None,
                    prompt_id=prompt_id,
                ),
            )
            yield {
                "type": "message_saved",
                "message_id": str(saved.id),
            }
