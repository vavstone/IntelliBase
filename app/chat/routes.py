import json
from typing import Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Query, UploadFile, File,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.core.config import get_settings


router = APIRouter(prefix="/chats", tags=["chats"])

_settings = get_settings()

class CreateChatIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "owner_external_id": "user-123",
                    "interface": "telegram",
                    "system_prompt": "Ты полезный ассистент.",
                    "force_new": False,
                }
            ]
        }
    )

    owner_external_id: str
    interface: str
    provider: Literal["openai", "ollama", "openrouter"] | None = None
    model: str | None = None
    system_prompt: str | None = None
    force_new: bool = False


class CreateChatOut(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"chat_id": "550e8400-e29b-41d4-a716-446655440000"}
            ]
        }
    )

    chat_id: UUID


def _resolve_defaults(body: CreateChatIn) -> tuple[str, str]:
    """Подставляет provider/model из настроек сервера, если клиент их не
    передал.  Это гарантирует, что .env :: LLM__DEFAULT_MODEL /
    LLM__DEFAULT_PROVIDER являются единственным источником истины."""
    provider = body.provider or _settings.llm.default_provider
    model = body.model or _settings.llm.default_model
    return provider, model


@router.post("", response_model=CreateChatOut, summary="Создать чат")
async def create_chat(
    body: CreateChatIn, chat_service: ChatServiceDep
) -> CreateChatOut:
    """По умолчанию идемпотентно по (owner_external_id, interface): повторный
    POST с теми же параметрами возвращает chat_id уже существующего чата.
    С `force_new=True` всегда создаёт новый чат (полезно для /start в боте).

    system_prompt применяется только при первом создании; если чат уже
    существует — игнорируется молча (override через PATCH вне scope).
    """
    provider, model = _resolve_defaults(body)
    if body.force_new:
        chat = await chat_service.create_chat(
            owner_external_id=body.owner_external_id,
            interface=body.interface,
            provider=provider,
            model=model,
            system_prompt=body.system_prompt,
        )
    else:
        chat = await chat_service.get_or_create_chat(
            owner_external_id=body.owner_external_id,
            interface=body.interface,
            provider=provider,
            model=model,
            system_prompt=body.system_prompt,
        )
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата")
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep) -> Chat:
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@router.post(
    "/{chat_id}/messages",
    summary="Послать сообщение (multipart, SSE streaming)"
)
async def post_message(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    content: str = Form(""),
	media: UploadFile | None = File(None),
    # owner_external_id: Annotated[str | None, Header(alias="X-Owner-External-Id")] = None
) -> StreamingResponse:

    async def event_source():
        try:
            async for event in chat_service.send_message(
                chat_id=chat_id, user_content=content, media=media,
            ):
                # service yield-ит уже структурированные dict-события:
                # {"type":"token","delta":"..."} и
                # {"type":"message_saved","message_id":"..."}.
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.get(
    "/{chat_id}/messages",
    response_model=list[ChatMessage],
    summary="История сообщений (хронологически)",
)
async def list_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    limit: int = Query(50, ge=1, le=500),
) -> list[ChatMessage]:
    return await chat_service.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages", summary="Очистить историю (soft delete)")
async def delete_messages(
    chat_id: UUID, chat_service: ChatServiceDep
) -> dict:
    await chat_service.clear_history(chat_id)
    return {"status": "ok"}
