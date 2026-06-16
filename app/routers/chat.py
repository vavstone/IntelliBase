import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post(
    "",
    response_model=ChatResponse,
    summary="Синхронный чат",
    description="Отправляет сообщения в LLM и возвращает полный ответ.",
    responses={
        200: {"description": "Успешный ответ"},
        400: {"description": "Некорректный запрос (контент заблокирован модерацией)"},
        401: {"description": "Ошибка аутентификации (невалидный API-ключ)"},
        403: {"description": "Доступ запрещён (страна или регион не поддерживается)"},
        422: {"description": "Невалидный запрос (ошибка валидации схемы)"},
        429: {"description": "Превышен лимит запросов к провайдеру"},
        502: {"description": "Внутренняя ошибка при обращении к LLM"},
        504: {"description": "Таймаут ожидания ответа от LLM"},
    },
)
async def chat_completions(req: ChatRequest, service: LLMServiceDep) -> ChatResponse:
    return await service.complete(req)


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="Стриминг чата",
    responses={
        200: {"description": "Потоковый ответ"},
        400: {"description": "Некорректный запрос (контент заблокирован модерацией)"},
        401: {"description": "Ошибка аутентификации (невалидный API-ключ)"},
        403: {"description": "Доступ запрещён (страна или регион не поддерживается)"},
        422: {"description": "Невалидный запрос (ошибка валидации схемы)"},
        429: {"description": "Превышен лимит запросов к провайдеру"},
        502: {"description": "Внутренняя ошибка при обращении к LLM"},
        504: {"description": "Таймаут ожидания ответа от LLM"},
    },
)
async def chat_completions_stream(req: ChatRequest, service: LLMServiceDep):
    # Пытаемся создать стрим – здесь может выброситься исключение
    # Если исключение выброшено, FastAPI вызовет @app.exception_handler(LLMError)
    stream = await service.create_stream(req)

    # Если всё ок – возвращаем StreamingResponse с генератором
    return StreamingResponse(
        service.iterate_stream(stream),
        media_type="text/event-stream"
    )