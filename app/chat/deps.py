from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_repo import (
    PostgresChatRepository,
)
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import get_settings
from app.deps.providers import SessionFactoryDep, LLMOllamaDep, LLMOpenaiDep, LLMOpenrouterDep


async def get_repository(
    session_factory: SessionFactoryDep,
) -> AsyncIterator[ChatRepository]:
    """Фабрика репозитория. Postgres-сессия живёт ровно один запрос.
    Реализована как yield-dependency, чтобы session корректно закрывалась.
    """
    settings = get_settings()
    if settings.chat_repository == "json":
        yield JsonChatRepository(settings.chat_storage_dir)
        return

    if settings.chat_repository == "postgres":
        if session_factory is None:
            raise RuntimeError(
                "session_factory not initialised — postgres repository unavailable"
            )
        async with session_factory() as session:
            yield PostgresChatRepository(session)
        return

    raise ValueError(f"unknown chat_repository: {settings.chat_repository}")


RepositoryDep = Annotated[ChatRepository, Depends(get_repository)]


def get_chat_service(
    repo: RepositoryDep,
    llm_ollama: LLMOllamaDep,
    llm_openai: LLMOpenaiDep,
    llm_openrouter: LLMOpenrouterDep,
) -> ChatService:

    settings = get_settings()
    return ChatService(
        repository=repo,
        llm_ollama=llm_ollama,
        llm_openai=llm_openai,
        llm_openrouter=llm_openrouter,
        context_window=settings.chat_context_window,
        default_provider=settings.llm.default_provider,
        default_model=settings.llm.default_model,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]