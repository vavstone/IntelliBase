from typing import Annotated, Any
from fastapi import Depends, Request

from app.core.config import Settings
from app.services.llm import LLMService
from app.core.config import get_settings


def get_ollama(request: Request):
    return request.app.state.llm_ollama

def get_openai(request: Request):
    return request.app.state.llm_openai

def get_openrouter(request: Request):
    return request.app.state.llm_openrouter

def get_cache(request: Request):
    return request.app.state.redis

SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMOllamaDep = Annotated[object, Depends(get_ollama)]
LLMOpenaiDep = Annotated[object, Depends(get_openai)]
LLMOpenrouterDep = Annotated[object, Depends(get_openrouter)]
CacheDep = Annotated[object, Depends(get_cache)]

def get_llm_service(
    llm_ollama: LLMOllamaDep,
    llm_openai: LLMOpenaiDep,
    llm_openrouter: LLMOpenrouterDep,
    cache: CacheDep,
    settings: SettingsDep,
) -> LLMService:
    return LLMService(
        llm_ollama=llm_ollama,
        llm_openai=llm_openai,
        llm_openrouter=llm_openrouter,
        cache=cache,
        ttl=settings.cache_ttl_seconds)

LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]

def get_session_factory(request: Request) -> Any:
    """Возвращает async_sessionmaker, выставленный в lifespan, либо None,
    если Postgres недоступен. Роуты, которым PG обязателен, должны явно
    проверять на None и отдавать 503/собственный fallback."""
    return request.app.state.session_factory

SessionFactoryDep = Annotated[Any, Depends(get_session_factory)]


def get_vector_store(request: Request):
    """Возвращает VectorStore из app.state (создан в lifespan)."""
    return getattr(request.app.state, "vector_store", None)

from app.services.vector_store import VectorStore as _VectorStore
VectorStoreDep = Annotated[_VectorStore | None, Depends(get_vector_store)]


def get_rag_service(request: Request):
    """RAG-сервис на LlamaIndex, собранный один раз в lifespan. None — если
    Qdrant/индекс был недоступен на старте: роут отдаёт 503."""
    return getattr(request.app.state, "rag_service", None)


RAGServiceDep = Annotated[Any, Depends(get_rag_service)]