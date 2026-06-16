from typing import Annotated
from functools import lru_cache
from fastapi import Depends, Request

from app.core.config import Settings
from app.services.llm import LLMService

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_openai(request: Request):
    return request.app.state.llm

def get_cache(request: Request):
    return request.app.state.redis

SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[object, Depends(get_openai)]
CacheDep = Annotated[object, Depends(get_cache)]

def get_llm_service(
    llm: LLMDep,
    cache: CacheDep,
    settings: SettingsDep,
) -> LLMService:
    return LLMService(llm=llm, cache=cache, ttl=settings.cache_ttl_seconds)

LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]