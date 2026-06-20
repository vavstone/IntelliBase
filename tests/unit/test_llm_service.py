import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from app.services.llm import LLMService
from app.schemas.chat import ChatRequest, ChatResponse, Usage
from app.core.exceptions import LLMRateLimitError, LLMError


@pytest.fixture
def mock_openai():
    """Фикстура для замоканного OpenAI клиента."""
    mock = AsyncMock()
    mock.chat.completions.create = AsyncMock()
    return mock


@pytest.fixture
def mock_cache():
    """Фикстура для замоканного Redis клиента."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def llm_service(mock_openai, mock_cache):
    """Сервис с замоканными зависимостями."""
    return LLMService(llm=mock_openai, cache=mock_cache, ttl=3600)


def test_llm_service_key_excludes_fields():
    """Проверяем, что ключ кеша не зависит от user_id, session_id и stream."""
    service = LLMService(llm=None, cache=None, ttl=3600)
    base_req = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )
    key1 = service._key(base_req)

    # Меняем user_id
    req2 = base_req.model_copy(update={"user_id": "u123"})
    key2 = service._key(req2)
    assert key1 == key2

    # Меняем session_id
    req3 = base_req.model_copy(update={"session_id": "s456"})
    key3 = service._key(req3)
    assert key1 == key3

    # Меняем stream
    req4 = base_req.model_copy(update={"stream": True})
    key4 = service._key(req4)
    assert key1 == key4

    # Меняем содержимое сообщения
    req5 = base_req.model_copy()
    req5.messages[0].content = "Goodbye"
    key5 = service._key(req5)
    assert key1 != key5


def test_llm_service_extract_prompt():
    service = LLMService(llm=None, cache=None, ttl=3600)
    req = ChatRequest(
        messages=[
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User1"},
            {"role": "assistant", "content": "Assist"},
            {"role": "user", "content": "User2"},
        ],
        model="gpt-4o-mini"
    )
    prompt = service._extract_prompt(req)
    assert prompt == "User2"

    # Если нет user-сообщений – возвращаем пустую строку
    req_no_user = ChatRequest(
        messages=[{"role": "system", "content": "System"}],
        model="gpt-4o-mini"
    )
    assert service._extract_prompt(req_no_user) == ""


@pytest.mark.asyncio
async def test_llm_complete_cache_hit(mocker, llm_service, mock_openai, mock_cache):
    """При наличии данных в кеше ответ берётся из кеша, OpenAI не вызывается."""
    # Подготавливаем закешированный ответ, используя реальный объект Usage
    cached_response = ChatResponse(
        content="Cached reply",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        finish_reason="stop"
    )
    mock_cache.get.return_value = cached_response.model_dump_json()

    req = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.0  # кешируем только при temperature=0
    )

    response = await llm_service.complete(req)

    # Проверяем, что вернулся закешированный ответ
    assert response.content == "Cached reply"
    assert response.cached is True
    # OpenAI не вызывался
    mock_openai.chat.completions.create.assert_not_called()
    # setex не вызывался (обновлять кеш не нужно)
    mock_cache.setex.assert_not_called()


@pytest.mark.asyncio
async def test_llm_complete_cache_miss(mocker, llm_service, mock_openai, mock_cache):
    """При отсутствии в кеше вызывается OpenAI и результат сохраняется."""
    # Мокаем ответ OpenAI, но используем реальный объект Usage
    mock_raw = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "Fresh reply"
    mock_choice.finish_reason = "stop"
    mock_raw.choices = [mock_choice]
    mock_raw.model = "gpt-4o-mini"
    # Создаём объект Usage для мока
    mock_usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    mock_raw.usage = mock_usage
    mock_openai.chat.completions.create.return_value = mock_raw

    req = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.0
    )

    response = await llm_service.complete(req)

    # Проверяем, что ответ свежий и не кешированный
    assert response.content == "Fresh reply"
    assert response.cached is False
    # OpenAI вызван один раз
    mock_openai.chat.completions.create.assert_called_once()
    # Данные сохранены в кеш
    mock_cache.setex.assert_called_once()
    # Проверяем, что в setex передан правильный ключ и TTL
    key = llm_service._key(req)
    mock_cache.setex.assert_called_with(key, 3600, response.model_dump_json())


@pytest.mark.asyncio
async def test_llm_complete_retry_on_rate_limit(mocker, llm_service, mock_openai, mock_cache):
    """
    При получении RateLimitError tenacity выполняет повторные попытки.
    Проверяем, что после нескольких ошибок вызов завершается успешно.
    """
    # Создаём мок, который дважды выбрасывает RateLimitError, затем возвращает успех
    mock_raw = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "Success after retry"
    mock_choice.finish_reason = "stop"
    mock_raw.choices = [mock_choice]
    mock_raw.model = "gpt-4o-mini"
    mock_usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    mock_raw.usage = mock_usage

    # Импортируем исключение из openai и создаём его правильно
    from openai import RateLimitError

    # Создаём объект исключения с нужными атрибутами
    # В новых версиях openai нужно передать response и body
    # Создаём мок-объекты для response и body
    mock_response = mocker.MagicMock()
    mock_response.status_code = 429
    mock_body = {"error": {"message": "rate limit"}}

    # Создаём исключение с правильными аргументами
    error1 = RateLimitError("rate limit", response=mock_response, body=mock_body)
    error2 = RateLimitError("rate limit again", response=mock_response, body=mock_body)

    # Настраиваем side_effect: два исключения, затем успех
    mock_openai.chat.completions.create.side_effect = [
        error1,
        error2,
        mock_raw
    ]

    req = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.0
    )

    # Должно отработать без исключений
    response = await llm_service.complete(req)

    # Проверяем, что create вызван 3 раза
    assert mock_openai.chat.completions.create.call_count == 3
    # Ответ должен быть успешным
    assert response.content == "Success after retry"
    # Кеш должен быть сохранён (после успешного вызова)
    mock_cache.setex.assert_called_once()


@pytest.mark.asyncio
async def test_llm_complete_retry_exhausted(mocker, llm_service, mock_openai, mock_cache):
    """
    Если все попытки исчерпаны, исключение пробрасывается как LLMRateLimitError.
    """
    from openai import RateLimitError

    # Создаём мок-объекты для response и body
    mock_response = mocker.MagicMock()
    mock_response.status_code = 429
    mock_body = {"error": {"message": "always rate limited"}}

    # Создаём исключение с правильными аргументами
    error = RateLimitError("always rate limited", response=mock_response, body=mock_body)

    mock_openai.chat.completions.create.side_effect = error

    req = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
        temperature=0.0
    )

    with pytest.raises(LLMRateLimitError):
        await llm_service.complete(req)

    # Проверяем, что create вызывался 3 раза (по умолчанию max_retries=3 в декораторе)
    # В коде LLMService._call украшен @retry(stop=stop_after_attempt(3))
    assert mock_openai.chat.completions.create.call_count == 3
    # Кеш не сохранялся
    mock_cache.setex.assert_not_called()