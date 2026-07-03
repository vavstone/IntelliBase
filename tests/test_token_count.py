import pytest
from typing import List, Dict

from app.services.token_counter import count_tokens
from app.core.config import get_settings
from app.services.llm import LLMService

pytestmark = pytest.mark.integration

@pytest.fixture
def api_key() -> str:
    settings = get_settings()
    key = settings.llm.openai_api_key.get_secret_value()
    if not key or key == "sk-test-placeholder":
        pytest.skip("OpenAI API key not set or is placeholder")
    return key


@pytest.fixture
def llm_service(api_key) -> LLMService:
    """Создаёт LLMService с реальным OpenAI клиентом (синхронная фикстура)."""
    settings = get_settings()
    from openai import AsyncOpenAI
    import httpx

    http_client = httpx.AsyncClient(
        proxy=settings.proxy_url,
        timeout=httpx.Timeout(settings.llm.request_timeout, connect=5.0),
    )
    llm_openai = AsyncOpenAI(
        base_url=settings.llm.openai_base_url,
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_client,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    # Остальные провайдеры не нужны для теста
    llm_ollama = None
    llm_openrouter = None
    cache = None
    return LLMService(
        llm_ollama=llm_ollama,
        llm_openai=llm_openai,
        llm_openrouter=llm_openrouter,
        cache=cache,
        ttl=settings.cache_ttl_seconds,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1-nano"])
async def test_count_tokens_accuracy(llm_service: LLMService, model: str):
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what about Germany?"},
    ]

    our_count = count_tokens(messages, model=model)

    client = llm_service.get_llm("openai")
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
            stream=False,
        )
    except Exception as e:
        pytest.skip(f"API call failed: {e}")

    usage = response.usage
    if usage is None:
        pytest.skip("Usage not returned by API")
    real_count = usage.prompt_tokens

    diff_percent = abs(our_count - real_count) / real_count * 100
    assert diff_percent <= 10.0, (
        f"Token count mismatch: our={our_count}, real={real_count}, diff={diff_percent:.2f}%"
    )


@pytest.mark.asyncio
async def test_single_message_count(llm_service: LLMService):
    model = "gpt-4o-mini"
    messages = [{"role": "user", "content": "Hello, world!"}]
    our_count = count_tokens(messages, model=model)

    client = llm_service.get_llm("openai")
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1,
        stream=False,
    )
    real_count = response.usage.prompt_tokens if response.usage else 0
    if real_count == 0:
        pytest.skip("Usage not returned")
    diff_percent = abs(our_count - real_count) / real_count * 100
    assert diff_percent <= 10.0