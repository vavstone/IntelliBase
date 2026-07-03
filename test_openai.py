import asyncio
import httpx
from openai import AsyncOpenAI
from app.core.config import get_settings


async def test():
    settings = get_settings()

    # Создаём HTTP-клиент с прокси (если он задан)
    http_client = httpx.AsyncClient(
        proxy=settings.proxy_url,  # может быть None
        timeout=httpx.Timeout(settings.llm.request_timeout, connect=5.0),
    )

    client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_client,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        await http_client.aclose()


asyncio.run(test())