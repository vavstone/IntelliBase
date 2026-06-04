import logging
from app.services.llm_client import AsyncLLMClient, SyncLLMClient


async def test_concurrency(requests_count:int, test_sync_batch_mode:bool=True, test_async_batch_mode:bool=True, test_async_stream_mode:bool=True) -> None:
    logger = logging.getLogger(__name__)
    print('-'*30)
    print_str = 'Начало работы...'
    print(print_str)
    logger.info(print_str)

    test_prompt = 'Объясни одним абзацем концепцию concurrency в программировании'
    print_str = f'Тестовый промпт: "{test_prompt}"'
    print(print_str)
    logger.info(print_str)

    if test_sync_batch_mode:
        print_str = f'Запуск {requests_count} запросов в синхронном режиме...'
        print(print_str)
        logger.info(print_str)

        sync_client = SyncLLMClient(max_retries=1, response_timeout=30)

        res = sync_client.batch_chat([test_prompt] * requests_count)
        ok = [r for r in res if not isinstance(r, Exception)]
        error = [r for r in res if isinstance(r, Exception)]
        print_str = f'Успешно: {len(ok)}, ошибок: {len(error)}'
        print('Ответы:')
        for n, success_answer in enumerate(ok, start=1):
            print(f'{n}: {success_answer}')
        print(print_str)
        logger.info(print_str)

        print_str = 'Ответы в синхронном режиме получены.'
        print(print_str)
        logger.info(print_str)

    if test_async_batch_mode:
        print_str = f'Запуск {requests_count} запросов в асинхронном режиме...'
        print(print_str)
        logger.info(print_str)

        async_client = AsyncLLMClient(max_retries=1, concurrency=5, response_timeout=30)

        res = await async_client.batch_chat([test_prompt]*requests_count)
        ok = [r for r in res if not isinstance(r, Exception)]
        error = [r for r in res if isinstance(r, Exception)]
        print_str = f'Успешно: {len(ok)}, ошибок: {len(error)}'
        print('Ответы:')
        for n, success_answer in enumerate(ok,start=1):
            print(f'{n}: {success_answer}')
        print(print_str)
        logger.info(print_str)

        print_str = 'Ответы в асинхронном режиме получены.'
        print(print_str)
        logger.info(print_str)

    if test_async_stream_mode:
        print_str = 'Запуск stream запроса в асинхронном режиме...'
        print(print_str)
        logger.info(print_str)
        async_client2 = AsyncLLMClient(max_retries=1, concurrency=1, response_timeout=30)
        res = async_client2.stream_chat(test_prompt)
        async for chunk in res:
            print(chunk,end='')

        print_str = 'Ответы на stream запрос в асинхронном режиме получен.'
        print(print_str)
        logger.info(print_str)

    print_str = 'Завершение работы.'
    print(print_str)
    print('-' * 30)
    print('\n\n')
    logger.info(print_str)
