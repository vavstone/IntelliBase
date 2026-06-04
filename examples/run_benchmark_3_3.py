import asyncio

from app.logger_setup import setup_logging
from app.scripts import benchmark

async def main() -> None:
    setup_logging()
    await benchmark.test_concurrency(requests_count=5,test_sync_batch_mode=True,test_async_batch_mode=True,test_async_stream_mode=True)


if __name__ == "__main__":
    asyncio.run(main())