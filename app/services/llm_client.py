import os
import time
import logging
from typing import AsyncIterator, Iterator
from openai import AsyncOpenAI, OpenAI
from asyncio import Semaphore, timeout, gather
from dotenv import load_dotenv
from app.prompts.loader import render_system_prompt

logger = logging.getLogger(__name__)

class AsyncLLMClient:
    def __init__(self, max_retries:int = 1, concurrency:int = 5, response_timeout:int = 30):
        load_dotenv()
        self._main_provider = os.getenv('MAIN_PROVIDER')
        self._main_model = os.getenv('MAIN_PROVIDER_MODEL')
        self._response_timeout = response_timeout
        self._max_retries = max_retries
        self._sem = Semaphore(concurrency)
        self._main_client = self._build_client(os.getenv('MAIN_PROVIDER_BASE_URL'), os.getenv('MAIN_PROVIDER_API_KEY'))
        self._messages_history = self._init_messages_history()

    def _build_client(self, base_url: str, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(timeout=self._response_timeout, max_retries=self._max_retries, base_url=base_url, api_key=api_key)

    @staticmethod
    def _init_messages_history() -> list[dict[str,str]]:
        return [{
            'role':'system',
            'content':render_system_prompt()
        }]

    def _build_messages(self, message_role, message_content) -> list[dict[str,str]]:
        self._messages_history.append({'role':message_role, 'content':message_content})
        return self._messages_history

    async def complete(self, message:str)-> str|None:
        messages = self._build_messages('user',message)
        start_time = time.perf_counter()
        try:
            async with timeout(self._response_timeout):
                response = await self._main_client.chat.completions.create(
                    model=self._main_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200
                )
                answer = response.choices[0].message.content
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "llm.call",
                    extra={
                        "duration_ms": f'{duration_ms:.1f}',
                        "provider": self._main_provider,
                        "model": self._main_model,
                        "tokens": response.usage.total_tokens,
                        "status": "ok",
                    }
                )
                self._build_messages('assistant',answer)
                return answer
        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.call",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "timeout_error",
                }
            )
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.call",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "other_exception",
                    "error": e
                }
            )
            raise

    async def batch_chat(self, prompts: list[str]) -> list[str | Exception]:
        start_time = time.perf_counter()
        async with self._sem:
            coros = [self.complete(prompt) for prompt in prompts]
            res = await gather(*coros, return_exceptions=True)
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.batch_chat",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "status": "ok",
                }
            )
            return res


    async def stream_chat(self, prompt: str) -> AsyncIterator[str]:
        messages = self._build_messages('user', prompt)
        start_time = time.perf_counter()
        answer:str = ''
        try:
            async with timeout(self._response_timeout):
                response = await self._main_client.chat.completions.create(
                    model=self._main_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200,
                    stream=True,
                    stream_options={"include_usage":True}
                )
                ttft_ms:float|None = None
                async for chunk in response:
                    if not ttft_ms:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                    if chunk.choices and chunk.choices[0].delta.content:
                        answer+=chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(
                            "llm.stream_chat",
                            extra={
                                "ttft_ms": f'{ttft_ms:.1f}',
                                "duration_ms": f'{duration_ms:.1f}',
                                "provider": self._main_provider,
                                "model": self._main_model,
                                "tokens": chunk.usage.total_tokens,
                                "status": "ok"
                            }
                        )
                        self._build_messages('assistant', answer)
        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.stream_chat",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "timeout_error",
                }
            )
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.stream_chat",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "other_exception",
                    "error": e
                }
            )
            raise


class SyncLLMClient:
    def __init__(self, max_retries:int = 1, response_timeout:int = 30):
        load_dotenv()
        self._main_provider = os.getenv('MAIN_PROVIDER')
        self._main_model = os.getenv('MAIN_PROVIDER_MODEL')
        self._response_timeout = response_timeout
        self._max_retries = max_retries
        self._main_client = self._build_client(os.getenv('MAIN_PROVIDER_BASE_URL'), os.getenv('MAIN_PROVIDER_API_KEY'))
        self._messages_history = self._init_messages_history()

    def _build_client(self, base_url: str, api_key: str) -> OpenAI:
        return OpenAI(timeout=self._response_timeout, max_retries=self._max_retries, base_url=base_url, api_key=api_key)

    @staticmethod
    def _init_messages_history() -> list[dict[str,str]]:
        return [{
            'role':'system',
            'content':render_system_prompt()
        }]

    def _build_messages(self, message_role, message_content) -> list[dict[str,str]]:
        self._messages_history.append({'role':message_role, 'content':message_content})
        return self._messages_history

    def complete(self, message:str)-> str|None:
        messages = self._build_messages('user',message)
        start_time = time.perf_counter()
        try:
            response = self._main_client.chat.completions.create(
                model=self._main_model,
                messages=messages,
                temperature=0.2,
                max_tokens=200
            )
            answer = response.choices[0].message.content
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "llm.call",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": response.usage.total_tokens,
                    "status": "ok",
                }
            )
            self._build_messages('assistant',answer)
            return answer
        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.call",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "timeout_error",
                }
            )
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.call",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "other_exception",
                    "error": e
                }
            )
            raise

    def batch_chat(self, prompts: list[str]) -> list[str | Exception]:
        res_ar = []
        start_time = time.perf_counter()

        for prompt in prompts:
            try:
                res = self.complete(prompt)
                res_ar.append(res)
            except Exception as e:
                res_ar.append(e)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "llm.batch_chat",
            extra={
                "duration_ms": f'{duration_ms:.1f}',
                "provider": self._main_provider,
                "model": self._main_model,
                "status": "ok",
            }
        )
        return res_ar


    def stream_chat(self, prompt: str) -> Iterator[str]:
        messages = self._build_messages('user', prompt)
        start_time = time.perf_counter()
        answer:str = ''
        try:
            with timeout(self._response_timeout):
                response = self._main_client.chat.completions.create(
                    model=self._main_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200,
                    stream=True,
                    stream_options={"include_usage":True}
                )
                ttft_ms:float|None = None
                for chunk in response:
                    if not ttft_ms:
                        ttft_ms = (time.perf_counter() - start_time) * 1000
                    if chunk.choices and chunk.choices[0].delta.content:
                        answer+=chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(
                            "llm.stream_chat",
                            extra={
                                "ttft_ms": f'{ttft_ms:.1f}',
                                "duration_ms": f'{duration_ms:.1f}',
                                "provider": self._main_provider,
                                "model": self._main_model,
                                "tokens": chunk.usage.total_tokens,
                                "status": "ok"
                            }
                        )
                        self._build_messages('assistant', answer)
        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.stream_chat",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "timeout_error",
                }
            )
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else self._response_timeout * 1000
            logger.info(
                "llm.stream_chat",
                extra={
                    "duration_ms": f'{duration_ms:.1f}',
                    "provider": self._main_provider,
                    "model": self._main_model,
                    "tokens": 0,
                    "status": "other_exception",
                    "error": e
                }
            )
            raise