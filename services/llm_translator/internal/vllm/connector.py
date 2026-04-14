import asyncio
from typing import List, Dict
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
import logging


class SomeLLMError(Exception):
    pass


class VLLMTimeoutError(Exception):
    pass


class VLLMConnectionError(Exception):
    pass


class VLLMConnector:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self,
                 system_messages: List[Dict[str, str]],
                 base_url: str,
                 model: str,
                 api_key: str = '',
                 timeout: int = 10,
                 max_retries: int = 10,
                 max_concurrency: int = 32,
                 ):

        logging.info(f'Initializing a vLLM connector instance')

        if hasattr(self, '_initialized'):
            return

        self.system_messages = system_messages
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._active_requests = 0
        self._total_timeouts = 0
        self._total_errors = 0

        self._initialized = True
        logging.info("vLLM connector initialized")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls, **kwargs) -> 'VLLMConnector':
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    async def translate(self, prompt: str, max_tokens: int) -> str:

        async with self._semaphore:
            self._active_requests += 1

            try:
                logging.info(f'Requesting vLLM {self._active_requests} / {self.max_concurrency}')

                messages = self.system_messages + [
                    {"role": "user", "content": prompt},
                ]
                completion = await self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_tokens,
                    model=self.model,
                    stream=False,
                    timeout=self.timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
                logging.info("vLLM completion requested successfully")
                return completion.choices[0].message.content.strip()

            except APITimeoutError as e1:
                self._total_timeouts += 1
                logging.warning(f"vLLM timeout after {self.timeout}s (total timeouts: {self._total_timeouts})")
                raise VLLMTimeoutError("vLLM connection timeout") from e1

            except APIConnectionError as e2:
                self._total_errors += 1
                logging.error(f"vLLM connection error: {e2}")
                raise VLLMConnectionError("vLLM connection error") from e2

            except Exception as e3:
                self._total_errors += 1
                logging.error(f"vLLM unexpected error: {e3}")
                raise SomeLLMError("vLLM unexpected error") from e3

            finally:
                self._active_requests -= 1
