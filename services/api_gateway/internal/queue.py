import json
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Awaitable, Callable

from rabbitmq_tools import async_rabbitmq


class MessageProducer(ABC):
    @abstractmethod
    async def produce(self, message: dict):
        pass

    async def close(self):
        return None


class MessageConsumer(ABC):
    @abstractmethod
    async def start_consuming(self, handler: Callable[[object], Awaitable[None]]):
        pass

    async def close(self):
        return None


class RabbitProducer(MessageProducer):
    def __init__(self, config: SimpleNamespace):
        self._producer = async_rabbitmq.RabbitProducerAIO(
            uri=config.RABBITMQ_URI,
            exchange=config.RABBITMQ_EXCHANGE,
            key=config.RABBITMQ_ROUTING_KEY,
            retries=config.RABBITMQ_RETRIES,
        )

    async def produce(self, message: dict):
        payload = json.dumps(message).encode("utf-8")
        await self._producer.produce(payload)

    async def close(self):
        for method_name in ("close", "stop"):
            method = getattr(self._producer, method_name, None)
            if method is None:
                continue

            result = method()
            if hasattr(result, "__await__"):
                await result
            return


class RabbitConsumer(MessageConsumer):
    def __init__(self, config: SimpleNamespace):
        self._consumer = async_rabbitmq.RabbitConsumerAIO(
            uri=config.RABBITMQ_URI,
            prefetch_count=1,
            queue=config.RABBITMQ_OUTPUT_QUEUE,
        )

    async def start_consuming(self, handler: Callable[[object], Awaitable[None]]):
        await self._consumer.consume(handler)

    async def close(self):
        for method_name in ("close", "stop"):
            method = getattr(self._consumer, method_name, None)
            if method is None:
                continue

            result = method()
            if hasattr(result, "__await__"):
                await result
            return


def get_producer(config: SimpleNamespace) -> MessageProducer:
    if config.MESSAGE_BROKER == "rabbitmq":
        return RabbitProducer(config)

    raise ValueError("Unsupported broker")


def get_consumer(config: SimpleNamespace) -> MessageConsumer:
    if config.MESSAGE_BROKER == "rabbitmq":
        return RabbitConsumer(config)

    raise ValueError("Unsupported broker")
