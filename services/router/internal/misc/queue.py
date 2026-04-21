from abc import ABC, abstractmethod
from typing import Callable
from types import SimpleNamespace
from rabbitmq_tools import async_rabbitmq

class IncomingMessage(ABC):
    def __init__(self, body: bytes):
        self.body = body

    @abstractmethod
    async def ack(self):
        pass

    @abstractmethod
    async def nack(self, requeue: bool = True):
        pass

class RabbitMessage(IncomingMessage):
    def __init__(self, raw_message):
        super().__init__(raw_message.body)
        self._msg = raw_message

    async def ack(self):
        await self._msg.ack()

    async def nack(self, requeue: bool = True):
        await self._msg.nack(requeue=requeue)

class KafkaMessage(IncomingMessage):
    def __init__(self, record, consumer):
        super().__init__(record.value)
        self._record = record
        self._consumer = consumer

    async def ack(self):
        # commit offset
        await self._consumer.commit()

    async def nack(self, requeue: bool = True):
        if requeue:
            return
        else:
            await self._consumer.commit()

class MessageProducer(ABC):
    @abstractmethod
    async def produce(self, message, key=None):
        pass


class MessageConsumer(ABC):
    @abstractmethod
    async def start_consuming(self, handler: Callable):
        pass

class KafkaProducer(MessageProducer):
    def __init__(self, config: SimpleNamespace):
        # init aiokafka producer here
        pass

    async def produce(self, message, key=None):
        pass


class KafkaConsumer(MessageConsumer):
    def __init__(self, config: SimpleNamespace):
        pass

    async def start_consuming(self, handler: Callable):
        pass
class RabbitProducer(MessageProducer):
    def __init__(self, config: SimpleNamespace):
        self._producer = async_rabbitmq.RabbitProducerAIO(
            uri=config.RABBITMQ_URI,
            exchange=config.RABBITMQ_EXCHANGE,
            key=config.RABBITMQ_ROUTING_KEY,
            retries=config.RABBITMQ_RETRIES,
        )

    async def produce(self, message, key=None):
        await self._producer.produce(body=message, routing_key=key)


class RabbitConsumer(MessageConsumer):
    def __init__(self, config: SimpleNamespace):
        self._consumer = async_rabbitmq.RabbitConsumerAIO(
            uri=config.RABBITMQ_URI,
            prefetch_count=int(config.PREFETCH_COUNT),
            queue=config.RABBITMQ_QUEUE,
        )

    async def start_consuming(self, handler: Callable):
        await self._consumer.consume(handler)

def get_broker(config: SimpleNamespace):
    if config.MESSAGE_BROKER == "rabbitmq":
        return {
            "producer": RabbitProducer(config),
            "consumer": RabbitConsumer(config)
        }
    elif config.MESSAGE_BROKER == "kafka":
        return {
            "producer": KafkaProducer(config),
            "consumer": KafkaConsumer(config)
        }
    else:
        raise ValueError("Unsupported broker")
