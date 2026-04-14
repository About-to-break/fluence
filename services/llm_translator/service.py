import logging
import asyncio
from logging_tools import configure_global_logging
from config import load_config, get_prompt
from internal.misc import queue
from internal.vllm.connector import *
from internal import pipeline


async def serve():
    try:
        config = load_config()

        configure_global_logging(
            level=config.LOG_LEVEL,
            file=config.LOG_FILE
        )

        logging.debug(config)

        if config.PIPELINE == "prod":
            logging.info(f"Production pipeline with system prompt: {config.SYSTEM_PROMPT[:50]}...")
            system_prompt = get_prompt(config.SYSTEM_PROMPT)
            vllm_connector = await VLLMConnector.get_instance(
                base_url=config.OPENAI_API_BASE_URL,
                model=config.OPENAI_API_MODEL,
                api_key=config.OPENAI_API_KEY,
                timeout=config.OPENAI_API_TIMEOUT,
                max_retries=config.OPENAI_API_MAX_RETRIES,
                max_concurrency=config.OPENAI_API_MAX_CONCURRENCY,
                system_messages=system_prompt
            )
        elif config.PIPELINE == "test":
            logging.info("Test pipeline")
            system_prompt = None
            vllm_connector = None
        else:
            raise ValueError("Unsupported pipeline type")

        active_pipeline = pipeline.get_pipeline(config)
        broker = queue.get_broker(config)
        producer = broker["producer"]
        consumer = broker["consumer"]

        await producer.connect()

        # Семафор для graceful shutdown, а не для ограничения конкурентности
        shutdown_event = asyncio.Event()

        async def handler(message):
            """Асинхронный обработчик — конкурентность управляется семафором в VLLMConnector"""
            try:
                result = await active_pipeline(
                    vllm_connector,
                    config.OPENAI_API_MAX_TOKENS,
                    message.body
                )

                await producer.produce(result)
                logging.debug(f"Successfully processed message")

            except pipeline.EmptyPayloadError as e0:
                logging.warning(f"Empty payload, discarding: {e0}")
                # Ack — плохое сообщение удаляем из очереди

            except pipeline.EmptyResponseError as e1:
                logging.error(f"Empty response, sending original: {e1}")
                # Отправляем оригинальное сообщение как fallback
                await producer.produce(message.body)

            except VLLMTimeoutError as te:
                logging.warning(f"vLLM timeout, fallback to original: {te}")
                # Отправляем то, что было до вызова LLM (оригинал из очереди)
                await producer.produce(message.body)

            except Exception as e2:
                logging.exception(f"Unexpected error, requeue: {e2}")
                # Nack с requeue — пусть другая реплика попробует
                raise  # Пробрасываем для nack в consumer'е

        async def main():
            await consumer.start_consuming(handler=handler)

            # Ждем сигнала завершения
            await shutdown_event.wait()

        # Запускаем с обработкой graceful shutdown
        try:
            await main()
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
            shutdown_event.set()
        finally:
            # await consumer.close()
            # await producer.close()
            logging.info("Graceful shutdown complete")

    except Exception as e:
        logging.exception(f"Critical error: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(serve())