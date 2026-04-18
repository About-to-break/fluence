import logging
import asyncio
import sys
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

        handler_semaphore = asyncio.Semaphore(int(config.MAX_CONCURRENT_REQUESTS))

        active_pipeline = pipeline.get_pipeline(config)
        broker = queue.get_broker(config)
        producer = broker["producer"]
        consumer = broker["consumer"]

        await producer.connect()

        shutdown_event = asyncio.Event()
        logging.info("Server started")

        # ========== ОСНОВНОЙ HANDLER ==========
        async def handler(message, wait_timer):
            async with handler_semaphore:
                try:
                    result = await active_pipeline(
                        vllm_connector,
                        config.OPENAI_API_MAX_TOKENS,
                        message.body
                    )

                    await producer.produce(result)
                    await message.ack()
                    logging.debug(f"Successfully processed message")
                    return result

                except pipeline.EmptyPayloadError as e0:
                    logging.warning(f"Empty payload, discarding: {e0}")
                    await message.nack(requeue=False)
                    return None

                except pipeline.EmptyResponseError as e1:
                    logging.error(f"Empty response, sending original: {e1}")
                    await producer.produce(message.body)
                    await message.ack()
                    return message.body.decode()

                except VLLMTimeoutError as te:
                    logging.warning(f"vLLM timeout, fallback to original: {te}")
                    await producer.produce(message.body)
                    await message.ack()
                    return message.body.decode()

                except VLLMConnectionError as ce:
                    logging.error(f"vLLM connection error, requeue: {ce}")
                    await message.nack(requeue=True)
                    raise

                except Exception as e2:
                    logging.exception(f"Unexpected error, requeue: {e2}")
                    await message.nack(requeue=True)
                    raise

        # ========== FIRE-AND-FORGET ОБЁРТКА ==========
        async def background_handler(message, wait_timer):
            """Запускает handler в фоне, не дожидаясь результата."""
            asyncio.create_task(handler(message, wait_timer))

        async def main():
            await consumer.start_consuming(handler=background_handler)
            await shutdown_event.wait()

        try:
            await main()
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
            shutdown_event.set()
        finally:
            logging.info("Graceful shutdown complete")

    except Exception as e:
        logging.exception(f"Critical error: {e}")
        raise


if __name__ == '__main__':
    if sys.platform != "win32":
        import uvloop

        uvloop.install()

    asyncio.run(serve())