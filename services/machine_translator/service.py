from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from logging_tools import logging_tools
from config import load_config
from internal.misc import healthcheck
from internal.misc import queue
from internal.nmt import nmt
from internal import pipeline


def serve():
    config = load_config()

    logger = logging_tools.get_logger(
        level=config.LOG_LEVEL,
        file=config.LOG_FILE
    )

    if not healthcheck.healthcheck(logger):
        logger.error("Server is unhealthy. Stopping.")
        return

    translator = nmt.MachineTranslator(
        source_language=config.NMT_SRC_LANG,
        target_language=config.NMT_TGT_LANG,
        model_name=config.NMT_MODEL,
        max_tokens=config.NMT_MAX_NEW_TOKENS
    )

    active_pipeline = pipeline.get_pipeline(config)

    broker = queue.get_broker(config)

    producer = broker["producer"]
    consumer = broker["consumer"]

    semaphore = asyncio.Semaphore(1)
    executor = ThreadPoolExecutor(max_workers=1)

    async def handler(message):
        async with semaphore:
            loop = asyncio.get_running_loop()

            try:
                func = partial(active_pipeline.run_pipeline, translator, message.body)

                result = await loop.run_in_executor(
                    executor,
                    func
                )

                # all good do ack
                await producer.send(result)
                await message.ack()

            except pipeline.EmptyPayloadError as e0:
                logger.warning(e0)
                # do ack to kill bad message
                await message.ack()
            except nmt.TranslationEmptyError as e1:
                logger.error(e1)
                # do ack cause it is our error
                await message.ack()
            except Exception as e2:
                logger.exception(e2)
                # do nack cause its unexpected our error
                await message.nack(requeue=True)


    async def main():
        await consumer.start_consuming(handler=handler)

    asyncio.run(main())
    executor.shutdown(wait=True)