import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
import os
import nltk
from logging_tools import configure_global_logging
from config import load_config
from internal.misc import queue
from internal.routing_core.router import NoneRouterDecisionException
from internal import pipeline


def setup_nltk():
    """Download NLTK data if not present."""
    nltk_data_dir = os.environ.get('NLTK_DATA', None)

    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/words', 'words'),
        ('corpora/wordnet', 'wordnet'),
    ]

    for path, resource in resources:
        try:
            if nltk_data_dir:
                nltk.data.find(path, paths=[nltk_data_dir])
            else:
                nltk.data.find(path)
            logging.debug(f"NLTK resource '{resource}' already exists")
        except LookupError:
            logging.info(f"Downloading NLTK resource: {resource}")
            if nltk_data_dir:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            else:
                nltk.download(resource, quiet=True)


def serve():
    try:
        # Сначала скачиваем NLTK данные
        setup_nltk()

        config = load_config()

        configure_global_logging(
            level=config.LOG_LEVEL,
            file=config.LOG_FILE
        )

        logging.debug(config)

        if config.PIPELINE == "prod":
            logging.info("Set to production pipeline")
        elif config.PIPELINE == "test":
            logging.info("Set to test pipeline")
        else:
            raise ValueError("Pipeline type not supported")

        active_pipeline = pipeline.get_pipeline(config)

        fast_key_value = config.KEY_FAST
        quality_key_value = config.KEY_QUALITY

        broker = queue.get_broker(config)

        producer = broker["producer"]
        consumer = broker["consumer"]

        semaphore = asyncio.Semaphore(1)
        executor = ThreadPoolExecutor(max_workers=1)

        logging.info("Server started")

        async def handler(message):
            async with semaphore:
                loop = asyncio.get_running_loop()

                try:
                    func = partial(active_pipeline, message.body, fast_key_value, quality_key_value)

                    key = await loop.run_in_executor(executor, func)

                    await producer.produce(message=message.body, key=key)
                    await message.ack()

                except pipeline.EmptyPayloadError as e0:
                    logging.warning(f"Empty payload: {e0}")
                    await message.ack()  # Подтверждаем плохое сообщение
                except NoneRouterDecisionException as e1:
                    logging.error(f"No routing decision was calculated: {e1}")
                    await message.ack()  # Наша ошибка, но сообщение обработано
                except Exception as e2:
                    logging.exception(f"Unexpected error: {e2}")
                    await message.nack(requeue=True)  # Возвращаем в очередь

        async def main():
            await consumer.start_consuming(handler=handler)

        asyncio.run(main())
        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        logging.info("Shutting down")
        return
    except Exception as e:
        logging.exception(f"Critical error: {e}")
        return


if __name__ == '__main__':
    serve()