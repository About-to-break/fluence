import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from logging_tools import configure_global_logging
from config import load_config
from internal.misc import healthcheck
from internal.misc import queue
from internal.nmt import nmt
from internal import pipeline


def serve():
    try:
        config = load_config()

        configure_global_logging(
            level=config.LOG_LEVEL,
            file=config.LOG_FILE
        )

        logging.debug(config)

        logging.info("Performing health check...")
        if healthcheck.healthcheck():
            logging.info("Health check is done")
        else:
            logging.error("Health check failed")
            raise SystemExit(1)

        if config.PIPELINE == "prod":
            logging.info("Set to production pipeline\n Downloading model...")
            translator = nmt.MachineTranslator(
                source_language=config.NMT_SRC_LANG,
                target_language=config.NMT_TGT_LANG,
                model_name=config.NMT_MODEL,
                max_tokens=config.NMT_MAX_NEW_TOKENS
            )
        elif config.PIPELINE == "test":
            logging.info("Set to test pipeline")
            translator = None

        active_pipeline = pipeline.get_pipeline(config)

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
                    func = partial(active_pipeline, translator, message.body)

                    result = await loop.run_in_executor(
                        executor,
                        func
                    )

                    # all good do ack
                    await producer.produce(result)

                except pipeline.EmptyPayloadError as e0:
                    logging.warning(f"Empty payload: {e0}")
                    # do ack to kill bad message
                    raise ValueError(f"Empty payload: {e0}")
                except nmt.TranslationEmptyError as e1:
                    logging.error(f"Translation empty: {e1}")
                    # do ack because it is our error
                    raise
                except Exception as e2:
                    logging.exception(f"Unexpected error: {e2}")
                    # do nack cause its unexpected our error
                    raise

        async def main():
            await consumer.start_consuming(handler=handler)

        asyncio.run(main())
        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        return
    except Exception:
        return


if __name__ == '__main__':
    serve()
