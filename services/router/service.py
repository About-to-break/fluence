import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from logging_tools import configure_global_logging
from config import load_config
from internal.misc import queue
from internal.routing_core.router import NoneRouterDecisionException
from internal import pipeline


def serve():
    try:
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
                    func = partial(active_pipeline,
                                   message.body,
                                   fast_key_value,
                                   quality_key_value)

                    # Actually the same message
                    key = await loop.run_in_executor(
                        executor,
                        func
                    )

                    # all good do ack
                    await producer.produce(message=message, key=key)

                except pipeline.EmptyPayloadError as e0:
                    logging.warning(f"Empty payload: {e0}")
                    # do ack to kill bad message
                    raise ValueError(f"Empty payload: {e0}")
                except NoneRouterDecisionException as e1:
                    logging.error(f"No routing decision was calculated: {e1}")
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
        logging.info("Shutting down")
        return
    except Exception as e:
        logging.exception(f"Critical error: {e}")
        return


if __name__ == '__main__':
    serve()
