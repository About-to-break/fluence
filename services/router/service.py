import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import nltk
from logging_tools import configure_global_logging
from telephon import telemetry

from config import load_config
from internal import decision_metrics, pipeline
from internal.misc import queue
from internal.routing_core.router import NoneRouterDecisionException


def setup_nltk():
    """Download NLTK data if not present."""
    nltk_data_dir = os.environ.get("NLTK_DATA", None)

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/words", "words"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, resource in resources:
        try:
            if nltk_data_dir:
                nltk.data.find(path, paths=[nltk_data_dir])
            else:
                nltk.data.find(path)
            logging.debug("NLTK resource '%s' already exists", resource)
        except LookupError:
            logging.info("Downloading NLTK resource: %s", resource)
            if nltk_data_dir:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            else:
                nltk.download(resource, quiet=True)


def _resolve_wait_time(t_hand: float, t_arr) -> float:
    if isinstance(t_arr, (int, float)):
        return max(0.0, t_hand - float(t_arr))

    return 0.0


def _monotonic() -> float:
    return time.monotonic()


def _record_request_end(metrics_server, *, success: bool, t_hand: float, t_arr) -> None:
    metrics_server.record_request_end(
        success=success,
        service_time=_monotonic() - t_hand,
        wait_time=_resolve_wait_time(t_hand, t_arr),
    )


async def handle_message(
    message,
    *,
    producer,
    metrics_server,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    active_pipeline,
    fast_key_value: str | None,
    quality_key_value: str | None,
    t_arr=None,
):
    async with semaphore:
        loop = asyncio.get_running_loop()
        t_hand = _monotonic()
        metrics_server.record_request_start()

        try:
            func = partial(active_pipeline, message.body, fast_key_value, quality_key_value)
            key = await loop.run_in_executor(executor, func)

            await producer.produce(message=message.body, key=key)
            await message.ack()
            _record_request_end(metrics_server, success=True, t_hand=t_hand, t_arr=t_arr)

        except pipeline.EmptyPayloadError as exc:
            logging.warning("Empty payload: %s", exc)
            await message.ack()
            _record_request_end(metrics_server, success=False, t_hand=t_hand, t_arr=t_arr)
        except NoneRouterDecisionException as exc:
            logging.error("No routing decision was calculated: %s", exc)
            await message.ack()
            _record_request_end(metrics_server, success=False, t_hand=t_hand, t_arr=t_arr)
        except Exception as exc:
            logging.exception("Unexpected error: %s", exc)
            await message.nack(requeue=True)
            _record_request_end(metrics_server, success=False, t_hand=t_hand, t_arr=t_arr)
            raise


def serve():
    executor = None
    metrics_server = None
    decision_metrics_server = None

    try:
        setup_nltk()

        config = load_config()

        configure_global_logging(
            level=config.LOG_LEVEL,
            file=config.LOG_FILE,
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

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_REQUESTS)
        metrics_server = telemetry.init_metrics(
            service_name=config.METRICS_SERVICE_NAME,
            max_concurrent=config.MAX_CONCURRENT_REQUESTS,
            prefetch_count=config.PREFETCH_COUNT,
        )
        metrics_server.start(port=config.METRICS_PORT)
        decision_metrics_server = decision_metrics.init_decision_metrics()
        try:
            decision_metrics_server.start(port=config.DECISION_METRICS_PORT)
        except Exception as exc:
            logging.warning("Decision metrics server did not start: %s", exc)

        logging.info("Server started")

        async def handler(message, t_arr=None, **kwargs):
            await handle_message(
                message,
                producer=producer,
                metrics_server=metrics_server,
                semaphore=semaphore,
                executor=executor,
                active_pipeline=active_pipeline,
                fast_key_value=fast_key_value,
                quality_key_value=quality_key_value,
                t_arr=t_arr,
            )

        async def main():
            await consumer.start_consuming(handler=handler)

        asyncio.run(main())

    except KeyboardInterrupt:
        logging.info("Shutting down")
        return
    except Exception as exc:
        logging.exception("Critical error: %s", exc)
        return
    finally:
        if decision_metrics_server is not None:
            decision_metrics_server.stop()
        if metrics_server is not None:
            metrics_server.stop()
        if executor is not None:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    serve()
