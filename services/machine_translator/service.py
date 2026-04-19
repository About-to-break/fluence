import sys
import logging
import asyncio
import concurrent.futures
import time

from logging_tools import configure_global_logging
from telephon import telemetry
from config import load_config
from internal.misc import queue
from internal.nmt import nmt, ct2nmt
from internal import pipeline

translation_executor = None

async def serve():
    try:
        config = load_config()

        configure_global_logging(
            level=config.LOG_LEVEL,
            file=config.LOG_FILE
        )

        translation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(config.TRANSLATION_WORKERS),
            thread_name_prefix="translator"
        )

        logging.debug(config)

        if config.PIPELINE == "prod":
            logging.info("Set to production pipeline\n Downloading model...")
            translator = nmt.MachineTranslator(
                source_language=config.NMT_SRC_LANG,
                target_language=config.NMT_TGT_LANG,
                model_name=config.NMT_MODEL,
                max_new_tokens=config.NMT_MAX_NEW_TOKENS,
                max_length=config.NMT_MAX_LENGTH,
            )
        elif config.PIPELINE == "test":
            logging.info("Set to test pipeline")
            translator = None
        elif config.PIPELINE == "batch":
            logging.info("Set to batch pipeline")
            ct2nmt.download_model(
                hf_model_repo=config.CT2NMT_MODEL_REPO,
                target_dir=config.CT2NMT_MODEL_DIR,
                token=config.HF_TOKEN,
            )
            translator = ct2nmt.CT2Translator(
                model_path=config.CT2NMT_MODEL_DIR,
                tgt_lang=config.CT2NMT_TGT_LANG,
                src_lang=config.CT2NMT_SRC_LANG,
            )
            logging.info(f"Set to translate {config.CT2NMT_SRC_LANG} to {config.CT2NMT_TGT_LANG}")
        else:
            raise ValueError("Pipeline type not supported")

        metrics_server = telemetry.init_metrics(service_name="nmt service",
                                                max_concurrent=int(config.MAX_CONCURRENT_REQUESTS),
                                                prefetch_count=int(config.PREFETCH_COUNT)
                                                )

        handler_semaphore = asyncio.Semaphore(int(config.MAX_CONCURRENT_REQUESTS))

        active_pipeline = pipeline.get_pipeline(config, executor=translation_executor)
        broker = queue.get_broker(config)
        producer = broker["producer"]
        consumer = broker["consumer"]

        metrics_server.start(port=int(config.METRICS_PORT))

        await producer.connect()

        shutdown_event = asyncio.Event()
        logging.info("Server started")

        # ========== ОСНОВНОЙ HANDLER (синхронный по логике) ==========
        async def handler(message, t_arr):
            async with handler_semaphore:
                t_hand = time.monotonic()
                metrics_server.record_request_start()
                try:
                    result = await active_pipeline(translator=translator, message=message.body)
                    await producer.produce(result)
                    await message.ack()
                    logging.debug(f"Successfully processed message")
                    metrics_server.record_request_end(success=True, wait_time=t_hand - t_arr, service_time=time.monotonic() - t_hand)
                    return result

                except pipeline.EmptyPayloadError as e0:
                    logging.warning(f"Empty payload, discarding: {e0}")
                    metrics_server.record_request_end(success=True, wait_time=t_hand - t_arr, service_time=time.monotonic() - t_hand)
                    await message.nack(requeue=False)
                    return None

                except nmt.TranslationEmptyError as e1:
                    logging.error(f"Empty response, sending original: {e1}")
                    metrics_server.record_request_end(success=False, wait_time=t_hand - t_arr, service_time=time.monotonic() - t_hand)
                    await producer.produce(message.body)
                    await message.ack()
                    return message.body.decode()

                except Exception as e2:
                    logging.exception(f"Unexpected error, requeue: {e2}")
                    metrics_server.record_request_end(success=False, wait_time=t_hand - t_arr, service_time=time.monotonic() - t_hand)
                    await message.nack(requeue=True)
                    raise

        # ========== FIRE-AND-FORGET ОБЁРТКА ==========
        async def background_handler(message, t_arr):
            """Запускает handler в фоне, не дожидаясь результата."""
            loop = asyncio.get_running_loop()
            loop.create_task(handler(message, t_arr))

        async def main():
            # Передаём background_handler вместо handler
            await consumer.start_consuming(handler=background_handler)
            await shutdown_event.wait()

        # Запускаем с обработкой graceful shutdown
        try:
            await main()
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
            shutdown_event.set()
        finally:
            metrics_server.stop()
            translation_executor.shutdown(wait=True)
            logging.info("Graceful shutdown complete")


    except Exception as e:
        logging.exception(f"Critical error: {e}")
        raise


if __name__ == '__main__':
    if sys.platform != "win32":
        import uvloop

        uvloop.install()

    asyncio.run(serve())