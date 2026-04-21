import logging
from types import SimpleNamespace
import orjson
from .nmt.nmt import *
from .nmt.ct2nmt import *


class EmptyPayloadError(Exception):
    pass


def run_pipeline(translator: MachineTranslator, message: bytes):
    parsed_body = orjson.loads(message.decode("utf-8"))
    text = parsed_body["text"]
    uuid = parsed_body["uuid"]

    logging.info(f"Decoded body with uuid {uuid} and text {text}")

    if text == "" or uuid == "":
        raise EmptyPayloadError("Empty payload")

    logging.info(f"Requesting translation...")
    translation = translator.translate(text)

    if translation == "":
        raise TranslationEmptyError(f"Generated empty translation for {text}")

    logging.info(f"Translated text: {translation}")

    new_message = orjson.dumps({"uuid": uuid, "text": translation})

    return new_message


def run_test_pipeline(translator, message: bytes):
    logging.info(f"Test pipeline worked")
    return message


async def run_batch_pipeline(translator: CT2Translator, message: bytes, executor=None, max_batch_size=8):
    """
    Новый пайплайн — асинхронный батчинг.
    translator здесь используется только для получения batcher'а.
    """
    parsed_body = orjson.loads(message.decode("utf-8"))
    text = parsed_body["text"]
    uuid = parsed_body["uuid"]

    if text == "" or uuid == "":
        raise EmptyPayloadError("Empty payload")

    batcher = get_batcher(translator, executor=executor, max_batch_size=max_batch_size)
    translation = await batcher.translate(text)

    if translation == "":
        raise TranslationEmptyError(f"Empty translation for {text}")

    return orjson.dumps({"uuid": uuid, "text": translation})


def get_pipeline(config: SimpleNamespace, executor=None, max_batch_size=8):
    if config.PIPELINE == "prod":
        return run_pipeline
    elif config.PIPELINE == "batch":
        async def batch_with_executor(translator, message, **kwargs):
            return await run_batch_pipeline(translator, message, executor=executor, max_batch_size=max_batch_size)

        return batch_with_executor
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
