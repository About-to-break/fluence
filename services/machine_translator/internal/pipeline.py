import json
import logging
from types import SimpleNamespace
## починить
from .nmt.nmt import *
from .nmt.ct2nmt import *


class EmptyPayloadError(Exception):
    pass
def run_pipeline(translator: MachineTranslator, message: bytes):
    parsed_body = json.loads(message.decode("utf-8"))
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

    new_message = json.dumps({"uuid": uuid, "text": translation}).encode("utf-8")

    return new_message


def run_test_pipeline(translator, message: bytes):
    logging.info(f"Test pipeline worked")
    return message

async def run_batch_pipeline(translator: CT2Translator, message: bytes):
    """
    Новый пайплайн — асинхронный батчинг.
    translator здесь используется только для получения batcher'а.
    """
    parsed_body = json.loads(message.decode("utf-8"))
    text = parsed_body["text"]
    uuid = parsed_body["uuid"]

    if text == "" or uuid == "":
        raise EmptyPayloadError("Empty payload")

    batcher = get_batcher(translator)
    translation = await batcher.translate(text)

    if translation == "":
        raise TranslationEmptyError(f"Empty translation for {text}")

    return json.dumps({"uuid": uuid, "text": translation}, ensure_ascii=False).encode("utf-8")





def get_pipeline(config: SimpleNamespace):
    if config.PIPELINE == "prod":
        return run_pipeline  # старый поштучный
    elif config.PIPELINE == "batch":
        return run_batch_pipeline  # новый батчевый
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
