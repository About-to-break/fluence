import json
import logging
from types import SimpleNamespace
from .nmt.nmt import *

class EmptyPayloadError(Exception):
    pass
def run_pipeline(translator: MachineTranslator, message: bytes):

    parsed_body = json.loads(message.decode("utf-8"))
    text = parsed_body["text"]
    uuid = parsed_body["uuid"]

    logging.debug(f"Decoded body with uuid {uuid} and text {text}")

    if text == "" or uuid == "":
        raise EmptyPayloadError("Empty payload")

    translation = translator.translate(text)

    if translation == "":
        raise TranslationEmptyError(f"Generated empty translation for {text}")

    logging.debug(f"Translated text: {translation}")

    new_message = json.dumps({"uuid": uuid, "text": translation}).encode("utf-8")

    return new_message

def run_test_pipeline(translator, message: bytes):
    logging.debug(f"Test pipeline worked")
    return message

def get_pipeline(config: SimpleNamespace):
    if config.PIPELINE == "prod":
        return run_test_pipeline
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
