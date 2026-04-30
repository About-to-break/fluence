import orjson
import logging
from types import SimpleNamespace
from .vllm.connector import *


class EmptyPayloadError(Exception):
    pass

class EmptyResponseError(Exception):
    pass


async def run_pipeline(connector: VLLMConnector, max_tokens: int, message: bytes):
    parsed_body = orjson.loads(message.decode("utf-8"))
    text = parsed_body["text"]
    uuid = parsed_body["uuid"]

    logging.info(f"Decoded body with uuid {uuid} and text {text}")

    if text == "" or uuid == "":
        raise EmptyPayloadError("Empty payload")

    logging.info(f"Requesting translation...")
    try:
        translation = await connector.translate(prompt=text, max_tokens=max_tokens)

        if translation == "":
            raise EmptyResponseError(f"Generated empty translation for {text}")

        logging.info(f"Translated text: {translation}")

        parsed_body.update({"uuid": uuid, "text": translation})

        return orjson.dumps(parsed_body)

    except VLLMTimeoutError as e:
        raise VLLMTimeoutError(f"Encountered vLLM timeout error with {uuid}, fallbacking to nmt...") from e


async def run_test_pipeline(connector: VLLMConnector, max_tokens: int, message: bytes):
    logging.info(f"Test pipeline worked")
    return message


def get_pipeline(config: SimpleNamespace):
    if config.PIPELINE == "prod":
        return run_pipeline
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
