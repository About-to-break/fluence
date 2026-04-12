import json
import logging
from types import SimpleNamespace


class EmptyPayloadError(Exception):
    pass


def run_pipeline(message: bytes):
    logging.info(f"Still empty")
    return message


def run_test_pipeline(message: bytes):
    logging.info(f"Test pipeline worked")
    return message


def get_pipeline(config: SimpleNamespace):
    if config.PIPELINE == "prod":
        return run_pipeline
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
