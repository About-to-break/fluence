import json
import logging
from types import SimpleNamespace


class EmptyPayloadError(Exception):
    pass


def run_pipeline(message: bytes, option_fast: str = None, option_quality: str = None):
    logging.info(f"Prod pipeline still empty")
    key = option_fast
    return key


def run_test_pipeline(message: bytes, option_fast: str = None, option_quality: str = None):
    logging.info(f"Test pipeline worked")
    key = option_fast


def get_pipeline(config: SimpleNamespace):
    if config.PIPELINE == "prod":
        return run_pipeline
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
