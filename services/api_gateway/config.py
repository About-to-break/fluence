import logging
import os
from types import SimpleNamespace

from dotenv import load_dotenv


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(env_file: str = ".env") -> SimpleNamespace:
    if os.path.exists(env_file):
        load_dotenv(env_file)

    config_vars = {
        key: os.getenv(key)
        for key in os.environ.keys()
        if os.getenv(key) is not None
    }

    log_level_str = str(config_vars.get("LOG_LEVEL", "INFO")).upper()
    config_vars["LOG_LEVEL"] = getattr(logging, log_level_str, logging.INFO)

    if "API_PORT" in config_vars:
        config_vars["API_PORT"] = _to_int(config_vars["API_PORT"], 8080)
    else:
        config_vars["API_PORT"] = 8080

    config_vars["RABBITMQ_OUTPUT_PREFETCH_COUNT"] = _to_int(
        config_vars.get("RABBITMQ_OUTPUT_PREFETCH_COUNT"),
        32,
    )

    return SimpleNamespace(**config_vars)
