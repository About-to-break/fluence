import os
import logging
from dotenv import load_dotenv
from types import SimpleNamespace

def load_config(env_file=".env") -> SimpleNamespace:

    if os.path.exists(env_file):
        load_dotenv(env_file)

    config_vars = {key: os.getenv(key) for key in os.environ.keys() if os.getenv(key) is not None}

    try:
        log_level_str = config_vars.get("LOG_LEVEL", "INFO").upper()
        config_vars["LOG_LEVEL"] = getattr(logging, log_level_str, logging.INFO)
    except Exception:
        pass

    return SimpleNamespace(**config_vars)