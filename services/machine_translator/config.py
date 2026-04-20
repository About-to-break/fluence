import os
import logging
import re
from dotenv import load_dotenv
from types import SimpleNamespace


def _normalize_metrics_service_name(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate).strip("_")

    if not candidate or candidate[0].isdigit():
        return "nmt"

    return candidate


def load_config(env_file=".env") -> SimpleNamespace:

    if os.path.exists(env_file):
        load_dotenv(env_file)

    config_vars = {key: os.getenv(key) for key in os.environ.keys() if os.getenv(key) is not None}
    config_vars.setdefault("METRICS_PORT", "9090")
    config_vars.setdefault("MAX_CONCURRENT_REQUESTS", "32")
    config_vars.setdefault("PREFETCH_COUNT", config_vars["MAX_CONCURRENT_REQUESTS"])
    config_vars["METRICS_SERVICE_NAME"] = _normalize_metrics_service_name(
        config_vars.get("METRICS_SERVICE_NAME", "nmt")
    )

    try:
        log_level_str = config_vars.get("LOG_LEVEL", "INFO").upper()
        config_vars["LOG_LEVEL"] = getattr(logging, log_level_str, logging.INFO)
    except Exception:
        pass

    return SimpleNamespace(**config_vars)
