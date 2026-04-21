import logging
import os
import re
from types import SimpleNamespace

from dotenv import load_dotenv


_DEFAULT_METRICS_SERVICE_NAME = "router"
_METRICS_SERVICE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_:]+")


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


def _normalize_metrics_service_name(
    value: str | None,
    default: str = _DEFAULT_METRICS_SERVICE_NAME,
) -> str:
    if value is None:
        return default

    normalized = _METRICS_SERVICE_NAME_PATTERN.sub("_", value.strip().lower()).strip("_:")
    if not normalized:
        return default

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_:]*$", normalized):
        return default

    return normalized


def load_config(env_file=".env") -> SimpleNamespace:
    if os.path.exists(env_file):
        load_dotenv(env_file)

    config_vars = {
        key: os.getenv(key)
        for key in os.environ.keys()
        if os.getenv(key) is not None
    }

    try:
        log_level_str = config_vars.get("LOG_LEVEL", "INFO").upper()
        config_vars["LOG_LEVEL"] = getattr(logging, log_level_str, logging.INFO)
    except Exception:
        pass

    config_vars["MAX_CONCURRENT_REQUESTS"] = _to_int(
        config_vars.get("MAX_CONCURRENT_REQUESTS"),
        1,
    )
    config_vars["PREFETCH_COUNT"] = _to_int(
        config_vars.get("PREFETCH_COUNT"),
        1,
    )
    config_vars["METRICS_PORT"] = _to_int(
        config_vars.get("METRICS_PORT"),
        9090,
    )
    config_vars["DECISION_METRICS_PORT"] = _to_int(
        config_vars.get("DECISION_METRICS_PORT"),
        9093,
    )
    config_vars["METRICS_SERVICE_NAME"] = _normalize_metrics_service_name(
        config_vars.get("METRICS_SERVICE_NAME"),
    )
    config_vars["PROMETHEUS_ENABLED"] = _to_bool(
        config_vars.get("PROMETHEUS_ENABLED"),
        True,
    )
    config_vars["PROMETHEUS_URL"] = config_vars.get(
        "PROMETHEUS_URL",
        "http://prometheus:9090",
    )
    config_vars["PROMETHEUS_TIMEOUT_SECONDS"] = _to_float(
        config_vars.get("PROMETHEUS_TIMEOUT_SECONDS"),
        1.0,
    )
    config_vars["PROMETHEUS_QUERY_TTL_SECONDS"] = _to_float(
        config_vars.get("PROMETHEUS_QUERY_TTL_SECONDS"),
        15.0,
    )

    return SimpleNamespace(**config_vars)
