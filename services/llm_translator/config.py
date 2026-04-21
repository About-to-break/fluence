import os
import json
import logging
import re
from typing import List, Dict
from dotenv import load_dotenv
from types import SimpleNamespace

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


def _normalize_metrics_service_name(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate).strip("_")

    if not candidate or candidate[0].isdigit():
        return "llm"

    return candidate


def get_prompt(prompt_name: str = "default") -> List[Dict[str, str]]:
    with open("system_prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found")

    prompt_config = prompts[prompt_name]

    # Если старый строковый формат
    if isinstance(prompt_config, str):
        return [{"role": "system", "content": prompt_config}]

    # Если новый формат с "messages" — возвращаем ТОЛЬКО список messages
    if isinstance(prompt_config, dict) and "messages" in prompt_config:
        return prompt_config["messages"]

    # Если уже список — возвращаем как есть
    if isinstance(prompt_config, list):
        return prompt_config

    raise ValueError(f"Invalid prompt format for '{prompt_name}'")

def load_config(env_file=".env") -> SimpleNamespace:

    if os.path.exists(env_file):
        load_dotenv(env_file)

    config_vars = {key: os.getenv(key) for key in os.environ.keys() if os.getenv(key) is not None}

    try:
        log_level_str = config_vars.get("LOG_LEVEL", "INFO").upper()
        config_vars["LOG_LEVEL"] = getattr(logging, log_level_str, logging.INFO)
        config_vars["METRICS_SERVICE_NAME"] = _normalize_metrics_service_name(
            config_vars.get("METRICS_SERVICE_NAME", "llm")
        )

        if "OPENAI_API_TIMEOUT" in config_vars:
            config_vars["OPENAI_API_TIMEOUT"] = _to_int(config_vars["OPENAI_API_TIMEOUT"], 10)

        if "OPENAI_API_MAX_RETRIES" in config_vars:
            config_vars["OPENAI_API_MAX_RETRIES"] = _to_int(config_vars["OPENAI_API_MAX_RETRIES"], 10)

        if "OPENAI_API_MAX_TOKENS" in config_vars:
            config_vars["OPENAI_API_MAX_TOKENS"] = _to_int(config_vars["OPENAI_API_MAX_TOKENS"], 256)

        if "OPENAI_API_MAX_CONCURRENCY" in config_vars:
            config_vars["OPENAI_API_MAX_CONCURRENCY"] = _to_int(config_vars["OPENAI_API_MAX_CONCURRENCY"], 3)

        if "RABBITMQ_PREFETCH_COUNT" in config_vars:
            config_vars["RABBITMQ_PREFETCH_COUNT"] = _to_int(config_vars["RABBITMQ_PREFETCH_COUNT"], 3)

        if "OPENAI_API_TEMP" in config_vars:
            config_vars["OPENAI_API_TEMP"] = _to_float(config_vars["OPENAI_API_TEMP"], 0.0)



    except Exception:
        pass

    return SimpleNamespace(**config_vars)
