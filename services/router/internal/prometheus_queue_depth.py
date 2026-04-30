import json
import logging
import math
import time
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


class PrometheusQueueDepthClient:
    QUERY = 'q_total_llm{service="llm"}'

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 1.0,
        ttl_seconds: float = 15.0,
        enabled: bool = True,
        opener=None,
        monotonic=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(float(timeout_seconds), 0.001)
        self.ttl_seconds = max(float(ttl_seconds), 0.0)
        self.enabled = bool(enabled)
        self._opener = opener or urlopen
        self._monotonic = monotonic or time.monotonic
        self._last_success_value: int | None = None
        self._last_fetch_monotonic: float | None = None

    @classmethod
    def from_config(cls, config: SimpleNamespace) -> "PrometheusQueueDepthClient":
        return cls(
            base_url=config.PROMETHEUS_URL,
            timeout_seconds=config.PROMETHEUS_TIMEOUT_SECONDS,
            ttl_seconds=config.PROMETHEUS_QUERY_TTL_SECONDS,
            enabled=config.PROMETHEUS_ENABLED,
        )

    def get_llm_queue_depth(self) -> int | None:
        if not self.enabled:
            return None

        now = self._monotonic()
        if self._last_fetch_monotonic is not None:
            age = now - self._last_fetch_monotonic
            if age < self.ttl_seconds:
                return self._last_success_value

        self._last_fetch_monotonic = now

        try:
            value = self._query_queue_depth()
        except Exception as exc:
            logger.warning("Failed to refresh LLM queue depth from Prometheus: %s", exc)
            return self._last_success_value

        self._last_success_value = value
        return value

    def _query_queue_depth(self) -> int:
        query_string = urlencode({"query": self.QUERY})
        request = Request(
            f"{self.base_url}/api/v1/query?{query_string}",
            headers={"Accept": "application/json"},
        )

        with self._opener(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)

        return self._parse_response(payload)

    def _parse_response(self, payload: dict) -> int:
        if not isinstance(payload, dict):
            raise ValueError("Prometheus response is not a JSON object")

        if payload.get("status") != "success":
            raise ValueError("Prometheus query did not succeed")

        data = payload.get("data")
        if not isinstance(data, dict) or data.get("resultType") != "vector":
            raise ValueError("Prometheus query did not return an instant vector")

        result = data.get("result")
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("Prometheus query did not return exactly one series")

        sample = result[0]
        if not isinstance(sample, dict):
            raise ValueError("Prometheus query returned an invalid sample")

        value = sample.get("value")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Prometheus query returned an invalid value tuple")

        numeric = float(value[1])
        if not math.isfinite(numeric):
            raise ValueError("Prometheus query returned a non-finite value")
        if numeric < 0:
            raise ValueError("Prometheus query returned a negative queue depth")

        return int(numeric)
