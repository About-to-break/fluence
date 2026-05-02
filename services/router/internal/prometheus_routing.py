import json
import logging
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingMetricsSnapshot:
    rho_nmt: float
    rho_llm: float
    backlog_drain_nmt_seconds: float
    backlog_drain_llm_seconds: float
    kingman_penalty_nmt_seconds: float
    kingman_penalty_llm_seconds: float

    @property
    def system_rho(self) -> float:
        return max(self.rho_nmt, self.rho_llm)


class PrometheusRoutingClient:
    _EXPECTED_SERIES = {
        "approx_rho_nmt": ("rho_nmt", "nmt"),
        "approx_rho_llm": ("rho_llm", "llm"),
        "approx_backlog_drain_nmt_seconds": ("backlog_drain_nmt_seconds", "nmt"),
        "approx_backlog_drain_llm_seconds": ("backlog_drain_llm_seconds", "llm"),
        "approx_kingman_penalty_nmt_seconds": ("kingman_penalty_nmt_seconds", "nmt"),
        "approx_kingman_penalty_llm_seconds": ("kingman_penalty_llm_seconds", "llm"),
    }
    QUERY = " or ".join(
        f'{series_name}{{service="{service}"}}'
        for series_name, (_, service) in _EXPECTED_SERIES.items()
    )

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
        self._last_success_value: RoutingMetricsSnapshot | None = None
        self._last_fetch_monotonic: float | None = None

    @classmethod
    def from_config(cls, config: SimpleNamespace) -> "PrometheusRoutingClient":
        return cls(
            base_url=config.PROMETHEUS_URL,
            timeout_seconds=config.PROMETHEUS_TIMEOUT_SECONDS,
            ttl_seconds=config.PROMETHEUS_QUERY_TTL_SECONDS,
            enabled=config.PROMETHEUS_ENABLED,
        )

    def get_routing_snapshot(self) -> RoutingMetricsSnapshot | None:
        if not self.enabled:
            return self._last_success_value

        now = self._monotonic()
        if self._last_fetch_monotonic is not None:
            age = now - self._last_fetch_monotonic
            if age < self.ttl_seconds:
                return self._last_success_value

        self._last_fetch_monotonic = now

        try:
            value = self._query_snapshot()
        except Exception as exc:
            logger.warning("Failed to refresh routing snapshot from Prometheus: %s", exc)
            return self._last_success_value

        self._last_success_value = value
        return value

    def _query_snapshot(self) -> RoutingMetricsSnapshot:
        query_string = urlencode({"query": self.QUERY})
        request = Request(
            f"{self.base_url}/api/v1/query?{query_string}",
            headers={"Accept": "application/json"},
        )

        with self._opener(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)

        return self._parse_response(payload)

    def _parse_response(self, payload: dict) -> RoutingMetricsSnapshot:
        if not isinstance(payload, dict):
            raise ValueError("Prometheus response is not a JSON object")

        if payload.get("status") != "success":
            raise ValueError("Prometheus query did not succeed")

        data = payload.get("data")
        if not isinstance(data, dict) or data.get("resultType") != "vector":
            raise ValueError("Prometheus query did not return an instant vector")

        result = data.get("result")
        if not isinstance(result, list) or not result:
            raise ValueError("Prometheus query did not return any series")

        parsed_values: dict[str, float] = {}
        for sample in result:
            if not isinstance(sample, dict):
                raise ValueError("Prometheus query returned an invalid sample")

            metric = sample.get("metric")
            if not isinstance(metric, dict):
                raise ValueError("Prometheus query returned an invalid metric descriptor")

            series_name = metric.get("__name__")
            if series_name not in self._EXPECTED_SERIES:
                raise ValueError(f"Prometheus query returned unexpected series {series_name!r}")

            field_name, service_name = self._EXPECTED_SERIES[series_name]
            if metric.get("service") != service_name:
                raise ValueError(f"Prometheus query returned unexpected labels for {series_name!r}")
            if field_name in parsed_values:
                raise ValueError(f"Prometheus query returned duplicate series {series_name!r}")

            value = sample.get("value")
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Prometheus query returned an invalid value tuple")

            numeric = float(value[1])
            if not math.isfinite(numeric):
                raise ValueError("Prometheus query returned a non-finite value")
            if numeric < 0:
                raise ValueError("Prometheus query returned a negative routing metric")

            parsed_values[field_name] = numeric

        missing = {
            field_name
            for field_name, _ in self._EXPECTED_SERIES.values()
            if field_name not in parsed_values
        }
        if missing:
            raise ValueError(f"Prometheus query omitted required routing metrics: {sorted(missing)!r}")

        return RoutingMetricsSnapshot(
            rho_nmt=parsed_values["rho_nmt"],
            rho_llm=parsed_values["rho_llm"],
            backlog_drain_nmt_seconds=parsed_values["backlog_drain_nmt_seconds"],
            backlog_drain_llm_seconds=parsed_values["backlog_drain_llm_seconds"],
            kingman_penalty_nmt_seconds=parsed_values["kingman_penalty_nmt_seconds"],
            kingman_penalty_llm_seconds=parsed_values["kingman_penalty_llm_seconds"],
        )
