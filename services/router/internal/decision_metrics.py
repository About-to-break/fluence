import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional


logger = logging.getLogger(__name__)

_ALLOWED_PATHS = ("fast", "heavy")
_ALLOWED_MODES = ("low", "normal", "high", "critical")
_SUMMARY_METRICS = (
    ("p_llm", "router_decision_p_llm", "Exact p_llm observations"),
    ("threshold", "router_decision_threshold", "Exact threshold observations"),
    ("decision_margin", "router_decision_margin", "Exact decision margin observations"),
    ("queue_depth_input", "router_decision_queue_depth_input", "Exact queue-depth inputs used for decisions"),
    ("source_length_words", "router_decision_source_length_words", "Exact source_length_words feature observations"),
    ("log_perplexity", "router_decision_log_perplexity", "Exact log_perplexity feature observations"),
    ("perplexity_per_word", "router_decision_perplexity_per_word", "Exact perplexity_per_word feature observations"),
)


def _coerce_numeric(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None

    numeric = float(value)
    if not math.isfinite(numeric):
        return None

    return numeric


class SafeDecisionMetricsCollector:
    """Thread-safe companion exporter for exact router decision metrics."""

    def __init__(self):
        self._metrics_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._server: Optional[HTTPServer] = None
        self._port = 9093
        self._lock = threading.Lock()

        self._decision_totals = {
            (path, mode): 0 for path in _ALLOWED_PATHS for mode in _ALLOWED_MODES
        }
        self._summary_totals = {
            metric_key: {
                (path, mode): {"sum": 0.0, "count": 0}
                for path in _ALLOWED_PATHS
                for mode in _ALLOWED_MODES
            }
            for metric_key, _, _ in _SUMMARY_METRICS
        }

    def start(self, port: int = 9093):
        if self._metrics_thread and self._metrics_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._port = port

        collector = self

        class DecisionMetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/metrics/model":
                        metrics_text = collector._render_metrics()
                        self._send_response(200, metrics_text)
                    elif self.path == "/health":
                        self._send_response(200, "OK", content_type="text/plain")
                    else:
                        self._send_response(404, "Not Found", content_type="text/plain")
                except Exception as exc:
                    logger.error("Error in %s: %s", self.path, exc, exc_info=True)
                    self._send_response(500, f"Internal Error: {exc}", content_type="text/plain")

            def _send_response(self, code: int, body: str, content_type: str = "text/plain; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, format, *args):
                pass

        self._server = HTTPServer(("0.0.0.0", self._port), DecisionMetricsHandler)
        self._server.timeout = 1.0
        self._metrics_thread = threading.Thread(
            target=self._run_metrics_server,
            daemon=True,
            name="RouterDecisionMetricsHTTP",
        )
        self._metrics_thread.start()
        logger.info("Decision metrics server started on port %s", port)

    def stop(self):
        self._shutdown_event.set()
        if self._metrics_thread:
            self._metrics_thread.join(timeout=5.0)
            logger.info("Decision metrics server stopped")
        self._metrics_thread = None
        self._server = None

    def record_decision(self, decision, path: str) -> None:
        if path not in _ALLOWED_PATHS:
            logger.warning("Skipping decision metrics for unexpected path %r", path)
            return

        mode = getattr(decision, "mode", None)
        if mode not in _ALLOWED_MODES:
            logger.warning("Skipping decision metrics for unexpected mode %r", mode)
            return

        features = getattr(decision, "features", {}) or {}
        p_llm = _coerce_numeric(getattr(decision, "p_llm", None))
        threshold = _coerce_numeric(getattr(decision, "threshold", None))

        values = {
            "p_llm": p_llm,
            "threshold": threshold,
            "decision_margin": None if p_llm is None or threshold is None else p_llm - threshold,
            "queue_depth_input": _coerce_numeric(getattr(decision, "queue_depth", None)),
            "source_length_words": _coerce_numeric(features.get("source_length_words")),
            "log_perplexity": _coerce_numeric(features.get("log_perplexity")),
            "perplexity_per_word": _coerce_numeric(features.get("perplexity_per_word")),
        }

        with self._lock:
            self._decision_totals[(path, mode)] += 1

            for metric_key, metric_value in values.items():
                if metric_value is None:
                    continue

                slot = self._summary_totals[metric_key][(path, mode)]
                slot["sum"] += metric_value
                slot["count"] += 1

    def _run_metrics_server(self):
        assert self._server is not None

        try:
            while not self._shutdown_event.is_set():
                self._server.handle_request()
        finally:
            self._server.server_close()

    def _render_metrics(self) -> str:
        with self._lock:
            decision_totals = self._decision_totals.copy()
            summary_totals = {
                metric_key: {
                    label_pair: values.copy()
                    for label_pair, values in label_totals.items()
                }
                for metric_key, label_totals in self._summary_totals.items()
            }

        lines = [
            "# HELP router_decision_total Exact router decisions by semantic path and hysteresis mode",
            "# TYPE router_decision_total counter",
        ]
        for path in _ALLOWED_PATHS:
            for mode in _ALLOWED_MODES:
                lines.append(
                    f'router_decision_total{{path="{path}",mode="{mode}"}} {decision_totals[(path, mode)]}'
                )

        for metric_key, metric_name, metric_help in _SUMMARY_METRICS:
            lines.extend(
                [
                    "",
                    f"# HELP {metric_name}_sum {metric_help}",
                    f"# TYPE {metric_name}_sum counter",
                ]
            )
            for path in _ALLOWED_PATHS:
                for mode in _ALLOWED_MODES:
                    metric_values = summary_totals[metric_key][(path, mode)]
                    lines.append(
                        f'{metric_name}_sum{{path="{path}",mode="{mode}"}} {metric_values["sum"]:.12g}'
                    )

            lines.extend(
                [
                    "",
                    f"# HELP {metric_name}_count Count of {metric_help.lower()}",
                    f"# TYPE {metric_name}_count counter",
                ]
            )
            for path in _ALLOWED_PATHS:
                for mode in _ALLOWED_MODES:
                    metric_values = summary_totals[metric_key][(path, mode)]
                    lines.append(
                        f'{metric_name}_count{{path="{path}",mode="{mode}"}} {metric_values["count"]}'
                    )

        lines.append("")
        return "\n".join(lines)


_collector: Optional[SafeDecisionMetricsCollector] = None
_initialized = False


def init_decision_metrics() -> SafeDecisionMetricsCollector:
    global _collector, _initialized

    if not _initialized:
        _collector = SafeDecisionMetricsCollector()
        _initialized = True

    return _collector


def get_decision_metrics() -> Optional[SafeDecisionMetricsCollector]:
    return _collector


def reset_decision_metrics():
    global _collector, _initialized

    if _collector is not None:
        _collector.stop()

    _collector = None
    _initialized = False
