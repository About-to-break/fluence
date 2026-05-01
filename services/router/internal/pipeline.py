"""
Production pipeline for adaptive routing.
"""

import orjson
import time
import logging
import os
from types import SimpleNamespace

from . import decision_metrics
from .prometheus_queue_depth import PrometheusQueueDepthClient
from .prometheus_routing import PrometheusRoutingClient, RoutingMetricsSnapshot

# Импорты из routing_core
from .routing_core.features import FeatureExtractor
from .routing_core.router import get_router, HysteresisRouter

# Глобальные объекты (инициализируются один раз)
_extractor: FeatureExtractor = None
_router: HysteresisRouter = None
_queue_depth: int = 0
_prometheus_queue_depth_client: PrometheusQueueDepthClient | None = None
_prometheus_routing_client: PrometheusRoutingClient | None = None


class EmptyPayloadError(Exception):
    """Raised when message body is empty."""
    pass


class NoneRouterDecisionException(Exception):
    """Raised when router fails to make decision."""
    pass


def _initialize(config: SimpleNamespace = None):
    """Initialize feature extractor and router (called once)."""
    global _extractor, _router, _queue_depth, _prometheus_queue_depth_client, _prometheus_routing_client

    if _extractor is None:
        # Пути к моделям
        routing_core_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(routing_core_dir, 'routing_core', 'models')
        data_dir = os.path.join(routing_core_dir, 'routing_core', 'data')

        kenlm_path = os.path.join(data_dir, 'kenlm_wiki_en.bin')
        _extractor = FeatureExtractor(kenlm_path)
        _router = get_router(
            models_dir,
            p_llm_threshold=getattr(config, "ROUTER_P_LLM_THRESHOLD", 0.45),
            p_llm_penalty_alpha=getattr(config, "ROUTER_P_LLM_PENALTY_ALPHA", 0.2),
            overload_enter_rho=getattr(config, "ROUTER_OVERLOAD_ENTER_RHO", 0.8),
            overload_exit_rho=getattr(config, "ROUTER_OVERLOAD_EXIT_RHO", 0.75),
        )

        # Установить начальную глубину очереди из конфига
        if config and hasattr(config, 'QUEUE_DEPTH_STATIC'):
            _queue_depth = int(config.QUEUE_DEPTH_STATIC)
        if config and getattr(config, "PROMETHEUS_ENABLED", False):
            _prometheus_routing_client = PrometheusRoutingClient.from_config(config)
            _prometheus_queue_depth_client = PrometheusQueueDepthClient.from_config(config)

        logging.info(f"Pipeline initialized. Queue depth: {_queue_depth}")


def update_queue_depth(depth: int):
    """Update LLM queue depth (called by monitoring system)."""
    global _queue_depth
    _queue_depth = int(depth)
    if _router:
        _router.update_mode(_queue_depth)
        logging.debug(f"Queue depth updated: {_queue_depth}, mode: {_router.current_mode}")


def run_pipeline(message: bytes, option_fast: str = None, option_quality: str = None) -> str:
    global _queue_depth
    t_total = time.time()
    routing_snapshot: RoutingMetricsSnapshot | None = None

    # Parse message
    t_parse = time.time()
    try:
        body_str = message.decode('utf-8')
        data = orjson.loads(body_str)
        source_text = data.get('source', data.get('text', ''))
    except (UnicodeDecodeError, orjson.JSONDecodeError) as e:
        logging.error(f"Failed to parse message: {e}")
        raise EmptyPayloadError(f"Invalid message format: {e}")
    logging.debug(f"Parse: {(time.time() - t_parse) * 1000:.2f} ms")

    if not source_text:
        logging.warning("Empty source text")
        raise EmptyPayloadError("Source text is empty")

    if _prometheus_routing_client is not None:
        try:
            routing_snapshot = _prometheus_routing_client.get_routing_snapshot()
        except Exception as exc:
            logging.exception("Failed to resolve routing snapshot from Prometheus: %s", exc)

    if routing_snapshot is None and _prometheus_queue_depth_client is not None:
        try:
            queue_depth = _prometheus_queue_depth_client.get_llm_queue_depth()
        except Exception as exc:
            logging.exception("Failed to resolve queue depth from Prometheus: %s", exc)
        else:
            if queue_depth is not None:
                update_queue_depth(queue_depth)

    # Extract features
    t_features = time.time()
    try:
        features = _extractor.extract(source_text)
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        raise NoneRouterDecisionException(f"Feature extraction error: {e}")
    logging.debug(f"Features: {(time.time() - t_features) * 1000:.2f} ms")

    # Make routing decision
    t_router = time.time()
    try:
        decision = _router.should_use_llm(features, _queue_depth, routing_snapshot=routing_snapshot)
    except Exception as e:
        logging.error(f"Router decision failed: {e}")
        raise NoneRouterDecisionException(f"Router error: {e}")
    logging.debug(f"Router: {(time.time() - t_router) * 1000:.2f} ms")

    try:
        collector = decision_metrics.get_decision_metrics()
        if collector is not None:
            collector.record_decision(decision, "heavy" if decision.use_llm else "fast")
    except Exception as exc:
        logging.exception("Decision metrics emission failed: %s", exc)

    score_fast = getattr(decision, "score_fast", None)
    score_heavy = getattr(decision, "score_heavy", None)
    threshold = getattr(decision, "threshold", None)

    if score_fast is not None and score_heavy is not None:
        decision_detail = (
            f"score_fast={score_fast:.3f} | "
            f"score_heavy={score_heavy:.3f}"
        )
    elif threshold is not None:
        decision_detail = f"threshold={threshold:.2f}"
    else:
        decision_detail = "threshold=n/a"

    # Log decision
    logging.info(
        f"Routing: {'LLM' if decision.use_llm else 'NMT'} | "
        f"p_llm={decision.p_llm:.3f} | {decision_detail} | "
        f"mode={decision.mode} | queue={_queue_depth} | "
        f"text='{source_text[:50]}...'"
    )

    logging.debug(f"TOTAL: {(time.time() - t_total) * 1000:.2f} ms")

    # Return appropriate routing key
    if decision.use_llm:
        return option_quality or option_fast
    else:
        return option_fast or option_quality


def run_test_pipeline(message: bytes, option_fast: str = None, option_quality: str = None) -> str:
    """
    Test pipeline - always returns fast path for testing.
    """
    logging.info(f"Test pipeline: always routing to fast path")

    try:
        body_str = message.decode('utf-8')
        data = orjson.loads(body_str)
        source_text = data.get('source', data.get('text', ''))
        logging.info(f"Test message: {source_text[:100]}")
    except:
        pass

    return option_fast


def get_pipeline(config: SimpleNamespace):
    """Get pipeline function based on config."""
    # Инициализируем с конфигом
    _initialize(config)

    if config.PIPELINE == "prod":
        return run_pipeline
    elif config.PIPELINE == "test":
        return run_test_pipeline
    else:
        raise ValueError("Unsupported pipeline type")
