import json
import logging
import time
import os
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..prometheus_routing import RoutingMetricsSnapshot

logger = logging.getLogger(__name__)
class NoneRouterDecisionException(Exception):
    """Raised when router fails to make a decision."""
    pass

@dataclass
class InternalDecision:
    use_llm: bool
    p_llm: float
    threshold: float
    mode: str
    queue_depth: Optional[int]
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class HysteresisRouter:

    def __init__(
        self,
        config_path: str,
        model_path: str,
        *,
        p_llm_threshold: float = 0.45,
        overload_enter_rho: float = 0.8,
        overload_exit_rho: float = 0.75,
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.model = joblib.load(model_path)
        self.feature_names = self.config['features']

        # Thresholds for each mode
        self.thresholds = {
            'low': 0.30,
            'normal': 0.45,
            'high': 0.60,
            'critical': 0.75
        }

        self.boundaries = {
            'low_to_normal': 20,
            'normal_to_low': 10,
            'normal_to_high': 80,
            'high_to_normal': 60,
            'high_to_critical': 150,
            'critical_to_high': 120
        }

        self.current_mode = 'normal'
        self.routing_mode = 'healthy'
        self.p_llm_threshold = float(p_llm_threshold)
        self.overload_enter_rho = max(float(overload_enter_rho), 0.0)
        self.overload_exit_rho = max(float(overload_exit_rho), 0.0)
        if self.overload_exit_rho > self.overload_enter_rho:
            self.overload_exit_rho = self.overload_enter_rho
        self.mode_history: List[Dict] = []

        self.stats = {
            'total_requests': 0,
            'llm_requests': 0,
            'nmt_requests': 0,
            'mode_changes': 0
        }

        logger.info(f"HysteresisRouter initialized. Mode: {self.current_mode}")

    def update_mode(self, queue_depth: int) -> str:
        prev_mode = self.current_mode

        if self.current_mode == 'low':
            if queue_depth > self.boundaries['low_to_normal']:
                self.current_mode = 'normal'
        elif self.current_mode == 'normal':
            if queue_depth < self.boundaries['normal_to_low']:
                self.current_mode = 'low'
            elif queue_depth > self.boundaries['normal_to_high']:
                self.current_mode = 'high'
        elif self.current_mode == 'high':
            if queue_depth < self.boundaries['high_to_normal']:
                self.current_mode = 'normal'
            elif queue_depth > self.boundaries['high_to_critical']:
                self.current_mode = 'critical'
        elif self.current_mode == 'critical':
            if queue_depth < self.boundaries['critical_to_high']:
                self.current_mode = 'high'

        if prev_mode != self.current_mode:
            self.mode_history.append({
                'timestamp': time.time(),
                'from_mode': prev_mode,
                'to_mode': self.current_mode,
                'queue_depth': queue_depth
            })
            self.stats['mode_changes'] += 1
            logger.warning(f"Mode changed: {prev_mode} -> {self.current_mode} (queue: {queue_depth})")

        return self.current_mode

    def get_threshold(self, queue_depth: int) -> float:
        self.update_mode(queue_depth)
        return self.thresholds[self.current_mode]

    def update_routing_mode(self, routing_snapshot: RoutingMetricsSnapshot) -> str:
        prev_mode = self.routing_mode
        system_rho = routing_snapshot.system_rho

        if self.routing_mode == 'healthy':
            if system_rho >= self.overload_enter_rho:
                self.routing_mode = 'overloaded'
        elif self.routing_mode == 'overloaded':
            if system_rho <= self.overload_exit_rho:
                self.routing_mode = 'healthy'

        if prev_mode != self.routing_mode:
            self.mode_history.append({
                'timestamp': time.time(),
                'from_mode': prev_mode,
                'to_mode': self.routing_mode,
                'system_rho': system_rho,
                'rho_nmt': routing_snapshot.rho_nmt,
                'rho_llm': routing_snapshot.rho_llm,
                'source': 'prometheus',
            })
            self.stats['mode_changes'] += 1
            logger.warning(
                "Routing mode changed: %s -> %s (system_rho=%.3f)",
                prev_mode,
                self.routing_mode,
                system_rho,
            )

        return self.routing_mode

    def should_use_llm(
        self,
        features: Dict[str, float],
        queue_depth: int,
        routing_snapshot: Optional[RoutingMetricsSnapshot] = None,
    ) -> InternalDecision:
        X = np.array([[features.get(name, 0) for name in self.feature_names]])

        p_llm = self.model.predict_proba(X)[0, 1]

        if routing_snapshot is not None:
            mode = self.update_routing_mode(routing_snapshot)
            threshold = self.p_llm_threshold
            if mode == 'healthy':
                use_llm = p_llm >= threshold
            else:
                use_llm = routing_snapshot.score_llm <= routing_snapshot.score_nmt
        else:
            threshold = self.get_threshold(queue_depth)
            use_llm = p_llm > threshold
            mode = self.current_mode

        self.stats['total_requests'] += 1
        if use_llm:
            self.stats['llm_requests'] += 1
        else:
            self.stats['nmt_requests'] += 1

        return InternalDecision(
            use_llm=use_llm,
            p_llm=p_llm,
            threshold=threshold,
            mode=mode,
            queue_depth=queue_depth,
            features=features
        )

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['total_requests']
        return {
            'total_requests': total,
            'llm_requests': self.stats['llm_requests'],
            'nmt_requests': self.stats['nmt_requests'],
            'llm_ratio': self.stats['llm_requests'] / total if total > 0 else 0,
            'mode_changes': self.stats['mode_changes'],
            'current_mode': self.current_mode,
            'routing_mode': self.routing_mode,
        }


_router: Optional[HysteresisRouter] = None


def get_router(
    models_dir: str = None,
    *,
    p_llm_threshold: float = 0.45,
    overload_enter_rho: float = 0.8,
    overload_exit_rho: float = 0.75,
) -> HysteresisRouter:
    global _router

    if _router is None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')

        config_path = os.path.join(models_dir, 'router_config_xgb.json')
        model_path = os.path.join(models_dir, 'router_classifier_xgb.joblib')

        _router = HysteresisRouter(
            config_path,
            model_path,
            p_llm_threshold=p_llm_threshold,
            overload_enter_rho=overload_enter_rho,
            overload_exit_rho=overload_exit_rho,
        )

    return _router
