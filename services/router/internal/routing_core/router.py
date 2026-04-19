import json
import logging
import time
import os
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

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
    queue_depth: int
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class HysteresisRouter:

    def __init__(self, config_path: str, model_path: str):
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
            logger.warning(f"Mode changed: {prev_mode} → {self.current_mode} (queue: {queue_depth})")

        return self.current_mode

    def get_threshold(self, queue_depth: int) -> float:
        self.update_mode(queue_depth)
        return self.thresholds[self.current_mode]

    def should_use_llm(self, features: Dict[str, float], queue_depth: int) -> InternalDecision:
        X = np.array([[features.get(name, 0) for name in self.feature_names]])

        p_llm = self.model.predict_proba(X)[0, 1]

        threshold = self.get_threshold(queue_depth)

        use_llm = p_llm > threshold

        self.stats['total_requests'] += 1
        if use_llm:
            self.stats['llm_requests'] += 1
        else:
            self.stats['nmt_requests'] += 1

        return InternalDecision(
            use_llm=use_llm,
            p_llm=p_llm,
            threshold=threshold,
            mode=self.current_mode,
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
            'current_mode': self.current_mode
        }


_router: Optional[HysteresisRouter] = None


def get_router(models_dir: str = None) -> HysteresisRouter:
    global _router

    if _router is None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')

        config_path = os.path.join(models_dir, 'router_config_xgb.json')
        model_path = os.path.join(models_dir, 'router_classifier_xgb.joblib')

        _router = HysteresisRouter(config_path, model_path)

    return _router