"""Tier 2 state health monitoring for Mamba SSM snapshot persistence.

Tracks tensor norm baselines per conversation and detects anomalous drift
that structural validation (Tier 1) cannot catch. A state can pass all
NaN/Inf/dtype checks and still be poisoned — norms drifting far from
baseline indicates corruption that only statistical monitoring reveals.
"""

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.snapshot.mamba_snapshot import validate_state_tensors

logger = logging.getLogger("sglang.snapshot.state_health")

DEFAULT_SIGMA_THRESHOLD = 3.0
DEFAULT_WINDOW_SIZE = 100


class NormBaseline:
    """Rolling window of tensor norms for a single layer.

    Maintains a fixed-size deque of recent norm values and computes
    mean/std for anomaly detection via sigma threshold.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self._windows: Dict[int, collections.deque] = {}
        self._window_size = window_size

    def update(self, layer_idx: int, norm_value: float) -> None:
        if layer_idx not in self._windows:
            self._windows[layer_idx] = collections.deque(
                maxlen=self._window_size
            )
        self._windows[layer_idx].append(norm_value)

    def is_anomalous(
        self,
        layer_idx: int,
        norm_value: float,
        sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
    ) -> bool:
        if layer_idx not in self._windows:
            return False
        window = self._windows[layer_idx]
        if len(window) < 2:
            return False
        mean, std, _ = self.get_stats(layer_idx)
        if std == 0:
            return norm_value != mean
        return abs(norm_value - mean) > sigma_threshold * std

    def get_stats(self, layer_idx: int) -> Tuple[float, float, int]:
        """Return (mean, std, count) for a layer. Returns (0, 0, 0) if no data."""
        if layer_idx not in self._windows or len(self._windows[layer_idx]) == 0:
            return (0.0, 0.0, 0)
        window = self._windows[layer_idx]
        count = len(window)
        mean = sum(window) / count
        if count < 2:
            return (mean, 0.0, count)
        variance = sum((x - mean) ** 2 for x in window) / (count - 1)
        std = math.sqrt(variance)
        return (mean, std, count)

    def clear(self) -> None:
        self._windows.clear()


@dataclass
class HealthCheckResult:
    """Result of a state health check for a single conversation."""

    healthy: bool
    anomalous_layers: List[int] = field(default_factory=list)
    details: Dict[int, dict] = field(default_factory=dict)
    timestamp: float = 0.0
    conversation_id: str = ""
    turn_number: int = 0

    def __bool__(self) -> bool:
        return self.healthy


class StateHealthMonitor:
    """Monitors tensor norm baselines per conversation to detect drift.

    One instance per scheduler. Tracks per-layer norms in a rolling window
    and flags anomalies when norms deviate beyond a sigma threshold.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
    ):
        self._baselines: Dict[str, NormBaseline] = {}
        self._window_size = window_size
        self._sigma_threshold = sigma_threshold

    def _get_baseline(self, conversation_id: str) -> NormBaseline:
        if conversation_id not in self._baselines:
            self._baselines[conversation_id] = NormBaseline(self._window_size)
        return self._baselines[conversation_id]

    def check_state_health(
        self,
        conversation_id: str,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        turn_number: int,
    ) -> HealthCheckResult:
        """Check state health via structural validation then norm analysis.

        Structural validation (Tier 1) runs first. If it fails, returns
        unhealthy immediately without norm analysis.
        """
        now = time.time()

        # Structural validation first (Tier 1)
        validation = validate_state_tensors(conv_states, temporal_states)
        if not validation.is_valid:
            return HealthCheckResult(
                healthy=False,
                anomalous_layers=[],
                details={"structural_errors": validation.errors},
                timestamp=now,
                conversation_id=conversation_id,
                turn_number=turn_number,
            )

        baseline = self._get_baseline(conversation_id)
        anomalous_layers: List[int] = []
        details: Dict[int, dict] = {}

        # Check conv_states (one tensor per layer group)
        for layer_idx, tensor in enumerate(conv_states):
            norm = torch.linalg.norm(tensor.float()).item()
            is_anomalous = baseline.is_anomalous(
                layer_idx, norm, self._sigma_threshold
            )
            mean, std, count = baseline.get_stats(layer_idx)
            baseline.update(layer_idx, norm)

            if is_anomalous:
                anomalous_layers.append(layer_idx)
                sigma_dev = abs(norm - mean) / std if std > 0 else float("inf")
                details[layer_idx] = {
                    "norm": norm,
                    "mean": mean,
                    "std": std,
                    "sigma_deviation": sigma_dev,
                    "source": "conv_state",
                }

        # Check temporal_states as a single aggregate
        temporal_layer_idx = len(conv_states)
        temporal_norm = torch.linalg.norm(temporal_states.float()).item()
        is_temporal_anomalous = baseline.is_anomalous(
            temporal_layer_idx, temporal_norm, self._sigma_threshold
        )
        t_mean, t_std, t_count = baseline.get_stats(temporal_layer_idx)
        baseline.update(temporal_layer_idx, temporal_norm)

        if is_temporal_anomalous:
            anomalous_layers.append(temporal_layer_idx)
            sigma_dev = (
                abs(temporal_norm - t_mean) / t_std
                if t_std > 0
                else float("inf")
            )
            details[temporal_layer_idx] = {
                "norm": temporal_norm,
                "mean": t_mean,
                "std": t_std,
                "sigma_deviation": sigma_dev,
                "source": "temporal_states",
            }

        return HealthCheckResult(
            healthy=len(anomalous_layers) == 0,
            anomalous_layers=anomalous_layers,
            details=details,
            timestamp=now,
            conversation_id=conversation_id,
            turn_number=turn_number,
        )

    def get_conversation_health(
        self, conversation_id: str
    ) -> Optional[NormBaseline]:
        """Return the baseline for a conversation, or None."""
        return self._baselines.get(conversation_id)

    def reset_baseline(self, conversation_id: str) -> None:
        """Clear rolling window for a conversation (e.g. after known-good restore)."""
        if conversation_id in self._baselines:
            self._baselines[conversation_id].clear()
