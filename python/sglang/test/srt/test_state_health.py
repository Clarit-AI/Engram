"""Tests for Tier 2 state health monitoring (norm baseline anomaly detection)."""

import time

import torch

from sglang.srt.snapshot.state_health import (
    HealthCheckResult,
    NormBaseline,
    StateHealthMonitor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv_states(dtype=torch.float32, scale=1.0) -> list:
    """Return a conv_states list: one tensor, shape [24, 4, 16, 3]."""
    return [torch.rand(24, 4, 16, 3, dtype=dtype) * scale]


def _make_temporal_states(dtype=torch.float32, scale=1.0) -> torch.Tensor:
    """Return a temporal_states tensor, shape [24, 4, 16, 64]."""
    return torch.rand(24, 4, 16, 64, dtype=dtype) * scale


# ---------------------------------------------------------------------------
# 1. NormBaseline rolling window
# ---------------------------------------------------------------------------


def test_norm_baseline_rolling_window():
    baseline = NormBaseline(window_size=5)
    for i in range(10):
        baseline.update(0, float(i))

    mean, std, count = baseline.get_stats(0)
    # Window should contain [5, 6, 7, 8, 9] (last 5 values)
    assert count == 5
    assert abs(mean - 7.0) < 1e-6  # mean of 5,6,7,8,9


# ---------------------------------------------------------------------------
# 2. NormBaseline anomaly detection
# ---------------------------------------------------------------------------


def test_norm_baseline_anomaly_detection():
    baseline = NormBaseline(window_size=100)
    # Build stable baseline around 1.0
    for _ in range(20):
        baseline.update(0, 1.0)

    # Normal value should not be anomalous
    assert baseline.is_anomalous(0, 1.0) is False

    # 10x spike should be anomalous
    assert baseline.is_anomalous(0, 10.0) is True


# ---------------------------------------------------------------------------
# 3. NormBaseline insufficient samples
# ---------------------------------------------------------------------------


def test_norm_baseline_insufficient_samples():
    baseline = NormBaseline(window_size=100)
    # Zero samples
    assert baseline.is_anomalous(0, 100.0) is False
    # One sample
    baseline.update(0, 1.0)
    assert baseline.is_anomalous(0, 100.0) is False


# ---------------------------------------------------------------------------
# 4. Health monitor healthy state
# ---------------------------------------------------------------------------


def test_health_monitor_healthy_state():
    monitor = StateHealthMonitor()
    conv = _make_conv_states()
    temporal = _make_temporal_states()

    # First check — no baseline yet, always healthy
    result = monitor.check_state_health("conv-1", conv, temporal, turn_number=1)
    assert result.healthy is True
    assert result.anomalous_layers == []
    assert result.conversation_id == "conv-1"


# ---------------------------------------------------------------------------
# 5. Health monitor single layer anomaly
# ---------------------------------------------------------------------------


def test_health_monitor_single_layer_anomaly():
    monitor = StateHealthMonitor(sigma_threshold=3.0)

    # Build baseline with consistent norms
    for i in range(20):
        conv = _make_conv_states(scale=1.0)
        temporal = _make_temporal_states(scale=1.0)
        monitor.check_state_health("conv-1", conv, temporal, turn_number=i)

    # Spike only layer 0 of conv_states
    conv_spike = _make_conv_states(scale=1.0)
    conv_spike[0][0] *= 100.0  # spike layer 0 only
    temporal_normal = _make_temporal_states(scale=1.0)
    result = monitor.check_state_health(
        "conv-1", conv_spike, temporal_normal, turn_number=20
    )

    assert result.healthy is False
    assert 0 in result.anomalous_layers  # layer 0 (conv_state) should be flagged
    assert 0 in result.details
    assert result.details[0]["source"] == "conv_state"


# ---------------------------------------------------------------------------
# 6. Health monitor baseline reset
# ---------------------------------------------------------------------------


def test_health_monitor_baseline_reset():
    monitor = StateHealthMonitor()

    # Build baseline
    for i in range(20):
        conv = _make_conv_states(scale=1.0)
        temporal = _make_temporal_states(scale=1.0)
        monitor.check_state_health("conv-1", conv, temporal, turn_number=i)

    # Reset baseline
    monitor.reset_baseline("conv-1")

    # After reset, even a spike should pass (no baseline to compare against)
    conv_spike = _make_conv_states(scale=100.0)
    temporal_spike = _make_temporal_states(scale=100.0)
    result = monitor.check_state_health(
        "conv-1", conv_spike, temporal_spike, turn_number=20
    )
    assert result.healthy is True


# ---------------------------------------------------------------------------
# 7. Health monitor conversation isolation
# ---------------------------------------------------------------------------


def test_health_monitor_conversation_isolation():
    monitor = StateHealthMonitor()

    # Build baseline for conv-1 with scale=1.0
    for i in range(20):
        conv = _make_conv_states(scale=1.0)
        temporal = _make_temporal_states(scale=1.0)
        monitor.check_state_health("conv-1", conv, temporal, turn_number=i)

    # conv-2 has no baseline, so a large value should pass
    conv_large = _make_conv_states(scale=100.0)
    temporal_large = _make_temporal_states(scale=100.0)
    result = monitor.check_state_health(
        "conv-2", conv_large, temporal_large, turn_number=0
    )
    assert result.healthy is True

    # But the same large value in conv-1 should be anomalous
    result2 = monitor.check_state_health(
        "conv-1", conv_large, temporal_large, turn_number=20
    )
    assert result2.healthy is False


# ---------------------------------------------------------------------------
# 8. HealthCheckResult fields
# ---------------------------------------------------------------------------


def test_health_check_result_fields():
    result = HealthCheckResult(
        healthy=True,
        anomalous_layers=[],
        details={},
        timestamp=time.time(),
        conversation_id="test-conv",
        turn_number=5,
    )
    assert bool(result) is True
    assert result.conversation_id == "test-conv"
    assert result.turn_number == 5

    result_bad = HealthCheckResult(
        healthy=False,
        anomalous_layers=[0, 1],
        details={0: {"norm": 10.0}},
        timestamp=time.time(),
        conversation_id="test-conv",
        turn_number=6,
    )
    assert bool(result_bad) is False
    assert len(result_bad.anomalous_layers) == 2


# ---------------------------------------------------------------------------
# 9. Structural validation failure (NaN) → unhealthy without norm analysis
# ---------------------------------------------------------------------------


def test_structural_validation_failure():
    monitor = StateHealthMonitor()
    conv = _make_conv_states()
    conv[0][0, 0, 0, 0] = float("nan")
    temporal = _make_temporal_states()

    result = monitor.check_state_health("conv-nan", conv, temporal, turn_number=1)
    assert result.healthy is False
    # Should have structural errors, not norm anomalies
    assert "structural_errors" in result.details
    assert result.anomalous_layers == []


# ---------------------------------------------------------------------------
# 10. Empty state handling
# ---------------------------------------------------------------------------


def test_empty_state_handling():
    monitor = StateHealthMonitor()
    # Empty conv_states list → structural validation catches it
    result = monitor.check_state_health("conv-empty", [], torch.zeros(1), turn_number=0)
    assert result.healthy is False
    assert "structural_errors" in result.details
