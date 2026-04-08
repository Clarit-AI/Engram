"""Tests for SSM state tensor validation (hidden state poisoning guards)."""

import tempfile
from pathlib import Path

import pytest
import torch

from sglang.srt.snapshot.mamba_snapshot import (
    MambaSnapshotManager,
    MambaSnapshotMetadata,
    SnapshotValidationError,
    ValidationResult,
    validate_state_tensors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv_states(dtype=torch.float32) -> list:
    """Return a realistic conv_states list: one tensor, shape [24, 4, 16, 3]."""
    return [torch.rand(24, 4, 16, 3, dtype=dtype)]


def _make_temporal_states(dtype=torch.float32) -> torch.Tensor:
    """Return a realistic temporal_states tensor, shape [24, 4, 16, 64]."""
    return torch.rand(24, 4, 16, 64, dtype=dtype)


def _make_metadata(conversation_id: str = "test-conv-123") -> MambaSnapshotMetadata:
    return MambaSnapshotMetadata(
        conversation_id=conversation_id,
        turn_number=0,
        timestamp=0.0,
        token_count=10,
        model_name="test-model",
        mamba_pool_idx=1,
        req_pool_idx=1,
        layer_config={"num_layers": 24},
    )


# ---------------------------------------------------------------------------
# 1. Valid state passes
# ---------------------------------------------------------------------------


def test_valid_state_passes():
    conv = _make_conv_states()
    temporal = _make_temporal_states()
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []


# ---------------------------------------------------------------------------
# 2. NaN detection
# ---------------------------------------------------------------------------


def test_nan_detection():
    conv = _make_conv_states()
    conv[0][0, 0, 0, 0] = float("nan")
    temporal = _make_temporal_states()
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is False
    assert any(
        "NaN" in e for e in result.errors
    ), f"Expected 'NaN' in errors: {result.errors}"


# ---------------------------------------------------------------------------
# 3. Inf detection
# ---------------------------------------------------------------------------


def test_inf_detection():
    conv = _make_conv_states()
    temporal = _make_temporal_states()
    temporal[0, 0, 0, 0] = float("inf")
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is False
    assert any(
        "Inf" in e for e in result.errors
    ), f"Expected 'Inf' in errors: {result.errors}"


# ---------------------------------------------------------------------------
# 4. All-zeros warning (non-strict)
# ---------------------------------------------------------------------------


def test_all_zeros_warning():
    conv = [torch.zeros(24, 4, 16, 3, dtype=torch.float32)]
    temporal = torch.zeros(24, 4, 16, 64, dtype=torch.float32)
    result = validate_state_tensors(conv, temporal, strict=False)
    assert result.is_valid is True
    assert any(
        "all-zero" in w for w in result.warnings
    ), f"Expected 'all-zero' warning: {result.warnings}"


# ---------------------------------------------------------------------------
# 5. All-zeros strict → error
# ---------------------------------------------------------------------------


def test_all_zeros_strict_error():
    conv = [torch.zeros(24, 4, 16, 3, dtype=torch.float32)]
    temporal = torch.zeros(24, 4, 16, 64, dtype=torch.float32)
    result = validate_state_tensors(conv, temporal, strict=True)
    assert result.is_valid is False
    assert any(
        "all-zero" in e for e in result.errors
    ), f"Expected 'all-zero' error: {result.errors}"


# ---------------------------------------------------------------------------
# 6. Wrong dtype → error mentioning "dtype"
# ---------------------------------------------------------------------------


def test_wrong_dtype_error():
    conv = [torch.zeros(24, 4, 16, 3, dtype=torch.int64)]
    temporal = torch.zeros(24, 4, 16, 64, dtype=torch.int64)
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is False
    assert any(
        "dtype" in e for e in result.errors
    ), f"Expected 'dtype' in errors: {result.errors}"


# ---------------------------------------------------------------------------
# 7. Empty tensor (zero numel) → error mentioning "empty"
# ---------------------------------------------------------------------------


def test_empty_tensor_error():
    conv = [torch.empty(0)]
    temporal = _make_temporal_states()
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is False
    assert any(
        "empty" in e for e in result.errors
    ), f"Expected 'empty' in errors: {result.errors}"


# ---------------------------------------------------------------------------
# 8. Empty conv_states list → error mentioning "empty"
# ---------------------------------------------------------------------------


def test_empty_conv_states_error():
    temporal = _make_temporal_states()
    result = validate_state_tensors([], temporal)
    assert result.is_valid is False
    assert any(
        "empty" in e for e in result.errors
    ), f"Expected 'empty' in errors: {result.errors}"


# ---------------------------------------------------------------------------
# 9. bfloat16 accepted
# ---------------------------------------------------------------------------


def test_bfloat16_accepted():
    conv = _make_conv_states(dtype=torch.bfloat16)
    temporal = _make_temporal_states(dtype=torch.bfloat16)
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is True
    assert result.errors == []


# ---------------------------------------------------------------------------
# 10. float16 accepted
# ---------------------------------------------------------------------------


def test_float16_accepted():
    conv = _make_conv_states(dtype=torch.float16)
    temporal = _make_temporal_states(dtype=torch.float16)
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is True
    assert result.errors == []


# ---------------------------------------------------------------------------
# 11. conversation_id appears in error/warning messages
# ---------------------------------------------------------------------------


def test_metadata_conversation_id_in_messages():
    conv_id = "test-conv-123"
    metadata = _make_metadata(conversation_id=conv_id)

    # Trigger a NaN error so we have something to inspect
    conv = _make_conv_states()
    conv[0][0, 0, 0, 0] = float("nan")
    temporal = _make_temporal_states()

    result = validate_state_tensors(conv, temporal, metadata=metadata)
    assert result.is_valid is False
    assert any(
        conv_id in e for e in result.errors
    ), f"Expected '{conv_id}' in errors: {result.errors}"

    # Also verify warnings carry the conversation_id (use all-zeros warning)
    conv_zeros = [torch.zeros(24, 4, 16, 3)]
    temporal_zeros = torch.zeros(24, 4, 16, 64)
    result_warn = validate_state_tensors(conv_zeros, temporal_zeros, metadata=metadata)
    assert any(
        conv_id in w for w in result_warn.warnings
    ), f"Expected '{conv_id}' in warnings: {result_warn.warnings}"


# ---------------------------------------------------------------------------
# 12. Multiple errors reported (NaN in both conv and temporal)
# ---------------------------------------------------------------------------


def test_multiple_errors_reported():
    conv = _make_conv_states()
    conv[0][0, 0, 0, 0] = float("nan")
    temporal = _make_temporal_states()
    temporal[0, 0, 0, 0] = float("nan")
    result = validate_state_tensors(conv, temporal)
    assert result.is_valid is False
    # Both conv_state[0] and temporal_states should report NaN
    nan_errors = [e for e in result.errors if "NaN" in e]
    assert len(nan_errors) >= 2, f"Expected at least 2 NaN errors, got: {result.errors}"


# ---------------------------------------------------------------------------
# 13. Integration: save_snapshot rejects NaN state
# ---------------------------------------------------------------------------


def test_save_rejects_nan_state():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = MambaSnapshotManager(Path(tmp_dir))
        metadata = MambaSnapshotMetadata(
            conversation_id="test-conv",
            turn_number=0,
            timestamp=0.0,
            token_count=10,
            model_name="test-model",
            mamba_pool_idx=1,
            req_pool_idx=1,
            layer_config={"num_layers": 24},
        )
        conv = _make_conv_states()
        conv[0][0, 0, 0, 0] = float("nan")
        temporal = _make_temporal_states()

        with pytest.raises(ValueError):
            manager.save_snapshot(conv, temporal, metadata)


# ---------------------------------------------------------------------------
# 14. ValidationResult __bool__ behaviour
# ---------------------------------------------------------------------------


def test_validation_result_bool():
    valid = ValidationResult(is_valid=True, warnings=[], errors=[])
    invalid = ValidationResult(is_valid=False, warnings=[], errors=["something wrong"])
    assert bool(valid) is True
    assert bool(invalid) is False


# ---------------------------------------------------------------------------
# 15. E2E: save → load → restore round-trip and quarantine on corruption
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_and_quarantine():
    """Save a valid snapshot, load it back, and verify quarantine on corruption."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = MambaSnapshotManager(Path(tmp_dir))
        metadata = _make_metadata(conversation_id="roundtrip-conv")

        conv = _make_conv_states()
        temporal = _make_temporal_states()

        # Save should succeed
        manager.save_snapshot(conv, temporal, metadata)

        # Load should succeed and return valid state
        loaded = manager.load_snapshot("roundtrip-conv", turn_number=0)
        assert loaded is not None
        loaded_conv, loaded_temporal, loaded_meta = loaded
        assert len(loaded_conv) == len(conv)
        assert loaded_temporal.shape == temporal.shape
        assert loaded_meta.conversation_id == "roundtrip-conv"

        # Verify loaded state passes validation
        result = validate_state_tensors(loaded_conv, loaded_temporal)
        assert result.is_valid is True

        # Now corrupt the file on disk and verify load raises ValueError
        conv_dir = Path(tmp_dir) / "roundtrip-conv"
        snapshot_files = list(conv_dir.glob("*.safetensors"))
        assert len(snapshot_files) > 0
        # Overwrite with garbage to trigger deserialization or validation failure
        for f in snapshot_files:
            f.write_bytes(b"corrupted data")

        with pytest.raises(SnapshotValidationError):
            manager.load_snapshot("roundtrip-conv", turn_number=0)
