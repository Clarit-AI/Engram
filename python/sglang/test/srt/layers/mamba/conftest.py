"""
Pytest fixtures for Mamba layer tests.

This module provides common fixtures for testing Mamba SSM components.
"""

import pytest
import torch
from typing import Dict, Any

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig


@pytest.fixture
def server_args_fixture() -> ServerArgs:
    """
    Provide default ServerArgs for testing.

    Returns:
        ServerArgs configured for single-GPU testing
    """
    return ServerArgs(
        model_path="state-spaces/mamba-130m",
        device="cuda",
        tp_size=1,
        dp_size=1,
        pp_size=1,
        ep_size=1,
        mem_fraction_static=0.8,
        max_total_tokens=2048,
    )


@pytest.fixture
def mamba_model_config_fixture() -> ModelConfig:
    """
    Provide minimal ModelConfig for Mamba-130M testing.

    Returns:
        ModelConfig for small Mamba model
    """
    # TODO: Create proper ModelConfig when Mamba model is implemented
    # This is a placeholder
    return ModelConfig(
        model_path="state-spaces/mamba-130m",
        trust_remote_code=True,
    )


@pytest.fixture
def mock_mamba_state(server_args_fixture: ServerArgs) -> torch.Tensor:
    """
    Create mock Mamba state tensor for testing.

    Args:
        server_args_fixture: Server configuration fixture

    Returns:
        Random state tensor of shape [batch, num_layers, d_state, d_conv]
    """
    # Typical Mamba-130M dimensions
    batch_size = 1
    num_layers = 24
    d_state = 16
    d_conv = 4

    return torch.randn(
        batch_size, num_layers, d_state, d_conv,
        device=server_args_fixture.device,
        dtype=torch.float16,
    )


@pytest.fixture
def mock_token_ids() -> torch.Tensor:
    """
    Create mock token IDs for testing.

    Returns:
        Tensor of token IDs
    """
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """
    Provide test configuration constants.

    Returns:
        Dictionary of test configuration values
    """
    return {
        "batch_size": 4,
        "seq_len": 128,
        "d_model": 768,
        "d_state": 16,
        "d_conv": 4,
        "num_layers": 24,
        "vocab_size": 50280,
        "tolerance": 1e-3,  # Numerical tolerance for fp16
    }
