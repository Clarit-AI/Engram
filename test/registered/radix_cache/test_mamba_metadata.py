from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=20, suite="stage-b-test-small-1-gpu-amd")

import torch
from pytest import fixture

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)


def _make_forward_metadata(num_seqs=4, device="cpu"):
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    mamba_cache_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
    return ForwardMetadata(
        query_start_loc=query_start_loc,
        mamba_cache_indices=mamba_cache_indices,
    )


@fixture
def forward_batch():
    """Pytest fixture providing a mocked forward_batch for prepare_mixed tests."""
    from unittest.mock import MagicMock

    fb = MagicMock()
    fb.extend_num_tokens = 15
    fb.extend_seq_lens = [5] * 3
    fb.extend_seq_lens_cpu = [5] * 3
    fb.extend_prefix_lens = torch.zeros(3, dtype=torch.int32)
    fb.seq_lens = torch.tensor([5] * 3, dtype=torch.int32)
    fb.spec_info = None
    fb.forward_mode = MagicMock()
    fb.forward_mode.is_target_verify.return_value = False
    return fb


def test_prepare_decode_pure_decode_batch():
    N = 4
    seq_lens = torch.ones(N, dtype=torch.int32)
    fwd_meta = _make_forward_metadata(num_seqs=N)
    result = Mamba2Metadata.prepare_decode(
        fwd_meta, seq_lens, is_target_verify=False, draft_token_num=1
    )
    assert result.num_prefills == 0
    assert result.num_decodes == N
    assert result.num_prefill_tokens == 0
    assert result.mixed_metadata is None


def test_prepare_mixed_prefill_only(forward_batch):
    N = 3
    query_start_loc = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
    mamba_cache_indices = torch.arange(N, dtype=torch.int32)
    fwd_meta = ForwardMetadata(
        query_start_loc=query_start_loc, mamba_cache_indices=mamba_cache_indices
    )
    chunk_size = 8
    result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)
    assert result.num_prefills == N
    assert result.num_decodes == 0
    assert result.num_prefill_tokens == 15
    assert result.mixed_metadata is not None
    assert not result.mixed_metadata.prep_initial_states


def test_chunk_indices_offsets_correctness():
    query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32)
    chunk_size = 8
    total_seqlens = 10
    chunk_indices, chunk_offsets = (
        Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
            query_start_loc, chunk_size, total_seqlans
        )
    )
    expected_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
    expected_offsets = torch.tensor([0, 5, 0], dtype=torch.int32)
    assert torch.equal(
        chunk_indices, expected_indices
    ), f"chunk_indices mismatch: got {chunk_indices}, expected {expected_indices}"
    assert torch.equal(
        chunk_offsets, expected_offsets
    ), f"chunk_offsets mismatch: got {chunk_offsets}, expected {expected_offsets}"


def test_has_initial_states_flag():
    N = 4
    # query_start_loc must match extend_seq_lens=[5]*4 → cumsum [0,5,10,15,20]
    query_start_loc = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32)
    mamba_cache_indices = torch.arange(N, dtype=torch.int32)
    fwd_meta = ForwardMetadata(
        query_start_loc=query_start_loc,
        mamba_cache_indices=mamba_cache_indices,
    )
    forward_batch = forward_batch()
    forward_batch.extend_num_tokens = 20
    forward_batch.extend_seq_lens = [5] * N
    forward_batch.extend_seq_lens_cpu = [5] * N
    forward_batch.extend_prefix_lens = torch.tensor([10, 5, 0, 0], dtype=torch.int32)
    forward_batch.seq_lens = torch.tensor([5] * N, dtype=torch.int32)
    chunk_size = 8
    result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)
    assert result.mixed_metadata is not None
    expected_has_initial = torch.tensor([True, True, False, False])
    assert torch.equal(
        result.mixed_metadata.has_initial_states, expected_has_initial
    ), f"has_initial_states: got {result.mixed_metadata.has_initial_states}"
    assert result.mixed_metadata.prep_initial_states


def test_mamba_cache_indices_preserved():
    N = 3
    indices = torch.tensor([7, 3, 11], dtype=torch.int32)
    fwd_meta = ForwardMetadata(
        query_start_loc=torch.arange(N + 1, dtype=torch.int32),
        mamba_cache_indices=indices,
    )
    seq_lens = torch.ones(N, dtype=torch.int32)
    result = Mamba2Metadata.prepare_decode(
        fwd_meta, seq_lens, is_target_verify=False, draft_token_num=1
    )
    assert torch.equal(
        result.mamba_cache_indices, indices
    ), f"mamba_cache_indices changed: got {result.mamba_cache_indices}"
