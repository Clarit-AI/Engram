# Phase 5 — Mamba2Metadata / ForwardMetadata Integrity

## Purpose

Verify that the metadata constructed during each forward pass is correct for decode-only, prefill-only, and mixed batches — without running a server. `Mamba2Metadata` carries `mamba_cache_indices`, `track_conv_indices`, chunk index/offset tensors, and the `has_initial_states` flag that controls whether chunked-prefill state initialization is triggered. Bugs here cause silent numerical errors in SSM state propagation that are hard to detect from outputs alone. These unit tests directly exercise `Mamba2Metadata.prepare_decode()`, `Mamba2Metadata.prepare_mixed()`, and the static chunk-index helper.

## Prerequisites

- Phase 0 complete
- Phase 2 complete (memory pool verified)
- No server required

## Key Files

- `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` — `ForwardMetadata`, `Mamba2Metadata`, `prepare_decode()`, `prepare_mixed()`, `_query_start_loc_to_chunk_indices_offsets()`
- `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch`, `ForwardMode`
- **New**: `test/registered/radix_cache/test_mamba_metadata.py`

## Environment Setup

```bash
cd /home/bbrenner/sglang-mamba
source test/phases/config.sh
pip install -e python/ --quiet

python -c "
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata, Mamba2Metadata
print('Mamba2Metadata import OK')
"
```

## Tasks

### Task 1: Write the metadata test file

Create `test/registered/radix_cache/test_mamba_metadata.py` with the content in **New Test File(s) to Write** below.

### Task 2: Run the tests

```bash
python -m pytest test/registered/radix_cache/test_mamba_metadata.py -v \
    2>&1 | tee /tmp/phase5_tests.log
```

### Task 3: Verify the docstring example

The docstring for `_query_start_loc_to_chunk_indices_offsets` in `mamba2_metadata.py` provides a worked example:
- `query_start_loc=[0,5,10]`, `chunk_size=8`, `total_seqlens=10`
- Expected: `chunk_indices=[0,0,1]`, `chunk_offsets=[0,5,0]`

`test_chunk_indices_offsets_correctness` must reproduce this exactly. If the test fails, inspect the method at `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` lines 68–150 to understand the expected behavior.

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_metadata.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=20, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=20, suite="stage-b-test-small-1-gpu-amd")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)


def _make_forward_metadata(num_seqs=4, device="cpu"):
    """Construct a minimal ForwardMetadata for testing."""
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    mamba_cache_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
    return ForwardMetadata(
        query_start_loc=query_start_loc,
        mamba_cache_indices=mamba_cache_indices,
    )


class TestMamba2Metadata(unittest.TestCase):

    def test_prepare_decode_pure_decode_batch(self):
        """prepare_decode: num_prefills=0, num_decodes=N, mixed_metadata=None."""
        N = 4
        seq_lens = torch.ones(N, dtype=torch.int32)
        fwd_meta = _make_forward_metadata(num_seqs=N)

        result = Mamba2Metadata.prepare_decode(
            fwd_meta, seq_lens, is_target_verify=False, draft_token_num=1
        )

        self.assertEqual(result.num_prefills, 0)
        self.assertEqual(result.num_decodes, N)
        self.assertEqual(result.num_prefill_tokens, 0)
        self.assertIsNone(result.mixed_metadata)

    def test_prepare_mixed_prefill_only(self):
        """prepare_mixed with extend requests: num_prefills=N, num_decodes=0, mixed_metadata populated."""
        N = 3
        device = "cpu"
        # query_start_loc must reflect actual cumulative token counts (5 tokens each).
        # arange(N+1) = [0,1,2,3] would mismatch output_size=15 in repeat_interleave.
        query_start_loc = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
        mamba_cache_indices = torch.arange(N, dtype=torch.int32)
        fwd_meta = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
        )

        # Build a mock ForwardBatch for N prefill requests (no decode)
        forward_batch = MagicMock()
        forward_batch.extend_num_tokens = 15          # total prefill tokens (N * 5)
        forward_batch.extend_seq_lens = [5] * N       # 5 tokens each
        forward_batch.extend_seq_lens_cpu = [5] * N
        forward_batch.extend_prefix_lens = torch.zeros(N, dtype=torch.int32)
        forward_batch.seq_lens = torch.tensor([5] * N, dtype=torch.int32)
        forward_batch.spec_info = None
        forward_batch.forward_mode = MagicMock()
        forward_batch.forward_mode.is_target_verify.return_value = False

        chunk_size = 8
        result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)

        self.assertEqual(result.num_prefills, N)
        self.assertEqual(result.num_decodes, 0)
        self.assertEqual(result.num_prefill_tokens, 15)
        # mixed_metadata may or may not be populated depending on has_initial_states
        # (prefix_lens are all 0, so no initial states expected)
        self.assertIsNotNone(result.mixed_metadata, "mixed_metadata should be populated (even if empty)")
        self.assertFalse(result.prep_initial_states, "No initial states should be prepared for prefix_lens=0")

    def test_chunk_indices_offsets_correctness(self):
        """Docstring worked example: query_start_loc=[0,5,10], chunk_size=8 → indices=[0,0,1], offsets=[0,5,0]."""
        query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32)
        chunk_size = 8
        total_seqlens = 10

        chunk_indices, chunk_offsets = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
            query_start_loc, chunk_size, total_seqlens
        )

        expected_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
        expected_offsets = torch.tensor([0, 5, 0], dtype=torch.int32)

        self.assertTrue(
            torch.equal(chunk_indices, expected_indices),
            f"chunk_indices mismatch: got {chunk_indices}, expected {expected_indices}"
        )
        self.assertTrue(
            torch.equal(chunk_offsets, expected_offsets),
            f"chunk_offsets mismatch: got {chunk_offsets}, expected {expected_offsets}"
        )

    def test_has_initial_states_flag(self):
        """context_lens with mixed zeros and non-zeros → has_initial_states correct; prep_initial_states=True."""
        N = 4
        device = "cpu"
        fwd_meta = _make_forward_metadata(num_seqs=N, device=device)

        forward_batch = MagicMock()
        forward_batch.extend_num_tokens = 20
        forward_batch.extend_seq_lens = [5] * N
        forward_batch.extend_seq_lens_cpu = [5] * N
        # Mix: first 2 have context (non-zero prefix), last 2 don't
        forward_batch.extend_prefix_lens = torch.tensor([10, 5, 0, 0], dtype=torch.int32)
        forward_batch.seq_lens = torch.tensor([5] * N, dtype=torch.int32)
        forward_batch.spec_info = None
        forward_batch.forward_mode = MagicMock()
        forward_batch.forward_mode.is_target_verify.return_value = False

        chunk_size = 8
        result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)

        self.assertIsNotNone(result.mixed_metadata)
        expected_has_initial = torch.tensor([True, True, False, False])
        self.assertTrue(
            torch.equal(result.mixed_metadata.has_initial_states, expected_has_initial),
            f"has_initial_states: got {result.mixed_metadata.has_initial_states}"
        )
        self.assertTrue(result.mixed_metadata.prep_initial_states)

    def test_mamba_cache_indices_preserved(self):
        """mamba_cache_indices from ForwardMetadata passes through to Mamba2Metadata unchanged."""
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

        self.assertTrue(
            torch.equal(result.mamba_cache_indices, indices),
            f"mamba_cache_indices changed: got {result.mamba_cache_indices}"
        )


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- All 5 tests pass
- `test_chunk_indices_offsets_correctness` matches the exact values from the docstring worked example
- `test_has_initial_states_flag` correctly identifies which requests have non-zero prefix lengths
- No import errors from `mamba2_metadata.py`
- No server or GPU required

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-05-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 5 — Mamba2Metadata / ForwardMetadata Integrity
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: <date>
**Result**: PASS | FAIL

## Test Results
| Test | Result |
|------|--------|
| test_prepare_decode_pure_decode_batch | PASS/FAIL |
| test_prepare_mixed_prefill_only | PASS/FAIL |
| test_chunk_indices_offsets_correctness | PASS/FAIL |
| test_has_initial_states_flag | PASS/FAIL |
| test_mamba_cache_indices_preserved | PASS/FAIL |

## Docstring Example Verification
query_start_loc=[0,5,10], chunk_size=8 → chunk_indices=<actual>, chunk_offsets=<actual>
Expected: [0,0,1] / [0,5,0] — Match: YES | NO

## Failures & Tracebacks
<paste here or "None">
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 5 RESULT: PASS | FAIL
Tests run: 5  Passed: X  Failed: Y
Docstring example verified: YES | NO
Report: $RESULTS_DIR/phase-05-<model>-<date>.md
```
