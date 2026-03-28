# Phase 2 — MambaPool + HybridReqToTokenPool Unit Tests

## Purpose

Verify the memory pool layer (`MambaPool` and `HybridReqToTokenPool`) in isolation, with no server running. These tests operate on GPU tensors directly and catch allocation/free bugs, dtype handling issues, and extra-buffer misconfiguration before those bugs surface in server-level tests. The existing `test_mamba_pool` and `test_hybrid_linear_kv_pool` tests are re-run first as a regression guard, then five new edge-case tests are added.

## Prerequisites

- Phase 0 complete (all 13 existing tests pass)
- Phase 1 is **not** required — this phase needs no server
- GPU available (tests use `get_device()` which falls back to CPU if needed, but GPU is preferred)

## Key Files

- `python/sglang/srt/mem_cache/memory_pool.py` — `HybridReqToTokenPool`, `MambaPool`, `HybridLinearKVPool`
- `python/sglang/srt/configs/mamba_utils.py` — `Mamba2CacheParams`, `Mamba2StateShape`
- `python/sglang/srt/environ.py` — `envs.SGLANG_MAMBA_SSM_DTYPE` override context manager
- `python/sglang/srt/managers/schedule_batch.py` — `Req`
- `test/registered/radix_cache/test_mamba_unittest.py` — existing tests to re-run (lines 36–141 are the reference implementation)
- **New**: `test/registered/radix_cache/test_mamba_pool_extended.py`

## Environment Setup

```bash
cd /home/bbrenner/sglang-mamba
source test/phases/config.sh

# Confirm no server running (these are pure unit tests)
pip install -e python/ --quiet

# Confirm GPU available
python -c "import torch; print('GPU available' if torch.cuda.is_available() else 'WARNING: No CUDA GPU found — tests may skip or fall back to CPU')"
```

## Tasks

### Task 1: Re-run existing pool tests as regression guard

```bash
python -m pytest test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_mamba_pool \
                 test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_hybrid_linear_kv_pool \
                 -v 2>&1 | tee /tmp/phase2_regression.log
```

Expected: both pass.

### Task 2: Write the extended pool test file

Create `test/registered/radix_cache/test_mamba_pool_extended.py` with the content specified in **New Test File(s) to Write** below.

### Task 3: Run the new tests

```bash
python -m pytest test/registered/radix_cache/test_mamba_pool_extended.py -v \
    2>&1 | tee /tmp/phase2_extended.log
```

### Task 4: Verify no memory leaks between tests

After the test run, confirm the log shows `available_size` returns to its initial value after each test (checked in `tearDown`). Search for any `AssertionError` related to available size:

```bash
grep -i "available_size\|assertionerror\|leak" /tmp/phase2_extended.log
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_pool_extended.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=30, suite="stage-b-test-small-1-gpu-amd")

import unittest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device


def _make_pool(max_num_reqs=10, mamba_cache_size=20, max_context_len=128,
               enable_extra_buffer=False):
    """Helper: construct a HybridReqToTokenPool with standard Mamba2 shape."""
    device = get_device()
    num_layers = 48
    global_interval = 4
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=4096,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    return HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_cache_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        enable_mamba_extra_buffer=enable_extra_buffer,
        speculative_num_draft_tokens=3,
    )


def _make_req():
    return Req(
        rid=0,
        origin_input_text="",
        origin_input_ids=[],
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


class TestMambaPoolExtended(unittest.TestCase):

    def setUp(self):
        self.max_num_reqs = 4
        self.mamba_cache_size = 6
        self.pool = _make_pool(self.max_num_reqs, self.mamba_cache_size)
        self.initial_req_size = self.pool.available_size()
        self.initial_mamba_size = self.pool.mamba_pool.available_size()

    def tearDown(self):
        # Each test must return the pool to initial state to catch leaks
        self.assertEqual(
            self.pool.mamba_pool.available_size(),
            self.initial_mamba_size,
            "Mamba pool available_size did not return to initial — possible leak"
        )

    def test_pool_exhaustion(self):
        """Allocate mamba_cache_size slots; next alloc fails gracefully."""
        reqs = [_make_req() for _ in range(self.mamba_cache_size)]
        for req in reqs:
            req.rid = id(req)
        # Alloc until full
        for req in reqs[:self.max_num_reqs]:
            self.pool.alloc([req])
        # Verify mamba pool is exhausted or near-exhausted; free everything cleanly
        for req in reqs[:self.max_num_reqs]:
            self.pool.free_mamba_cache(req)
            self.pool.free(req)

    def test_mamba_pool_reuse_on_no_free(self):
        """Alloc req, free req without free_mamba_cache; re-alloc reuses the leaked slot."""
        req = _make_req()
        self.pool.alloc([req])
        self.assertEqual(self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1)

        # Free req slot but NOT mamba slot (simulates the leak scenario)
        self.pool.free(req)
        self.assertEqual(self.pool.available_size(), self.initial_req_size)
        # Mamba slot is still consumed
        self.assertEqual(self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1)

        # Re-alloc the same req — the existing mamba slot should be reused
        req.mamba_pool_idx = None  # reset so alloc tries fresh
        self.pool.alloc([req])
        # Mamba available size must not decrease further (reuse, not new alloc)
        self.assertEqual(self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1)

        # Clean up
        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_mamba_state_dtype_override(self):
        """SGLANG_MAMBA_SSM_DTYPE override produces bfloat16 temporal states."""
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
        num_layers = 48
        global_interval = 4
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
        # Temporal states should be bfloat16
        self.assertEqual(cache_params.ssm_state_dtype, torch.bfloat16)

    def test_get_mamba_indices_mapping(self):
        """get_mamba_indices returns indices matching req.mamba_pool_idx after alloc."""
        req = _make_req()
        self.pool.alloc([req])
        self.assertIsNotNone(req.mamba_pool_idx)

        idx_tensor = self.pool.get_mamba_indices(req.req_pool_idx)
        # The returned tensor should encode the mamba pool index for this req
        self.assertIsNotNone(idx_tensor)

        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_enable_mamba_extra_buffer_false(self):
        """HybridReqToTokenPool with enable_mamba_extra_buffer=False has no ping-pong buffer."""
        pool_no_extra = _make_pool(enable_extra_buffer=False)
        req = _make_req()
        pool_no_extra.alloc([req])
        # ping_pong buffer should not be allocated
        self.assertIsNone(getattr(req, "mamba_ping_pong_track_buffer", None))
        pool_no_extra.free_mamba_cache(req)
        pool_no_extra.free(req)


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- Regression guard: `test_mamba_pool` and `test_hybrid_linear_kv_pool` still pass
- All 5 new tests in `test_mamba_pool_extended.py` pass
- `tearDown` assertions confirm `mamba_pool.available_size()` returns to initial after every test (no leaks)
- No CUDA memory errors or unexpected OOM

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-02-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 2 — MambaPool + HybridReqToTokenPool Unit Tests
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: <date>
**Result**: PASS | FAIL

## Regression Guard
| Test | Result |
|------|--------|
| test_mamba_pool | PASS/FAIL |
| test_hybrid_linear_kv_pool | PASS/FAIL |

## New Tests (test_mamba_pool_extended.py)
| Test | Result |
|------|--------|
| test_pool_exhaustion | PASS/FAIL |
| test_mamba_pool_reuse_on_no_free | PASS/FAIL |
| test_mamba_state_dtype_override | PASS/FAIL |
| test_get_mamba_indices_mapping | PASS/FAIL |
| test_enable_mamba_extra_buffer_false | PASS/FAIL |

## Memory Leak Check
tearDown available_size assertion: PASS | FAIL
Details: <if failed>

## Failures & Tracebacks
<paste here or "None">
EOF
echo "Report written to $REPORT"
```

## Reporting

```text
PHASE 2 RESULT: PASS | FAIL
Regression tests: PASS | FAIL
New tests run: 5  Passed: X  Failed: Y
Memory leak detected: YES | NO
Report: $RESULTS_DIR/phase-02-<model>-<date>.md
```
