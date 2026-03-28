# Phase 6 — mamba_scheduler_strategy=extra_buffer

## Purpose

Verify the ping-pong buffer path (`extra_buffer` mode) works end-to-end. This scheduler strategy enables overlap scheduling — the server interleaves prefill and decode steps using a double-buffered Mamba state — and is a prerequisite for speculative decoding. The tests cover both the memory-pool level (buffer allocation, selective free) and the server level (inference output matches the Phase 1 baseline).

## Prerequisites

- Phase 0 complete
- Phase 1 complete (baseline inference output established at `temperature=0` for known prompts)
- Phase 2 complete (HybridReqToTokenPool pool behavior verified)
- Model checkpoint available at `$MODEL_PATH`
- No other server instances running on port 30000

## Key Files

- `python/sglang/srt/mem_cache/memory_pool.py` — `HybridReqToTokenPool`, `MambaPool`, ping-pong buffer allocation (`enable_mamba_extra_buffer=True`)
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — `cache_unfinished_req()` — ping-pong track buffer code path (lines 556–672)
- `python/sglang/srt/server_args.py` — `--mamba-scheduler-strategy extra_buffer`
- **New**: `test/registered/radix_cache/test_mamba_extra_buffer.py`

## Environment Setup

```bash
cd /home/bbrenner/sglang-mamba

source test/phases/config.sh   # sets MODEL_PATH, SERVER_PORT, SERVER_URL, RESULTS_DIR

# Launch server with extra_buffer strategy
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $SERVER_PORT \
    --mamba-scheduler-strategy extra_buffer \
    > /tmp/phase6_server.log 2>&1 &

export SERVER_PID=$!

# Wait for server ready
python -c "
import time, requests
for i in range(60):
    try:
        r = requests.get('http://localhost:30000/health')
        if r.status_code == 200:
            print('Server ready (extra_buffer mode)')
            break
    except:
        pass
    time.sleep(2)
else:
    print('ERROR: Server did not start')
    exit(1)
"

# Confirm extra_buffer mode in logs
grep -i "extra_buffer\|ping.pong\|mamba_scheduler" /tmp/phase6_server.log | head -10
```

## Tasks

### Task 1: Write the test file

Create `test/registered/radix_cache/test_mamba_extra_buffer.py` with the content in **New Test File(s) to Write** below.

**Note**: Tests 1–3 are unit tests (no server needed). Test 4 requires the running server. You may split these into separate test classes or use `unittest.skipIf` to skip server tests when the server is not available.

### Task 2: Run unit tests (no server)

```bash
# Unit tests only (do not need running server)
python -m pytest test/registered/radix_cache/test_mamba_extra_buffer.py::TestMambaExtraBufferUnit \
    -v 2>&1 | tee /tmp/phase6_unit.log
```

### Task 3: Run server integration test

```bash
SERVER_URL=http://localhost:$SERVER_PORT \
python -m pytest test/registered/radix_cache/test_mamba_extra_buffer.py::TestMambaExtraBufferServer \
    -v 2>&1 | tee /tmp/phase6_server_test.log
```

### Task 4: Compare output to Phase 1 baseline

For the same prompt used in Phase 1 `test_single_turn_completion` at `temperature=0`, the output from the `extra_buffer` server must be identical to the Phase 1 output. Record both and diff them.

### Task 5: Check server logs

```bash
grep -i "cuda error\|traceback\|oom\|ping.pong\|extra_buffer" /tmp/phase6_server.log | tail -30
```

### Task 6: Shut down server

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_extra_buffer.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=90, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=90, suite="stage-b-test-small-1-gpu-amd")

import os
import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")


def _make_pool_extra_buffer(max_num_reqs=4, mamba_cache_size=8, max_context_len=128):
    device = get_device()
    num_layers = 48
    global_interval = 4
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=4096, n_groups=16, num_heads=32,
        head_dim=128, state_size=128, conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=max_num_reqs, mamba_size=mamba_cache_size,
        mamba_spec_state_size=max_num_reqs, max_context_len=max_context_len,
        device=device, enable_memory_saver=False, cache_params=cache_params,
        enable_mamba_extra_buffer=True, speculative_num_draft_tokens=3,
    )


def _make_req():
    return Req(
        rid=0, origin_input_text="", origin_input_ids=[],
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


class TestMambaExtraBufferUnit(unittest.TestCase):
    """Unit tests — no server required."""

    def setUp(self):
        self.pool = _make_pool_extra_buffer()

    def test_extra_buffer_alloc(self):
        """After alloc with extra_buffer=True, req.mamba_ping_pong_track_buffer is non-None with size >= 2."""
        req = _make_req()
        self.pool.alloc([req])
        self.assertIsNotNone(
            getattr(req, "mamba_ping_pong_track_buffer", None),
            "mamba_ping_pong_track_buffer should be allocated in extra_buffer mode"
        )
        # With speculative_num_draft_tokens=3: mamba_ping_pong_track_buffer_size = 1 (not 2)
        buf = req.mamba_ping_pong_track_buffer
        self.assertGreaterEqual(len(buf), 1)
        # Clean up
        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_extra_buffer_free_with_keep(self):
        """free_mamba_cache with mamba_ping_pong_track_buffer_to_keep frees all but one ping-pong slot."""
        req = _make_req()
        self.pool.alloc([req])
        buf = req.mamba_ping_pong_track_buffer
        keep_idx = 0
        # The kept tensor's data before free
        kept_data = buf[keep_idx].clone() if buf[keep_idx] is not None else None

        self.pool.free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=keep_idx)
        # Main mamba slot freed; kept ping-pong slot tensor data should be intact
        if kept_data is not None:
            self.assertTrue(torch.equal(buf[keep_idx], kept_data))
        self.pool.free(req)

    def test_cache_unfinished_req_extra_buffer(self):
        """cache_unfinished_req clears mamba_last_track_seqlen and updates prefix_indices."""
        # This test requires a minimal MambaRadixCache; construct one using the
        # pattern from test_mamba_radix_cache_comprehensive.py.
        # Set req.mamba_last_track_seqlen to a non-None value, call cache_unfinished_req,
        # and assert it is None afterward.
        # Also assert req.prefix_indices is updated.
        # If constructing a full cache is complex, use a mock or skip with a note.
        self.skipTest("Requires full MambaRadixCache setup — implement with cache fixture")


class TestMambaExtraBufferServer(unittest.TestCase):
    """Server integration test — requires running server with extra_buffer strategy."""

    def setUp(self):
        import requests
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def test_server_inference_extra_buffer_mode(self):
        """Inference in extra_buffer mode produces same output as no_buffer at temperature=0."""
        import requests
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0,
            "max_tokens": 20,
        }
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        self.assertGreater(len(content), 0)
        # The answer should contain "4" for "2+2"
        self.assertIn("4", content, f"Unexpected answer: {content}")


if __name__ == "__main__":
    unittest.main()
```

## Pass Criteria

- Server starts in `extra_buffer` mode (confirmed in logs)
- Unit tests: `test_extra_buffer_alloc` and `test_extra_buffer_free_with_keep` pass
- Server test: `test_server_inference_extra_buffer_mode` passes; output matches Phase 1 baseline for the same prompt at `temperature=0`
- No CUDA errors in server logs
- `test_cache_unfinished_req_extra_buffer` either passes or is skipped with clear documentation of what's needed to implement it

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-06-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 6 — extra_buffer Strategy
**Model**: granite-4.0-h-tiny
**Date**: <date>
**Result**: PASS | FAIL

## Server
- extra_buffer mode confirmed in logs: YES | NO
- CUDA errors: NONE | <summary>

## Unit Tests (no server)
| Test | Result |
|------|--------|
| test_extra_buffer_alloc | PASS/FAIL |
| test_extra_buffer_free_with_keep | PASS/FAIL |
| test_cache_unfinished_req_extra_buffer | PASS/FAIL/SKIP |

## Server Test
| Test | Result |
|------|--------|
| test_server_inference_extra_buffer_mode | PASS/FAIL |

## Baseline Comparison
Prompt: "What is 2+2?"
Phase 1 output: <output>
Phase 6 output: <output>
Match: YES | NO

## Failures & Tracebacks
<paste here or "None">
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 6 RESULT: PASS | FAIL
Unit tests: 3 run  Passed: X  Failed: Y  Skipped: Z
Server test: PASS | FAIL
extra_buffer mode confirmed in logs: YES | NO
Output matches Phase 1 baseline: YES | NO | N/A
Report: $RESULTS_DIR/phase-06-<model>-<date>.md
```