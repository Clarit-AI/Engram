# Phase 0 — Environment Verification
**Model**: granite-4.0-h-tiny
**Date**: 2026-03-24
**Result**: FAIL

## Environment
- sglang version: 0.0.0.dev9771+g02d060d36
- CUDA device: Tesla V100-SXM2-16GB (CUDA 12.2 driver, 16 GB VRAM)
- Python: 3.11.2
- PyTorch: 2.5.1+cu121 (installed system-wide via --break-system-packages; venv was empty)
- Model path confirmed: YES (`config.json` + 3 safetensors shards present)

## Installation Notes
- The project `.venv` contained only pip/setuptools. `torch` and `sglang` were not pre-installed.
- The system `/` partition had ~9 GB free. A first pip install attempt failed with `OSError: [Errno 28] No space left on device` while downloading `nvidia-nccl-cu12`.
- The pip HTTP cache (`/home/bbrenner/.cache/pip`) was consuming 8.7 GB. It was purged (`pip3 cache purge`) to recover ~4 GB, bringing free space to ~14 GB.
- `torch-2.5.1+cu121` was installed first (PyTorch wheel index), then `sglang` was installed via `pip3 install -e python/ --break-system-packages --no-cache-dir`. The sglang install replaced torch with `torch-2.9.1`.
- `pytest` was also not present and was installed separately.

## Test Results
| Test File | Tests Collected | Passed | Failed |
|-----------|----------------|--------|--------|
| test_mamba_unittest.py | 3 | 3 | 0 |
| test_mamba_radix_cache_comprehensive.py | 9 | 7 | 2 |
| **Total** | **12** | **10** | **2** |

> Note: The comprehensive suite collected **9** tests, not 10 as the phase document expects. One test may have been removed or renamed upstream.

## Failures

### 1. `test_full_cache_eviction`

```
test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py::TestMambaRadixCacheComprehensive::test_full_cache_eviction FAILED

self = <test_mamba_radix_cache_comprehensive.TestMambaRadixCacheComprehensive testMethod=test_full_cache_eviction>

    def test_full_cache_eviction(self):
        """Test behavior when cache is full and requires eviction."""
        reqs = []
        for i in range(self.mamba_cache_size):
            req = self._make_dummy_req()
            reqs.append(req)
            token_ids = [i]
            kv_indices = self.allocator.alloc(1)
>           mamba_value = req.mamba_pool_idx.unsqueeze(0)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: 'NoneType' object has no attribute 'unsqueeze'

test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py:290: AttributeError
```

**Root cause**: `req.mamba_pool_idx` is `None`. The `_make_dummy_req()` helper does not allocate a Mamba pool slot for the request, so `mamba_pool_idx` is `None` by the time the test tries to call `.unsqueeze(0)` on it. This is a test fixture issue — the dummy request is not wired to a real Mamba pool allocator.

### 2. `test_mamba_branching_seqlen`

```
test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py::TestMambaRadixCacheComprehensive::test_mamba_branching_seqlen FAILED

>       self.assertIsNotNone(
            match_result.mamba_branching_seqlen,
            "mamba_branching_seqlen should be set when tombstone is encountered"
        )
E       AssertionError: unexpectedly None : mamba_branching_seqlen should be set when tombstone is encountered

test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py:518: AssertionError
```

**Root cause**: After evicting a Mamba state to create a tombstone and then re-inserting a longer sequence, `match_prefix` returns a result where `mamba_branching_seqlen` is `None`. The cache's `match_prefix` logic does not set `mamba_branching_seqlen` when a tombstone node is encountered in the matched prefix path. This indicates a missing or broken branch in the Mamba radix cache's tombstone-aware prefix matching logic.

## CUDA Error Check
No CUDA-related errors (e.g., `CUDA error` or CUDA runtime failures) were found in the test logs.

## Notes
- The phase document states "expect 10 tests" for the comprehensive suite, but only 9 were collected. The missing test is not a failure of the runner — the suite simply contains 9 test methods.
- Total tests across both files: **12 collected**, **10 passed**, **2 failed**.
- Both failures are logic/fixture bugs in the test implementation or the underlying cache code, not environment issues.
- No CUDA errors occurred. The GPU (Tesla V100-SXM2-16GB) and CUDA driver are healthy.
- The environment is functional but does **not** meet the Phase 0 pass criterion of all 13 tests passing (actual: 12 collected, 10 passed).