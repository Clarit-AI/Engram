# Phase 3 — MambaRadixCache Component Tests

## Purpose

Exercise all `MambaRadixCache` operations in isolation — insert, match, evict (full and Mamba-only), tombstone lifecycle, dual LRU list integrity, lock reference counting, and copy-on-write semantics. No server is required. The existing test files are re-run first as a regression guard, then six "gauntlet" tests probe the most complex invariants: LRU ordering under interleaved mutations, tombstone visibility, `mamba_branching_seqlen` triggering, COW independence, lock ref symmetry across a chain, and size accounting conservation.

`sanity_check()` — which validates LRU list integrity end-to-end — is called in every `tearDown`.

## Prerequisites

- Phase 0 complete
- Phase 2 complete (memory pool layer verified)
- No server required

## Key Files

- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — `MambaRadixCache`, `TreeNode`, `LRUList`, `evict_mamba()`, `inc_lock_ref()`, `dec_lock_ref()`, `sanity_check()`, `cache_unfinished_req()`
- `python/sglang/srt/mem_cache/memory_pool.py` — `HybridReqToTokenPool`, `MambaPool`
- `python/sglang/srt/mem_cache/base_prefix_cache.py` — `EvictParams`, `InsertParams`, `MatchPrefixParams`
- `python/sglang/srt/mem_cache/cache_init_params.py` — `CacheInitParams`
- `python/sglang/srt/mem_cache/radix_cache.py` — `RadixKey`
- `test/registered/radix_cache/test_mamba_unittest.py` — reference setup pattern
- `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py` — existing 10 tests
- **New**: `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py`

## Environment Setup

```bash
cd /home/bbrenner/sglang-mamba
source test/phases/config.sh
pip install -e python/ --quiet

# Confirm GPU available
python -c "import torch; assert torch.cuda.is_available()"
```

## Tasks

### Task 1: Re-run existing radix cache tests as regression guard

```bash
python -m pytest \
    test/registered/radix_cache/test_mamba_unittest.py::TestMamba::test_mamba_radix_cache_1 \
    test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py \
    -v 2>&1 | tee /tmp/phase3_regression.log
```

Expected: **11 tests pass** (1 from unittest + 10 from comprehensive).

### Task 2: Write AND implement the gauntlet test file

> **Critical**: The test skeleton in **New Test File(s) to Write** contains `raise NotImplementedError` stubs for all 6 test bodies. These are intentional scaffolding — they define the contract but will immediately fail if run as-is. The executing agent must replace every stub with a complete implementation before running the tests. Reference `test_mamba_radix_cache_comprehensive.py` for the `setUpClass` pattern and working examples of insert/match/evict operations. Do not consider this task done until all 6 stubs are replaced and all 6 tests pass.

Create `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py` with the content specified in **New Test File(s) to Write** below, then implement all 6 test bodies.

### Task 3: Run the gauntlet tests

```bash
python -m pytest test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py -v \
    2>&1 | tee /tmp/phase3_gauntlet.log
```

### Task 4: Verify sanity_check passes throughout

```bash
grep -i "sanity\|assertionerror\|lru\|fail" /tmp/phase3_gauntlet.log | head -30
```

## New Test File(s) to Write

**Path**: `test/registered/radix_cache/test_mamba_radix_cache_gauntlet.py`

The test setup should mirror `test_mamba_radix_cache_comprehensive.py` — construct `HybridReqToTokenPool`, a `TokenToKVPoolAllocator`, a `CacheInitParams`, and `MambaRadixCache`. Use `page_size=1`. Call `self.cache.sanity_check()` in `tearDown`.

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=45, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=45, suite="stage-b-test-small-1-gpu-amd")

import unittest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


# --- Implement the following 6 test methods ---

class TestMambaRadixCacheGauntlet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mirror setup from test_mamba_radix_cache_comprehensive.py
        # Construct pool, allocator, cache — see that file for the exact pattern.
        pass

    def tearDown(self):
        self.cache.sanity_check()

    def test_interleaved_insert_evict_match(self):
        """Insert 10 seqs; interleave evict_mamba(1) and evict_full(1) between each; sanity_check after every op."""
        # Insert sequences [1], [1,2], ..., [1,2,...,10]
        # Between each insert: evict_mamba(1), evict_full(1)
        # After every operation: self.cache.sanity_check()
        raise NotImplementedError("Implement this test")

    def test_tombstone_does_not_match_mamba(self):
        """Insert [1,2,3] with mamba state; evict_mamba creates tombstone; match returns KV but mamba_value=None."""
        # 1. Insert RadixKey([1,2,3]) with both kv_indices and mamba_value
        # 2. Call evict_mamba(1) to tombstone the node
        # 3. match_prefix(RadixKey([1,2,3]))
        # 4. Assert: result.device_indices is not empty (KV still there)
        # 5. Assert: result.last_device_node.mamba_value is None
        raise NotImplementedError("Implement this test")

    def test_branching_seqlen_triggered(self):
        """KV hit extends past last Mamba node (tombstone gap) → mamba_branching_seqlen is set and chunk-aligned."""
        # 1. Insert long sequence with mamba state
        # 2. Tombstone the mamba state via evict_mamba
        # 3. Insert a longer extension of that sequence
        # 4. match_prefix on the extended sequence
        # 5. Assert result.mamba_branching_seqlen is not None
        # 6. Assert result.mamba_branching_seqlen % mamba_cache_chunk_size == 0
        raise NotImplementedError("Implement this test")

    def test_cow_state_independence(self):
        """After COW, modifying original cached node's mamba_value does not affect the copied state."""
        # 1. Insert sequence; cache node has mamba_value tensor T
        # 2. Trigger COW via match_prefix with cow_mamba=True
        # 3. Modify T in place (e.g., fill with zeros)
        # 4. Assert req's copied mamba state != zeros
        raise NotImplementedError("Implement this test")

    def test_inc_dec_lock_ref_symmetry(self):
        """For 3-node chain root→A→B: inc_lock_ref(B) propagates full_lock_ref up; dec_lock_ref(B) restores all to 0."""
        # 1. Build chain: insert [1,2] → node A, insert [1,2,3,4] → node B
        # 2. inc_lock_ref(B)
        # 3. Assert B.full_lock_ref > 0, A.full_lock_ref > 0
        # 4. Assert B.mamba_lock_ref > 0 (if B has mamba_value)
        # 5. dec_lock_ref(B)
        # 6. Assert B.full_lock_ref == 0, A.full_lock_ref == 0 (back to baseline)
        raise NotImplementedError("Implement this test")

    def test_full_evictable_and_protected_size_accounting(self):
        """After every inc/dec/insert/evict: full_evictable_size() + full_protected_size() == total tokens cached."""
        # For each operation, verify conservation:
        #   cache.full_evictable_size() + cache.full_protected_size() == total inserted tokens
        # Track total_tokens manually as you insert/evict.
        raise NotImplementedError("Implement this test")


if __name__ == "__main__":
    unittest.main()
```

> **Note to implementing agent**: Replace each `raise NotImplementedError` with the full test implementation. Use the setup pattern from `test_mamba_radix_cache_comprehensive.py` for `setUpClass`. The `NotImplementedError` stubs are intentional placeholders — they must all be replaced before the test file is considered complete.

## Pass Criteria

- Regression guard: 11 existing tests still pass
- All 6 new gauntlet tests pass (no `NotImplementedError` stubs remaining)
- `sanity_check()` passes in every `tearDown` (no LRU list corruption detected)
- No torch memory errors

## Write Report

```bash
REPORT="$RESULTS_DIR/phase-03-${MODEL_NAME}-$(date +%Y%m%d-%H%M).md"
cat > "$REPORT" << 'EOF'
# Phase 3 — MambaRadixCache Component Tests
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: <date>
**Result**: PASS | FAIL

## Regression Guard (11 tests)
test_mamba_radix_cache_1: PASS/FAIL
test_mamba_radix_cache_comprehensive (10): X/10 PASS

## Gauntlet Tests (test_mamba_radix_cache_gauntlet.py)
| Test | Result | sanity_check |
|------|--------|--------------|
| test_interleaved_insert_evict_match | PASS/FAIL | PASS/FAIL |
| test_tombstone_does_not_match_mamba | PASS/FAIL | PASS/FAIL |
| test_branching_seqlen_triggered | PASS/FAIL | PASS/FAIL |
| test_cow_state_independence | PASS/FAIL | PASS/FAIL |
| test_inc_dec_lock_ref_symmetry | PASS/FAIL | PASS/FAIL |
| test_full_evictable_and_protected_size_accounting | PASS/FAIL | PASS/FAIL |

## sanity_check violations in tearDown
<NONE or describe which test triggered a violation and what the LRU state was>

## Failures & Tracebacks
<paste here or "None">
EOF
echo "Report written to $REPORT"
```

## Reporting

```
PHASE 3 RESULT: PASS | FAIL
Regression tests: 11 / 11 PASS | X failures
Gauntlet tests: 6 run  Passed: X  Failed: Y
sanity_check violations: NONE | <details>
Report: $RESULTS_DIR/phase-03-<model>-<date>.md
```
