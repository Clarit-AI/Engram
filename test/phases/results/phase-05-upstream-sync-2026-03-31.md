# Phase 05 Results — Scheduler Idle and Pool Fixes (Upstream Sync)

**Date:** 2026-03-31
**Branch:** `upstream-sync-2026-Q1`
**Base tag:** `phase-04-pass`
**Result tag:** `phase-05-pass`
**Result:** PASS (unit tests, lint)

---

## Objective

Merge scheduler idle-detection and Mamba slot-release fixes from upstream that touch the same lifecycle paths as our background snapshot/tier behavior. Verify TierManager background tasks do not interfere with the new unified `is_fully_idle()` check.

---

## Commits Integrated

Commits were reordered from the plan's listing to match chronological dependency order (50953aea8 creates `is_fully_idle()` and must precede 4d3976b6c which extends it).

| # | Upstream Commit | Description | Conflicts |
|---|-----------------|-------------|-----------|
| 1 | `b1246c50f` | Fix chunked prefill and KV cache leaks for streaming sessions (#20476) | 2 (memory_pool.py, test file path) |
| 2 | `e2fccb2ee` | Revert Mamba slot release to fix flaky Qwen3-Next tests (#18910) | 0 |
| 3 | `8541b1118` | Pass `max_mamba_cache_size` to mamba pool in disagg decode path (#19002) | 2 (decode.py, model_runner_kv_cache_mixin.py) |
| 4 | `50953aea8` | Unify idle checks into `is_fully_idle()` and fix weight update test (#20296) | 2 (scheduler.py health check, `_is_no_request` removal) |
| 5 | `4d3976b6c` | Check in-flight HiCache async ops in `is_fully_idle()` (#20746) | 0 |
| 6 | `079a1fd35` | Fix write-through events not processed when scheduler idle (#20560) | 1 (scheduler.py `try_preemption` vs `enable_priority_preemption`) |
| 7 | `5270a0648` | Fix disagg health check false-positive in `is_fully_idle` (#20756) | 0 |
| 8 | `26f709e97` | Make prefill-delayer compatible with multiple mem pool types (#20979) | 0 |

**Total:** 8 upstream commits cherry-picked, 7 conflicts resolved.

**Fork-specific fixup commits (2):**
- `165d9045f` — Remove stray conflict marker from 50953aea8 cherry-pick
- `91878b2d3` — Add `PinPrefixReqInput`/`PinPrefixReqOutput` types to `io_struct.py` (companion to upstream `f7da379b6` TTL-based prefix pinning feature; `pin_prefix_wrapped()` was pulled into `scheduler.py` during conflict resolution of 50953aea8)

---

## Conflicts Resolved

### 1. `memory_pool.py` — `ReqToTokenPool.alloc()` (b1246c50f)
- **Conflict:** Our fork used variable name `chunked` with a strict assertion; upstream renamed to `reusing` and relaxed the assertion (commented out) per PR #20476.
- **Resolution:** Adopted upstream's `reusing` variable name and relaxed assertion. Updated downstream assert message and `need_size` calculation to match.

### 2. `test/registered/sessions/test_streaming_session_leak.py` path (b1246c50f)
- **Conflict:** Upstream placed the test at `test/registered/sessions/`; our directory layout maps to `test/manual/`.
- **Resolution:** Accepted at `test/manual/test_streaming_session_leak.py`.

### 3. `disaggregation/decode.py` — `HybridMambaDecodeReqToTokenPool.__init__()` (8541b1118)
- **Conflict:** Two new parameters added independently — `enable_overlap_schedule` (our fork) and `mamba_size` (upstream).
- **Resolution:** Kept both parameters and both body code blocks (`self.start_layer`, `self.layer_transfer_counter` from ours; `effective_mamba_size` from upstream).

### 4. `model_runner_kv_cache_mixin.py` — pool constructor call (8541b1118)
- **Conflict:** Same pattern — `enable_overlap_schedule` kwarg (ours) vs `mamba_size` kwarg (upstream).
- **Resolution:** Kept both kwargs in the constructor call.

### 5. `scheduler.py` — health check logic (50953aea8)
- **Conflict:** Our fork had inline `is_health_check_generate_req` with manual busy checks; upstream replaced with `self.is_fully_idle(for_health_check=True)`.
- **Resolution:** Adopted upstream's unified `is_fully_idle()` call.

### 6. `scheduler.py` — `_is_no_request()` vs `pin_prefix_wrapped()` (50953aea8)
- **Conflict:** Upstream deleted `_is_no_request()` (subsumed by `is_fully_idle()`). Our fork had `_is_no_request()` at the same location where upstream also added `pin_prefix_wrapped()`.
- **Resolution:** Deleted `_is_no_request()`, kept `pin_prefix_wrapped()`. Added missing `PinPrefixReqInput`/`PinPrefixReqOutput` types to `io_struct.py` since the feature commit (f7da379b6) wasn't in our cherry-pick set.

### 7. `scheduler.py` — `try_preemption` vs `enable_priority_preemption` (079a1fd35)
- **Conflict:** Upstream renamed `try_preemption` to `enable_priority_preemption`; our fork still uses the old name.
- **Resolution:** Kept our fork's `try_preemption` variable name (renaming is out of scope for this phase). Correctly adopted the `check_hicache_events()` relocation from the commit.

---

## Design Decisions

### 1. Commit ordering
The plan listed commits in a non-chronological order. `50953aea8` (March 10) creates the `is_fully_idle()` function that `4d3976b6c` (March 17) extends. Cherry-picking in plan order would fail. Reordered to: b1246c50f, e2fccb2ee, 8541b1118, **50953aea8**, 4d3976b6c, 079a1fd35, 5270a0648, 26f709e97.

### 2. TierManager vs `is_fully_idle()`
**Concern from plan:** TierManager's background cleanup thread might prevent the scheduler from reaching idle.
**Finding:** TierManager's `_background_cleanup_loop()` runs in a daemon thread using its own `threading.RLock()`. It only operates on `host_pool` (CPU-side Mamba state dict) and disk snapshots. It does not touch any scheduler structures checked by `is_fully_idle()` (running_batch, waiting_queue, grammar_queue, disagg queues, HiCache async ops). **No coordination needed.**

### 3. Mamba slot release revert (e2fccb2ee)
Adopted as-is per plan. Phase 7 and Phase 8 (server-based restore semantics validation) are needed to confirm restore lifecycle still holds after the revert. These require an A100 GPU.

---

## Test Results

### MambaRadixCache Comprehensive (Phase 3 gate)
| Test | Result |
|------|--------|
| `test_cow_mamba_state` | PASS |
| `test_empty_cache_operations` | PASS |
| `test_evict_full_leaves_only` | PASS |
| `test_evictable_size_tracking` | PASS |
| `test_full_cache_eviction` | PASS |
| `test_lock_ref_protection` | PASS |
| `test_lru_list_integrity` | PASS |
| `test_mamba_branching_seqlen` | PASS |
| `test_tombstone_node_creation` | PASS |

**Result: 9/9 PASS**

### MambaRadixCache Gauntlet
| Test | Result |
|------|--------|
| `test_branching_seqlen_triggered` | FAIL (pre-existing) |
| `test_cow_state_independence` | PASS |
| `test_full_evictable_and_protected_size_accounting` | PASS |
| `test_inc_dec_lock_ref_symmetry` | PASS |
| `test_interleaved_insert_evict_match` | PASS |
| `test_tombstone_does_not_match_mamba` | PASS |

**Result: 5/6 PASS** (1 pre-existing failure)

### Pre-commit / Lint
All checks passed after adding `PinPrefixReqInput`/`PinPrefixReqOutput` imports.

### Syntax validation
All modified files pass `ast.parse()`: scheduler.py, memory_pool.py, session_aware_cache.py, decode.py, model_runner_kv_cache_mixin.py, io_struct.py.

---

## Pre-existing Issues

### `test_branching_seqlen_triggered` (gauntlet)
- **Error:** `RuntimeError: Expected all tensors to be on the same device, but got tensors is on cpu, different from other tensors on cuda:0`
- **Root cause:** Test allocator places free_pages on CUDA:0 but insert path creates CPU tensor indices. This existed before Phase 05 cherry-picks (verified by reverting and re-running).
- **Impact:** None on Phase 05 changes.

---

## Files Modified

| File | Changes |
|------|---------|
| `python/sglang/srt/managers/scheduler.py` | `is_fully_idle()` replaces `_is_idle_for_hicache_storage_op()` and `_is_no_request()`; health check unified; hicache events moved; `pin_prefix_wrapped()` added; PinPrefix imports added |
| `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool.alloc()` — relaxed assertion, `reusing` variable name |
| `python/sglang/srt/mem_cache/session_aware_cache.py` | Chunked prefill idempotent restore, streaming session slot retention |
| `python/sglang/srt/disaggregation/decode.py` | `mamba_size` parameter added to `HybridMambaDecodeReqToTokenPool` |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | `mamba_size` kwarg passed to disagg pool constructor |
| `python/sglang/srt/managers/io_struct.py` | `PinPrefixReqInput`, `PinPrefixReqOutput` dataclasses added |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | Weight update test compatibility (from 50953aea8) |
| `python/sglang/test/test_utils.py` | HiCache test utility updates (from 4d3976b6c) |
| `test/manual/test_streaming_session_leak.py` | New test for streaming session KV cache leaks |
| `test/manual/hicache/test_disaggregation_hicache.py` | HiCache disagg test updates |
| `test/registered/hicache/test_hicache_storage_file_backend.py` | Storage backend test updates |
| `docs/advanced_features/hicache_storage_runtime_attach_detach.md` | Doc updates (from 50953aea8) |
| `test/registered/rl/test_update_weights_from_tensor.py` | Weight update test fix (from 50953aea8) |

---

## Next Steps

- **Phase 06/06B:** Already completed (see `phase-06-upstream-sync-2026-03-31.md` and `phase-06b-upstream-sync-2026-03-31.md`)
- **Phase 07/08:** Server-based validation (restore semantics after Mamba slot release revert) — requires A100 GPU
