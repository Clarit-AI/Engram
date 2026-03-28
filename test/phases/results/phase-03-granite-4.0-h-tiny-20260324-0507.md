# Phase 3 — MambaRadixCache Component Tests
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: 2026-03-24
**Result**: PASS

## Regression Guard (10 tests)
test_mamba_radix_cache_1: PASS
test_mamba_radix_cache_comprehensive (9): 9/9 PASS

## Gauntlet Tests (test_mamba_radix_cache_gauntlet.py)
| Test | Result | sanity_check |
|------|--------|--------------|
| test_interleaved_insert_evict_match | PASS | PASS |
| test_tombstone_does_not_match_mamba | PASS | PASS |
| test_branching_seqlen_triggered | PASS | PASS |
| test_cow_state_independence | PASS | PASS |
| test_inc_dec_lock_ref_symmetry | PASS | PASS |
| test_full_evictable_and_protected_size_accounting | PASS | PASS |

## sanity_check violations in tearDown
NONE

## Failures & Tracebacks
None
