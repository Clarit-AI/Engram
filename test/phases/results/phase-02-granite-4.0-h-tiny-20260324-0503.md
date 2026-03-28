# Phase 2 — MambaPool + HybridReqToTokenPool Unit Tests
**Model**: granite-4.0-h-tiny  *(model not loaded; unit tests only)*
**Date**: 2026-03-24
**Result**: PASS

## Regression Guard
| Test | Result |
|------|--------|
| test_mamba_pool | PASS |
| test_hybrid_linear_kv_pool | PASS |

## New Tests (test_mamba_pool_extended.py)
| Test | Result |
|------|--------|
| test_pool_exhaustion | PASS |
| test_mamba_pool_reuse_on_no_free | PASS |
| test_mamba_state_dtype_override | PASS |
| test_get_mamba_indices_mapping | PASS |
| test_enable_mamba_extra_buffer_false | PASS |

## Memory Leak Check
tearDown available_size assertion: PASS
Details: All 5 tests returned mamba_pool.available_size() to initial state.

## Failures & Tracebacks
None
