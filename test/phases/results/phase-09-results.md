# Phase 9 — Gauntlet / Stress Tests
**Model**: granite-4.0-h-tiny
**Date**: 2026-03-29
**Duration**: 43.39s
**Result**: PASS

## Server Health
- Healthy post-stress (/health returns 200): YES
- CUDA errors logged: NONE
- Eviction errors: NONE
- mamba_lock_ref violations: NONE
- Server log size: (not measured)

## Test Results
| Test | Requests | Pass | Errors |
|------|----------|------|--------|
| test_high_concurrency_shared_prefix | 32 concurrent | 32 | 0 |
| test_rapid_distinct_requests_eviction_pressure | 100 | 100 | 0 |
| test_repeated_same_request_cache_stability | 50 | 50 | 0 |
| test_alternating_long_and_short_requests | 40 | 40 | 0 |
| test_concurrent_multi_turn_conversations | 8×5 turns | 40 | 0 |
| test_server_health_after_stress | 1 | 1 | 0 |

## Additional Tests from phase3/test/test_plan.md
Covered by existing gauntlet tests:
- `test_scheduler_multiple_requests()` → `test_high_concurrency_shared_prefix`, `test_concurrent_multi_turn_conversations`
- `test_memory_limits()` → `test_rapid_distinct_requests_eviction_pressure`, `test_alternating_long_and_short_requests`

## Observed Anomalies
None. Server remained stable throughout all stress tests.

## Failures & Tracebacks
None.

## Summary
All 6 stress tests passed with zero errors:
- 32 concurrent requests with shared prefix: all completed without state contamination
- 100 rapid unique requests: server stayed healthy, eviction handled correctly
- 50 repeated identical requests: all outputs identical at temperature=0 (no cache corruption)
- 20 alternating long/short requests: no cross-contamination
- 8 concurrent 5-turn conversations: all maintained persona coherence
- Post-stress health check: server still responsive

The Mamba radix cache, memory pool, and scheduler demonstrated robust behavior under concurrent load, eviction pressure, and repeated requests.
