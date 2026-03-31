# Phase 6 — extra_buffer Strategy
**Model**: granite-4.0-h-tiny
**Date**: 2026-03-29
**Result**: PARTIAL PASS (unit tests only)

## Blocker
`extra_buffer` strategy is **not supported** for available models on this machine:
- `GraniteMoeHybridForCausalLM` (primary model) → `support_mamba_cache_extra_buffer=False`
- `NemotronHForCausalLM` (fallback) → `support_mamba_cache_extra_buffer=False`

Both models have this flag hard-coded in `python/sglang/srt/server_args.py`:
- Granite: line 1686
- Nemotron: line 1575

Models that support `extra_buffer` (per `server_args.py:1627`):
- `Qwen3_5MoeForConditionalGeneration`
- `Qwen3_5ForConditionalGeneration`
- `Qwen3NextForCausalLM`

## Server
- extra_buffer mode confirmed in logs: NO (server refused to start)
- Error: `AssertionError: mamba extra_buffer is not supported for GraniteMoeHybridForCausalLM model`

## Unit Tests (no server)
| Test | Result |
|------|--------|
| test_extra_buffer_alloc | PASS |
| test_extra_buffer_free_with_keep | PASS |
| test_cache_unfinished_req_extra_buffer | SKIP (requires full cache fixture) |

**Unit tests pass** because they test the pool-level allocation logic directly, which is independent of model architecture. The `HybridReqToTokenPool` with `enable_mamba_extra_buffer=True` correctly allocates `mamba_ping_pong_track_buffer` and supports selective freeing.

## Server Test
| Test | Result |
|------|--------|
| test_server_inference_extra_buffer_mode | SKIP (server not available) |

## Baseline Comparison
N/A — server integration test blocked by model support

## Conclusion
The `extra_buffer` code path at the memory pool level is **verified working** (unit tests pass). However, server-level integration testing requires a model with `support_mamba_cache_extra_buffer=True`, which is not available on this RunPod A100 instance.

**Recommendation**: Document Phase 6 as unit-tested only; proceed to Phase 9. Full `extra_buffer` integration testing requires a `Qwen3_5*` model or temporary removal of the architecture restriction in `server_args.py` for experimental purposes.
