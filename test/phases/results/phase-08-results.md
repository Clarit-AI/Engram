# Phase 8 — True Stateful Inference: PASS 4/4

**Date**: 2026-03-29
**Branch**: `phase-08-true-stateful-inference`
**Model**: `granite-4.0-h-tiny` on A100 (SM80)
**Server flags**: `--disable-radix-cache --enable-snapshot-persistence --mamba-scheduler-strategy no_buffer`

---

## Result

```text
4 passed, 1 warning in 13.24s
```

| Test | Result |
|------|--------|
| `test_stateful_recall_semantic` | PASS |
| `test_stateful_vs_full_resend_equivalence` | PASS |
| `test_multi_turn_stateful_chain` | PASS |
| `test_token_savings_quantification` | PASS (93.8% token savings) |

---

## Token Savings

```text
Full resend: 97 tokens
Stateful:     6 tokens
Savings:     91 tokens (93.8%)
```

---

## Bugs Found and Fixed

### Bug 8-1: libnuma1 not installed

**Symptom**: `sgl_kernel` failed to import — `libnuma.so.1: cannot open shared object file`
**Root cause**: sgl-kernel 0.3.21 routes SM80 → `sm100/` binary (regression in packaging). That `.so` links against `libnuma`.
**Fix**: `sudo apt-get install -y libnuma1`

### Bug 8-2: ninja not installed

**Symptom**: `flashinfer` JIT compilation failed — `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`
**Fix**: `sudo apt-get install -y ninja-build`

### Bug 8-3: Auto-snapshot fired after `free_mamba_cache` (state already gone)

**Symptom**: `save_snapshot` returned `success: False` — "no WARM tier state for: rid"
**Root cause**: `_trigger_snapshot_hooks(batch)` in `scheduler.py:3491` is called **after** `process_batch_result_decode`, which internally calls `release_kv_cache → free_mamba_cache`, setting `req.mamba_pool_idx = None`. The hook skips requests with `None` pool idx, so finished requests were never snapshotted.
**Files changed**:
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`: Added pre-free snapshot trigger inside `process_batch_result_decode`, directly before `release_kv_cache`, only for `req.finished()` reqs with live `mamba_pool_idx`.
- `python/sglang/srt/managers/scheduler.py`: Updated `_trigger_snapshot_hooks` comment and added `req.finished()` skip (since those are now handled pre-free).

### Bug 8-4 (pre-existing, partially masked): Auto-snapshot saved partial state after 1 decode token

**Symptom**: `test_multi_turn_stateful_chain` recalled Turn 1 context but not Turn 2 (original code, 3/4 pass)
**Root cause**: `_trigger_snapshot_hooks` fired on every decode step. Due to `min_snapshot_interval_seconds=1.0`, only the first step (1 output token) was saved. Turn 2 state in SSM was never captured.
**Fix**: Bug 8-3 fix inherently resolves this — the pre-free hook fires only once per request (at completion), capturing the full final state. `turn_number` uses `len(req.fill_ids)` so it's strictly increasing across turns.

---

## Environment Setup (new VM)

Required before first server start:
```bash
sudo apt-get install -y libnuma1 ninja-build
```

Both are now required system dependencies for sgl-kernel 0.3.21 on SM80.

---

## Next Phase

Phase 6 — `extra_buffer` cache sharing strategy
Doc: `test/phases/phase-06-extra-buffer-strategy.md`
