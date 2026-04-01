# Stateful Recall Re-Run — Post PR #20 Fix

**Date:** 2026-04-01
**Machine:** H200 (143.8 GB VRAM)
**Branch:** main (post PR #20 merge, HEAD b7777ae87)
**Fix:** `scheduler.py` — restored `continuation_ids` + `_stateful_generate` in `handle_restore_snapshot`; `tokenizer_manager.py` — added `output_ids → output_text` decode step

## Summary

All previously-BLOCKED models now PASS stateful recall. The regression was introduced during Phase 8 reconstruction (lost A100 session); PR #20 restored the three-component design.

| Model | Stateful Recall | Token Savings | Notes |
|-------|----------------|---------------|-------|
| Nemotron-3-Super-120B-A12B FP8 | **PASS (4/4)** | 73.7% (test suite) | Instruction-tuned; standard test file |
| Qwen3-Coder-Next FP8 | **PASS (4/4)** | 73.7% (test suite) | Instruction-tuned; standard test file |
| Granite 4.0-H-small BF16 | **PASS** | 75.0% | Base model — custom `/v1/completions` test |
| Nemotron-Cascade-2-30B-A3B BF16 | **PASS** | 75.7% | No chat template — custom `/v1/completions` test |

## Previously Confirmed

| Model | Stateful Recall | Notes |
|-------|----------------|-------|
| Granite 4.0-H-tiny BF16 | PASS (4/4) | PR #20 validation run, 73.7% token savings |

---

## Results Detail

### Nemotron-3-Super-120B-A12B FP8

Server load: 164s, VRAM: 133.4 GB / 143.8 GB

```
test_multi_turn_stateful_chain           PASSED
test_stateful_recall_semantic            PASSED
test_stateful_vs_full_resend_equivalence PASSED
test_token_savings_quantification        PASSED
4 passed in 23.76s
```

### Qwen3-Coder-Next FP8

Server load: 52s, `SGLANG_ENABLE_JIT_DEEPGEMM=0`

```
test_multi_turn_stateful_chain           PASSED
test_stateful_recall_semantic            PASSED
test_stateful_vs_full_resend_equivalence PASSED
test_token_savings_quantification        PASSED
4 passed in 13.30s
```

### Granite 4.0-H-small BF16 (base model)

Server load: 122s, VRAM: ~130 GB. No chat template → custom test using `/v1/completions`.

- Turn 1: `"The secret number is 42."` (via `/v1/completions`, rid=base-model-stateful-64d067f5)
- Snapshot: PASS — `base-model-stateful-64d067f5-t0`
- Continuation: `" What is the secret number? The answer is"`
- Output: `'42.Questions generated:\n1. Is the secret number a multiple of 5...'`
- **PASS** — recalled "42", 75.0% token savings (36 → 9 tokens)

### Nemotron-Cascade-2-30B-A3B BF16

Server load: 132s. No chat template → custom `/v1/completions` test.

**Note:** Model backup was missing `configuration_nemotron_h.py`. Copied from H200's HF module cache (Nemotron-3-Super uses the same `NemotronHConfig` class — compatible). Model loaded cleanly after copy.

- Turn 1: `"The secret number is 42."` (rid=cascade-stateful-3bf2911b)
- Snapshot: PASS — `cascade-stateful-3bf2911b-t0`
- Continuation: `" What is the secret number? The answer is"`
- Output: `'... 42. The secret number is 42. The answer is ... 42'`
- **PASS** — recalled "42", 75.7% token savings (37 → 9 tokens)

---

## Infrastructure Notes

- All 5 models (including granite-tiny from PR #20) confirm the fix generalizes: dense Mamba2, MoE Mamba2, LatentMoE Mamba2, and GLA (Qwen3)
- Granite-small and Nemotron-Cascade have no chat template; stateful generate operates at token level — endpoint format irrelevant
- **Nemotron-Cascade setup:** `configuration_nemotron_h.py` must be present in model directory for `--trust-remote-code` load. Copy from `~/.cache/huggingface/modules/transformers_modules/NVIDIA_hyphen_Nemotron_hyphen_3_hyphen_Super_hyphen_120B_hyphen_A12B_hyphen_FP8/` if missing.
