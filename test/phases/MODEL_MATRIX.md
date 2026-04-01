# Engram — Model Compatibility Matrix

This document tracks all models validated against the Engram snapshot infrastructure. Each row represents a completed run of the [Model Compatibility Protocol](MODEL_COMPAT_PROTOCOL.md).

**Last updated:** 2026-04-01 (stateful recall re-run post PR #20)

---

## Compatibility Matrix

| Model | Vendor | Architecture | Recurrent Type | Format | Params (Total/Active) | Test Protocol | Pass Rate | Stateful Recall | Verdict | Date | Issue |
|-------|--------|-------------|----------------|--------|----------------------|---------------|-----------|-----------------|---------|------|-------|
| Granite 4.0-H-tiny | IBM | `GraniteMoeHybridForCausalLM` | Mamba2 SSM | BF16 | 4B | Full phases 0–8 + 10 | 77/82 (93.9%) | PASS | COMPATIBLE | 2026-03-31 | — |
| Granite 4.0-H-small | IBM | `GraniteMoeHybridForCausalLM` | Mamba2 SSM | BF16 | 32B | Compat protocol (full)⁵ | 0 model-specific failures | PASS⁶ | COMPATIBLE | 2026-04-01 | — |
| Nemotron-Cascade-2-30B | NVIDIA | `NemotronHForCausalLM` | Mamba2 SSM | BF16 | 30B / 3B active | Phase 10 cross-model⁴ | 5/5 (100%) | PASS⁶ | COMPATIBLE | 2026-04-01 | — |
| Nemotron-3-Super-120B-A12B | NVIDIA | `NemotronHForCausalLM` | Mamba2 SSM | FP8 | 120B / 12B active | Ad-hoc (partial)¹ | 54/56 (96.4%) | PASS (4/4)⁶ | COMPATIBLE | 2026-04-01 | KHA-203 |
| Qwen3-Coder-Next | Alibaba | `Qwen3NextForCausalLM` | Gated Linear Attention | FP8 | ~75B / 3.9B active | Compat protocol (full) | 62/62 (100%) | PASS (4/4)⁶ | COMPATIBLE | 2026-04-01 | KHA-204 |
| Codestral Mamba 7B | Mistral | `Mamba2ForCausalLM` | Mamba2 SSM (pure) | BF16 | 7B | Gate 1 only | 0/0 | — | BLOCKED³ | 2026-03-31 | KHA-185 |

### Footnotes

¹ **Ad-hoc (partial):** Ran registered test suites (phases 2, 3, 5, 7, 8 equivalent) plus manual baseline checks. Did NOT run formal phase 0 environment verification, phase 4 log-line inspection, or phase 6 extra_buffer strategy. Stateful recall (phase 8 semantic) not validated. Future runs should use the [Model Compatibility Protocol](MODEL_COMPAT_PROTOCOL.md).

² **BLOCKED (stateful recall):** ~~Restore API gap — fixed in PR #20.~~ No longer applicable — see footnote ⁶.

⁶ **PASS (stateful recall, post PR #20):** Re-validated 2026-04-01 after PR #20 merged. `restore_snapshot` with `continuation_ids` + `create_new_request=true` now correctly runs stateful generation and returns `output_text`. All 4 previously-blocked models recalled a fact established in a prior turn using only the new continuation tokens. Results in `results/stateful-recall-rerun-20260401.md`.

³ **BLOCKED (no model class):** SGLang has no native `Mamba2ForCausalLM` model class for pure Mamba2 architectures. Only HuggingFace Transformers fallback exists. Tracked in upstream sglang issues #7429 and #18458.

⁴ **Phase 10 cross-model:** 5 structured compatibility tests per model covering: baseline inference, snapshot save/restore, memory leak detection, rapid-fire throughput, and snapshot sizing. Run on RunPod A100-SXM4-80GB. Result files recovered from `backup/phase-10-model-testing-20260330` branch.

⁵ **Compat protocol (base model):** Full protocol run on H200. granite-small has no chat template (base model, not instruction-tuned). 14 test failures = chat endpoint returning HTTP 400 (test harness assumes instruction-tuned model). 2 failures = pre-existing restore API gap (all models). Zero model-specific infrastructure failures. Custom gauntlet using `/generate` endpoint: 142/142 PASS.

---

## Architecture Coverage

| Architecture Family | Recurrent State Type | Models Tested | Status |
|--------------------|--------------------|---------------|--------|
| Dense Mamba2 hybrid | Mamba2 SSM (conv_state + ssm_state) | Granite 4.0-H-tiny (full phases), Granite 4.0-H-small (Phase 10) | Full + cross-model |
| MoE Mamba2 hybrid | Mamba2 SSM | Nemotron-Cascade-2-30B (Phase 10, 5/5) | Cross-model |
| LatentMoE Mamba2 hybrid | Mamba2 SSM | Nemotron-3-Super-120B-A12B (FP8) | Compat protocol |
| GLA + MoE hybrid | Gated Linear Attention (via Mamba cache) | Qwen3-Coder-Next (FP8) | Compat protocol |
| Pure Mamba2 SSM | Mamba2 SSM (no attention) | Codestral Mamba 7B | Blocked — no SGLang model class |

### Key Finding: Recurrent State Generalization

SGLang routes Gated Linear Attention (GLA) recurrent state through the same Mamba cache subsystem used for Mamba2 SSM. Engram's snapshot infrastructure captures and persists GLA state identically — **zero code changes required**. This means Engram generalizes to any architecture whose recurrent state flows through SGLang's Mamba cache path, including GLA, DeltaNet, and potentially RWKV/RetNet variants.

---

## Quantization Coverage

| Format | Models Tested | Notes |
|--------|---------------|-------|
| BF16 (safetensors) | Granite tiny/small, Nemotron-Cascade, Codestral | Standard unquantized |
| FP8 (safetensors) | Nemotron-3-Super-120B, Qwen3-Coder-Next | No dtype mismatches observed. FP8 KV cache works correctly. |
| GGUF | — | Not tested. SGLang GGUF support is limited. |
| GPTQ / AWQ | — | Not tested. |

---

## Operational Notes

| Model | Note |
|-------|------|
| Nemotron-3-Super-120B FP8 | VRAM tight: 133.4 GB / 143 GB on H200. Won't fit on 80GB GPUs without TP. |
| Qwen3-Coder-Next FP8 | **Must set `SGLANG_ENABLE_JIT_DEEPGEMM=0`** — avoids ~13hr JIT kernel compilation for FP8 block-quantized MoE. |
| Nemotron-3-Super-120B FP8 | Bug found: `is_multimodal_gen` missing from ModelConfig (commit `e224e2512`). |
| Codestral Mamba 7B | Bug found: `Mamba2Config` missing `num_attention_heads` — fixed with hasattr guards (commit `10ec1e4c4`). |
| Nemotron-Cascade-2-30B | Star performer in Phase 10: 3x faster inference than granite-small (0.059s vs 0.172s), 100% snapshot save rate, ~47MB per snapshot. Recommended as primary test model for speed. |
| Granite 4.0-H-small | **Base model — no chat template.** All `/v1/chat/completions` tests fail HTTP 400; use `/v1/completions` or `/generate`. Snapshot save: 145.7MB, instant WARM tier hit. Gauntlet 142/142 via `/generate`. VRAM: 130.3 GB / 143.8 GB on H200. Piecewise CUDA graph disabled (non-GQA attention). |
| All reasoning models | Default `temperature=1.0` and `<think>` CoT blocks can cause test failures in harnesses that expect deterministic short answers. Set `temperature=0` explicitly in test prompts. |

---

## Result Files

| Model | Result File |
|-------|-------------|
| Granite 4.0-H-tiny (phases) | `results/phase-*-granite-4.0-h-tiny-*.md` |
| Granite 4.0-H-small (Phase 10) | `results/phase-10-h-small-results.md` |
| Granite 4.0-H-small (compat protocol) | `results/compat-granite-4.0-h-small-20260401.md` |
| Nemotron-Cascade-2-30B (Phase 10) | `results/phase-10-nemotron-results.md` |
| Phase 10 summary | `results/phase-10-final-report.md` |
| Phase 10f resilience | `results/phase-10-resilience-results.md` |
| Nemotron-3-Super-120B FP8 | `results/compat-nemotron-3-super-120b-fp8-20260401.md` |
| Qwen3-Coder-Next FP8 | `results/compat-qwen3-coder-next-fp8-2026-04-01.md` |
| Codestral Mamba 7B | (standalone — see KHA-185 in Linear) |

Future model tests should follow the [Model Compatibility Protocol](MODEL_COMPAT_PROTOCOL.md) and save results to `results/compat-<model>-<date>.md`.

---

## How to Add a Model

1. Follow the [Model Compatibility Protocol](MODEL_COMPAT_PROTOCOL.md)
2. Save results to `test/phases/results/compat-<model_short_name>-<date>.md`
3. Add a row to the matrix table above
4. Update Architecture Coverage and Quantization Coverage if the model introduces a new variant
5. Add any operational notes
6. Create a Linear issue for tracking (optional but recommended)
