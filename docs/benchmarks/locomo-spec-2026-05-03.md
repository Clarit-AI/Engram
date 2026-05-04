# LoCoMo Benchmark Spec — Engram KHA-255
**Date**: 2026-05-03  
**Ticket**: [KHA-255](https://linear.app/khaentertainment/issue/KHA-255/recognized-benchmark-suite-for-engram-validation)  
**Branch**: `feat/kha-255-locomo-harness-prep`  
**Harness**: `scripts/benchmarks/locomo_runner.py`

---

## Goal

Produce credible, reproducible LoCoMo QA scores comparing:

1. **Stateless baseline** — standard LLM inference, full conversation context injected on every question, no snapshot persistence.
2. **Engram mode** — snapshot save/restore via KHA-348 fix; model builds state incrementally then restores per question.

The primary claim under test: **Engram mode matches or exceeds stateless baseline on LoCoMo F1**, demonstrating that snapshot-based state recall is functionally correct and not regressive.

Secondary measurement: **snapshot save/restore latency at LoCoMo scale** (~500–700 turns per conversation), relevant to the locus-of-intelligence demo.

---

## What Success Looks Like

| Condition | Threshold |
|-----------|-----------|
| Engram overall F1 ≥ stateless overall F1 − 0.02 | Core parity claim validated |
| Engram per-category F1 ≥ stateless per-category F1 − 0.05 | No category-level regression |
| Mean restore latency ≤ 200ms | Acceptable for interactive demo |
| All 1986 questions scored, no crashes | Run completed cleanly |

Results suitable for grant applications / partner decks require a full 10-conversation run on both modes.

---

## Models in Scope

### Smoke run (must pass before full run)
- **Granite-4.0-H-tiny** (IBM, ~4B, FP16)  
  - Path on OVH: `/mnt/models/granite-4.0-h-tiny` (staged from S3 during RunPod wind-down)
  - Purpose: fast iteration, validates harness end-to-end in <30 min
  - Target: 1 conversation (`--conversations 1`), 199 QA items

### Full run — primary
- **Nemotron-3-Super-120B-A12B** (NVIDIA, FP8)  
  - Leads all publishable results per KHA-255 model priority
  - 120B total / 12B active MoE; native 1M context; leads RULER 128K (0.917)
  - Snapshot size: ~5.3 GB — best demo of constant-size property at scale
  - Run all 10 conversations on both modes

### Full run — secondary (time/budget permitting)
- **Qwen3-Coder-Next** (~75B, Alibaba, FP8)  
  - Cross-vendor story; GLA architecture proves Engram works beyond Mamba2
  - Snapshot size: ~23.7 GB

### Deferred (not this session)
- Granite-4.0-H-small (32B) — edge story, local hardware only
- Any Omni, Cascade, or Mamba-3 variants
- Any model not listed above

---

## Context Length Plan

LoCoMo conversations range from ~12K to ~45K tokens (plain text, no images).

| Model | Max context | Policy |
|-------|-------------|--------|
| Granite-4.0-H-tiny | 32K | Truncate to most recent 28K turns if needed |
| Nemotron-3-Super-120B | 1M native | No truncation required |
| Qwen3-Coder-Next | 128K | Truncate to most recent 120K if needed |

**Truncation policy**: truncate from the oldest session forward, preserving the most recent sessions. Log truncated conversation IDs and counts in the output JSON.

---

## Conditions

| Condition | `--mode` | Snapshot ops | Notes |
|-----------|----------|--------------|-------|
| Stateless baseline | `stateless` | None | Context injected fresh per question |
| Engram | `engram` | save after full context load, restore per QA | Exercises KHA-348 fix |

Run both conditions on the same model back-to-back. Order: stateless first (establishes baseline before engram mode touches snapshot state).

---

## Success Criteria

### Green — proceed to full run
Smoke passes if all of:
- Harness completes 1 conversation without crash
- Stateless F1 on conv-26 (199 questions) ≥ 0.10 (minimal sanity — model is answering)
- Engram F1 on conv-26 ≥ stateless F1 − 0.05
- At least one restore latency measurement recorded

### Red — STOP and escalate
Stop immediately if any of:
- **Engram F1 matches stateless within noise (<0.01 difference) AND both are near-zero**  
  This pattern indicates KHA-348 is regressed — the snapshot restore is not providing context.
- Any `restore_snapshot` call returns `success: false`
- Server crashes or OOM during context loading
- Harness throws unhandled exception

### Anomalous — flag but continue with caution
- Engram F1 > stateless F1 by more than 0.10 (unexpected; check for scoring bug)
- Restore latency > 500ms mean (acceptable for this run but flag for optimization)
- Category 5 (adversarial) score < 0.20 on either mode (model not following "no info" instruction)

---

## Output Destination

All results written to:

```
s3://engram/benchmark-results/locomo/
  locomo_stateless_<model>_<timestamp>.json
  locomo_stateless_<model>_<timestamp>_summary.csv
  locomo_engram_<model>_<timestamp>.json
  locomo_engram_<model>_<timestamp>_summary.csv
```

---

## Estimated GPU Hours and Cost

Estimates assume OVH H100 PCIe 80GB @ ~$2.49/hr (OVH public pricing, subject to change).

| Condition | Model | Conversations | Est. hours | Est. cost |
|-----------|-------|--------------|------------|-----------|
| Smoke (stateless + engram) | Granite-4.0-H-tiny | 1 | 0.5 | ~$1.25 |
| Full stateless | Nemotron-120B FP8 | 10 | 4–6 | ~$10–15 |
| Full engram | Nemotron-120B FP8 | 10 | 5–8 | ~$12–20 |
| Full stateless | Qwen3-75B FP8 | 10 | 3–5 | ~$7–12 |
| Full engram | Qwen3-75B FP8 | 10 | 4–6 | ~$10–15 |
| **Re-run budget (1x)** | Both | 10 each | 10–14 | ~$25–35 |
| **Total (smoke + primary + secondary + reruns)** | | | **~30–45** | **~$75–110** |

Well under $1,000 with significant margin. If secondary model is skipped: ~$50–60.

---

## Pre-flight Gates (run before smoke)

```bash
# 1. Verify KHA-348 fix is in git history
python3 scripts/benchmarks/preflight_kha348.py
# Exit 0 = proceed; Exit 1 = STOP, fix not present

# 2. Verify repo is clean and within 5 commits of origin/main
bash scripts/benchmarks/preflight_base.sh
# Exit 0 = proceed; Exit 1 = STOP, sync first
```

Both must exit 0 before smoke run begins.

---

## Escalation Rule

If engram mode matches stateless mode within noise (|engram_f1 - stateless_f1| < 0.01) on the smoke **and** both scores are within normal range (not trivially near-zero), stop the benchmark, file a regression ticket against KHA-348, and do not run the full suite. The fix must be re-investigated before results are meaningful.

---

## Notes

- Images are not available in the LoCoMo dataset. Turns with `img_url` are passed with `blip_caption` as text substitute. This is consistent with other LLM evals on this dataset.
- Dataset source: `s3://engram/datasets/locomo/v1/locomo10.json`
- Harness repo: `feat/kha-255-locomo-harness-prep` branch; do not merge until smoke validates end-to-end.
