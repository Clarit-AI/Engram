# Phase 10 Results: Nemotron-Cascade-2-30B-A3B

**Date**: 2026-04-01

## Model Info
- **Architecture**: NemotronHForCausalLM (native SGLang support)
- **Parameters**: 30B total / 3B active (MoE)
- **Layers**: 52 (hybrid Mamba+MoE+Attention)
- **Hybrid pattern**: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
  - M = Mamba layers
  - E = MoE Expert layers (128 experts, 6 active)
  - * = Attention layers
- **Context**: 262144 (limited to 2048 for testing)
- **GPU Memory**: ~70GB VRAM on A100 80GB

## Test Results

| Test | Status | Details |
|------|--------|---------|
| Basic Inference | PASS | "4." for 2+2, 0.109s |
| Multi-turn (5 turns) | PASS | All 5 snapshots saved successfully |
| Rapid Fire (50) | PASS | 50/50, avg 0.059s, 0 GPU/RSS delta |
| Medium Context (711 tokens) | PASS | Snapshot saved |
| Snapshot Directory | PASS | 6 safetensors, 280.9MB |

## Resource Summary
| Metric | Start | End | Delta |
|--------|-------|-----|-------|
| GPU VRAM | 70,074 MB | 70,206 MB | +132 MB |
| Process RSS | 4,948 MB | 5,271 MB | +323 MB |
| Snapshot Storage | 0 MB | 280.9 MB | +280.9 MB |

## Memory Leak Assessment
- **GPU**: No leak. +132MB across ~60 requests, stable during rapid fire.
- **RSS**: +323MB growth during multi-turn and context tests. Acceptable.
- **Rapid fire**: 0 GPU delta, 0 RSS delta. No incremental leak.

## Performance Comparison with granite-4.0-h-small

| Metric | granite-4.0-h-small | Nemotron-30B | Winner |
|--------|---------------------|--------------|--------|
| Basic latency | 0.186s | 0.109s | Nemotron |
| Rapid fire avg | 0.172s | 0.059s | Nemotron (3x) |
| Snapshot save rate | 1/20 (5%) | 5/5 (100%) | Nemotron |
| GPU stability | +256MB | +132MB | Nemotron |
| RSS stability | +21MB | +323MB | Granite |
| Snapshot size | ~150MB each | ~47MB each | Nemotron (3x smaller) |
| Test score | 4/5 | 5/5 | Nemotron |

## Key Findings

1. **Nemotron outperforms Granite** in latency, snapshot reliability, and GPU stability
2. **MoE architecture** gives 3x faster inference (only 6/128 experts active)
3. **Snapshot saves 100% reliable** vs 5% for Granite — WARM tier retention better
4. **Per-snapshot size smaller**: ~47MB vs ~150MB (fewer total mamba layers in MoE config)
5. **No memory leaks detected** in either model
6. **Snapshot restore (stateful gen)** still hangs — same architectural bug
