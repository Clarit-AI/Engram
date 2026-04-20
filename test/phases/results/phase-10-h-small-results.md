# Phase 10 Results: granite-4.0-h-small

**Date**: 2026-04-01

## Model Info
- **Architecture**: GraniteMoeHybridForCausalLM
- **Parameters**: 32B dense hybrid (40 layers: 36 mamba + 4 attention)
- **MoE**: 72 experts, 10 active per token
- **Context**: 131072 (limited to 4096 for testing)
- **GPU Memory**: ~70GB VRAM, 80GB A100
- **Disk**: 61GB model weights

## Test Results

### Test 1: Basic Inference
- **Status**: PASS
- Response: "4." for 2+2
- Latency: 0.186s

### Test 2: Multi-turn (10 turns)
- **Status**: PASS
- All turns completed successfully
- Model remembered "Alice" across turns
- Latency range: 0.247s - 3.314s (increases with context)

### Test 3: Snapshot Save
- **Status**: PASS
- Save from WARM tier: 0.153s
- Snapshot ID format: {conversation_id}-t0

### Test 3b: Snapshot Restore (stateful generation)
- **Status**: FAIL (timeout)
- Root cause: Deferred output path returns None, HTTP future never resolved
- Architectural issue, not model-specific

### Test 3d: Multiple Snapshot Saves
- **Status**: PASS
- 5/5 snapshots saved successfully

### Test A: Sequential Conversations (20)
- **Status**: FAIL (1/20 snapshots saved)
- WARM tier LRU evicts states before manual save
- Only the most recently completed conversation has WARM state

### Test B: Multi-turn Single Conversation (8 turns)
- **Status**: PASS
- All 8 turns and snapshots completed
- Model consistently responded

### Test C: Rapid Fire (100 requests)
- **Status**: PASS
- 100/100 successful
- Avg latency: 0.172s
- GPU delta: +0MB, RSS delta: +9MB (no leak)

### Test D: Long Context (2K tokens)
- **Status**: PASS
- 2015 prompt tokens handled
- Snapshot saved successfully

### Test E: Snapshot Directory
- **Status**: PASS
- 182 conversation directories
- 17 safetensors files
- 2.5GB total snapshot storage

## Resource Summary
| Metric | Start | End | Delta |
|--------|-------|-----|-------|
| GPU VRAM | 70,244 MB | 70,500 MB | +256 MB |
| Process RSS | 14,647 MB | 14,668 MB | +21 MB |
| Snapshot Storage | 0 MB | 2,478 MB | +2,478 MB |

## Memory Leak Assessment
- **GPU**: No leak detected. +256MB across ~150+ requests is within normal variance.
- **RSS**: No leak detected. +21MB after warmup is excellent.
- **Snapshots**: Growing as expected (each snapshot ~150MB for 36 mamba layers).

## Key Findings

1. **Model loads and serves correctly** with `--context-length 4096 --mem-fraction-static 0.85`
2. **Snapshot save works reliably** for WARM tier hits
3. **Snapshot restore (stateful gen) has timeout bug** — deferred output path not connected to HTTP future
4. **No memory leaks detected** across 150+ requests
5. **Pure Mamba2 models incompatible** with SGLang (Mamba2ForCausalLM lacks attention backend support)
6. **granite-4.0-h-small is ~5x larger** than granite-4.0-h-tiny but uses similar VRAM per request
7. **Snapshot size per conversation**: ~150MB (36 mamba layers, bfloat16)

## Files
- Test script: test/phases/phase-10-h-small-test.py
- Detailed results: test/phases/results/phase-10-logs/h-small-detailed-20260330_025425.json
- Baseline results: test/phases/results/phase-10-logs/h-small-test-20260330_024527.json
