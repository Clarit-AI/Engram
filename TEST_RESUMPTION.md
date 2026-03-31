# Test Resumption Guide

## Status: Bug Fixed — Ready to Resume

The model config bug (`architectures: None` crash) was fixed. All phase plans and docs synced to gcloud.

---

## Available Models (on gcloud)

| Model | Path | Use When |
|-------|------|----------|
| granite-4.0-h-tiny | `/home/jeanclawdai/models/granite-4.0-h-tiny` | Primary — try first |
| Nemotron-4B | `/home/jeanclawdai/models/NVIDIA-Nemotron-3-Nano-4B-BF16` | Granite OOMs on V100 |
| Granite-Q4 | `/home/jeanclawdai/models/granite-4.0-h-tiny-gguf/granite-4.0-h-tiny-Q4_K_M.gguf` | Quantized comparison pass (args TBD) |

---

## Phase Execution Order

### Already Completed

| Phase | Result | Notes |
|-------|--------|-------|
| 00 | FAIL (2 fixture bugs) | 10/12 passed; 2 radix cache fixture bugs remain |
| 02 | PASS | MambaPool unit tests |
| 03 | PASS | 6 gauntlet tests |
| 05 | PASS | Mamba2Metadata unit tests |

### Not Yet Run (blocked by model config bug)

| Phase | Type | Notes |
|-------|------|-------|
| **01** | Server-based | Stateless inference baseline — **start here** |
| 04 | Server-based | Live server integration (no_buffer) |
| 06 | Server-based | extra_buffer strategy |
| 07 | Server-based | Snapshot system |
| 08 | Server-based | Gauntlet/stress tests |

---

## Quick Start Commands

### SSH to gcloud
```bash
gcloud compute ssh --zone "asia-east1-c" "sglang-test-v100-20260325-230245" --project "gen-lang-client-0471830999"
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
```

### Using Granite (default)
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
source test/phases/config.sh
```

### Using Nemotron (if granite OOMs)
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
source test/phases/config.sh
export MODEL_PATH=$NEMOTRON_MODEL_PATH
export MODEL_NAME=$NEMOTRON_MODEL_NAME
```

### Run Phase 01
```bash
# Start server
python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 --disable-radix-cache &

# Run tests
python -m pytest test/registered/radix_cache/test_mamba_baseline_inference.py -v
```

---

## Key Docs
- Phase plans: `test/phases/phase-0X-*.md`
- Results: `test/phases/results/`
- Config: `test/phases/config.sh`
- Codemap: `test/phases/codemap.md`

## Known Issues
1. **Phase 00 fixture bugs** — `test_full_cache_eviction` and `test_mamba_branching_seqlen` in comprehensive suite have fixture issues (separate from model config bug)
2. **Granite Q4 args** — server args for GGUF (`--dtype q4`?) are TBD — verify when testing
