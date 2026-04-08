# Phase 11a: Codestral Mamba 7B — Pure Mamba2 Baseline Compatibility

**Linear:** KHA-185
**Branch:** `feat/pure-mamba2-model-class`
**Model:** `mistralai/Mamba-Codestral-7B-v0.1`
**Architecture:** `Mamba2ForCausalLM` — 64 layers, all SSM, zero attention
**VRAM:** ~14.6 GB (bfloat16) — fits on A100 80GB

## Purpose

Validate that pure Mamba2 models (no attention layers) can load and run inference through SGLang's hybrid model infrastructure. This is the first pure-SSM model tested — all previous phases used hybrid architectures.

## Prerequisites

- Phase 0 PASS (environment verified)
- `feat/pure-mamba2-model-class` branch checked out and installed
- Codestral Mamba 7B downloaded: `huggingface-cli download mistralai/Mamba-Codestral-7B-v0.1`

## Config

```bash
export CODESTRAL_MODEL_PATH="mistralai/Mamba-Codestral-7B-v0.1"
# Or local path if downloaded:
# export CODESTRAL_MODEL_PATH="/mnt/models/Mamba-Codestral-7B-v0.1"
export CODESTRAL_PORT=30100
```

---

## Gate 1: Server Starts

**Goal:** Model loads without crashes. Config is parsed, model weights load, memory pools initialize.

```bash
python -m sglang.launch_server \
  --model-path $CODESTRAL_MODEL_PATH \
  --disable-radix-cache \
  --port $CODESTRAL_PORT \
  2>&1 | tee results/compat-codestral-gate1.log
```

**PASS criteria:**
- Server reaches "The server is fired up" message
- No `IndexError`, `AttributeError`, or `ZeroDivisionError`
- Log shows `mamba2_config` recognized (not None)
- MambaPool initialized with 64 layers

**FAIL actions:**
- Check for `architectures[0]` crash → verify model_config.py guards
- Check for `cell_size == 0` crash → verify kv_cache_mixin.py fix
- Check for missing `Mamba2Config` → verify configs/__init__.py export

---

## Gate 2: Single Inference

**Goal:** Basic text generation produces coherent output.

```bash
# Wait for server to be ready
python -c "
import requests, json
resp = requests.post('http://localhost:$CODESTRAL_PORT/v1/completions', json={
    'model': '$CODESTRAL_MODEL_PATH',
    'prompt': 'def fibonacci(n):',
    'max_tokens': 100,
    'temperature': 0.0,
})
print(json.dumps(resp.json(), indent=2))
"
```

**PASS criteria:**
- HTTP 200 response
- Generated text is syntactically valid Python (or at least coherent)
- No CUDA errors or NaN in output logits

---

## Gate 3: Snapshot Save/Load (with persistence enabled)

**Goal:** Mamba state can be persisted and restored for pure-SSM models.

```bash
# Restart server with snapshot persistence
python -m sglang.launch_server \
  --model-path $CODESTRAL_MODEL_PATH \
  --enable-snapshot-persistence \
  --snapshot-dir /tmp/codestral-snapshots \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port $CODESTRAL_PORT \
  2>&1 | tee results/compat-codestral-gate3.log
```

```bash
# Send a conversation to trigger snapshot save
python -c "
import requests, json
# Turn 1
resp = requests.post('http://localhost:$CODESTRAL_PORT/v1/chat/completions', json={
    'model': '$CODESTRAL_MODEL_PATH',
    'messages': [{'role': 'user', 'content': 'Write a Python class for a binary search tree'}],
    'max_tokens': 200,
    'temperature': 0.0,
})
print('Turn 1:', resp.status_code)
print(json.dumps(resp.json()['choices'][0]['message'], indent=2))
"
```

**PASS criteria:**
- Snapshot files appear in `/tmp/codestral-snapshots/`
- No `ValueError` from snapshot validation (Tier 1 checks pass)
- Snapshot metadata has correct `model_name`

---

## Gate 4: Multi-Turn State Restoration

**Goal:** State restoration across turns produces consistent output.

```bash
# Turn 2 referencing Turn 1 context
python -c "
import requests, json
resp = requests.post('http://localhost:$CODESTRAL_PORT/v1/chat/completions', json={
    'model': '$CODESTRAL_MODEL_PATH',
    'messages': [
        {'role': 'user', 'content': 'Write a Python class for a binary search tree'},
        {'role': 'assistant', 'content': '...'},  # paste Turn 1 output
        {'role': 'user', 'content': 'Now add an inorder traversal method'},
    ],
    'max_tokens': 200,
    'temperature': 0.0,
})
print('Turn 2:', resp.status_code)
print(json.dumps(resp.json()['choices'][0]['message'], indent=2))
"
```

**PASS criteria:**
- Turn 2 response references Turn 1 context correctly
- No state corruption errors
- Snapshot restore log shows successful warm-tier load

---

## Gate 5: Regression — Existing Hybrid Model

**Goal:** Granite 4.0-H-tiny (hybrid) still works after the changes.

```bash
source test/phases/config.sh
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --enable-snapshot-persistence \
  --snapshot-dir $SNAPSHOT_DIR \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port $SERVER_PORT \
  2>&1 | tee results/compat-codestral-regression.log
```

Run a quick inference to confirm no regressions:

```bash
python -c "
import requests
resp = requests.post('http://localhost:$SERVER_PORT/v1/completions', json={
    'model': '$MODEL_PATH',
    'prompt': 'Hello, world!',
    'max_tokens': 50,
})
print(resp.status_code, resp.json()['choices'][0]['text'][:100])
"
```

**PASS criteria:**
- Granite hybrid model loads and generates as before
- No new warnings or errors in logs

---

## Results Template

| Gate | Description | Result | Notes |
|------|-------------|--------|-------|
| 1 | Server starts | | |
| 2 | Single inference | | |
| 3 | Snapshot save/load | | |
| 4 | Multi-turn restoration | | |
| 5 | Hybrid regression | | |

**Overall:** PENDING

**Tester:** ___
**Date:** ___
**GPU:** ___
