---
name: engram-sglang
description: Engram (formerly sglang-mamba) — persistent stateful inference for Mamba/SSM hybrid models on SGLang. Snapshot save/restore, 3-tier memory hierarchy, conversation_id tracking, tested across Granite, Nemotron, and Qwen architectures on H200. Use when working on the Engram repo (formerly sglang-mamba), debugging snapshot pipeline, deploying Mamba models, or running test phases.
version: 2.1.0
author: Clarit.ai
license: Apache-2.0
tags: [Engram, SGLang, Mamba, Mamba2, SSM, State Snapshots, Persistent State, Stateful Inference, Inference Serving, 3-Tier Memory, Conversation Tracking, Edge AI, RadixAttention, MambaRadixCache, Hybrid Models, GLA, DeltaNet]
dependencies: [sglang, torch, transformers, safetensors, flashinfer]
---

# Engram (formerly sglang-mamba) — Persistent Stateful Inference

> **Agent Note**: This repo is **Engram**, formerly **sglang-mamba**. It is a fork of SGLang that adds persistent recurrent-state infrastructure for Mamba-family and related models. The old name still appears in local directory names, git history, some docs, and test artifacts. Treat `Engram` and `sglang-mamba` as the same project unless a document explicitly says otherwise.

## When to use this skill

Use this skill when:
- Working in or on the Engram repository
- Debugging the snapshot save/restore pipeline
- Deploying Mamba, hybrid, or other recurrent-state models with persistent state
- Running or writing test phases in `test/phases/`
- Working with `MambaRadixCache`, `MambaPool`, `TierManager`, `StateHealthMonitor`, or `SnapshotManager`
- Investigating snapshot metadata, state health, tiering, or restore behavior
- Configuring Mamba-specific server flags

## Key facts

- **Repo**: `https://github.com/Clarit-AI/Engram`
- **Upstream**: `https://github.com/sgl-project/sglang`
- **Primary feature flag**: `--enable-snapshot-persistence` (not `--enable-mamba-snapshots`)
- **Standard API compatibility**: Engram preserves standard SGLang serving behavior and OpenAI-compatible generation endpoints, then adds snapshot endpoints on top
- **Snapshot format**: safetensors with atomic tmp-rename writes
- **Snapshot size**: roughly constant across long contexts in current validation runs
- **Restore tiers**: ACTIVE/VRAM, WARM/host RAM, COLD/disk
- **GPU requirement**: sm75+ for Mamba2 CUDA kernels; older GPUs are limited to narrower validation paths

## Doc priority

When orienting yourself:
- Start with `docs/stateful_mamba/api_guide.md` if it exists on the current branch
- Otherwise use `docs/stateful_mamba/http_api_spec.md` for the current wire-level contract
- Treat older snapshot docs as potentially stale unless they explicitly say they are current

## Quick start

```bash
# Clone the Engram fork (not upstream SGLang)
git clone https://github.com/Clarit-AI/Engram.git
cd Engram
pip install -e "python[all]"

# Some local checkouts may still be named sglang-mamba.
# That is normal for older worktrees and docs.

python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --mamba-scheduler-strategy no_buffer \
  --port 30000
```

## Server flags (Engram-specific)

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-snapshot-persistence` | Enable snapshot persistence | `false` |
| `--snapshot-dir <path>` | Directory for snapshot storage | auto/default when enabled |
| `--mamba-scheduler-strategy <strategy>` | Mamba scheduler strategy | `no_buffer` |
| `--snapshot-retention-count <N>` | Max snapshots retained per conversation | `10` |
| `--snapshot-trigger-policy <policy>` | `every_turn`, `every_n_turns`, `on_tool_call`, `manual_only` | `every_turn` |
| `--enable-memory-tiers` | Enable VRAM -> RAM -> disk tiering | repo default |
| `--max-warm-conversations <N>` | Max conversations kept in warm RAM tier | `100` |
| `--enable-cross-session-refs` | Allow cross-conversation restore references | repo default |
| `--snapshot-health-check-interval <N>` | Run state health checks every N snapshots | `0` (disabled) |
| `--snapshot-health-failure-policy <policy>` | `log_and_continue` or `skip_snapshot` | `log_and_continue` |

## Public APIs

Engram keeps the usual SGLang/OpenAI-shaped generation flows. Snapshot support is additive, not a replacement API.

### HTTP endpoints

The snapshot endpoints are all `POST` routes:
- `POST /save_snapshot`
- `POST /list_snapshots`
- `POST /get_snapshot_info`
- `POST /restore_snapshot`
- `POST /delete_snapshot`

Use `docs/stateful_mamba/http_api_spec.md` for the exact request/response contract on this branch.

Important semantics:
- `rid` identifies the live request to save into or restore into
- `conversation_id` is the durable grouping key across turns
- `turn_number` and `branch_name` are the selectors for a specific snapshot
- `conversation_id` alone is not a complete restore strategy for all workflows
- `restore_snapshot` has two real modes:
  - restore state into an existing/live request path
  - restore and immediately continue generation with `create_new_request=true`

Representative save example:

```bash
curl http://localhost:30000/save_snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "rid": "req-123",
    "conversation_id": "session-1",
    "turn_number": 1
  }'
```

Representative list example:

```bash
curl http://localhost:30000/list_snapshots \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-1"
  }'
```

Representative delete example:

```bash
curl http://localhost:30000/delete_snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-1",
    "turn_number": 1
  }'
```

### Python surfaces

There are two public Python entry points:

1. Direct `ProgramState` helpers on `s`:
   - `s.save_snapshot(...)`
   - `s.list_snapshots(...)`
   - `s.get_snapshot_info(...)`
   - `s.restore_snapshot(...)`
   - `s.delete_snapshot(...)`

2. High-level endpoint-backed manager:

```python
import sglang as sgl

runtime = sgl.Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,
)

sm = sgl.SnapshotManager(runtime.endpoint)
snapshots = sm.list_conversation("conv_123")
info = sm.get_info("conv_123", turn_number=5)
```

Do not document or assume `SnapshotManager(engine)` or older snapshot-id-centric flows as the current public contract.

## Architecture

```
OpenAI-compatible and standard SGLang generation APIs
    |
    v
SGLang serving engine (scheduler, batching, TP/PP, tokenizer manager)
    |
    v
Engram additions:
    ├── Snapshot HTTP handlers and request/response structs
    ├── SnapshotManager + mamba_snapshot save/restore plumbing
    ├── Snapshot policies and conversation tracking
    ├── TierManager + MambaHostPool for ACTIVE/WARM/COLD storage
    ├── MambaRadixCache + memory pools for recurrent-state handling
    └── Optional StateHealthMonitor for structural + norm-based checks
```

## Key source files

| File | Purpose |
|------|---------|
| `python/sglang/srt/entrypoints/http_server.py` | Snapshot HTTP routes |
| `python/sglang/srt/managers/io_struct.py` | Snapshot request/response models |
| `python/sglang/srt/managers/scheduler.py` | Snapshot hooks, tiering, health-check integration |
| `python/sglang/srt/snapshot/mamba_snapshot.py` | State extraction, validation, serialization |
| `python/sglang/srt/snapshot/tier_manager.py` | VRAM/RAM/disk tier management |
| `python/sglang/srt/snapshot/snapshot_policy.py` | Snapshot timing and retention behavior |
| `python/sglang/srt/snapshot/conversation_tracker.py` | Conversation activity and tier transitions |
| `python/sglang/srt/snapshot/mamba_host_pool.py` | Warm-tier host RAM staging |
| `python/sglang/srt/snapshot/state_health.py` | Optional state health monitoring |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | Recurrent-state cache behavior |
| `python/sglang/srt/mem_cache/memory_pool.py` | GPU memory pool management |
| `python/sglang/lang/interpreter.py` | `ProgramState` snapshot helpers |
| `python/sglang/snapshot.py` | Public `SnapshotManager(runtime.endpoint)` API |

## Model and compatibility notes

| Model | Status | Notes |
|-------|--------|-------|
| Granite 4.0-H-tiny | PASS | Primary validation model |
| Granite 4.0-H-small | PASS | Base model; use `/generate`, not chat-completions |
| Nemotron-Cascade-2-30B | PASS | Hybrid/MoE validation path |
| Nemotron-3-Super-120B FP8 | PASS | High-end H200 validation run |
| Qwen3-Coder-Next FP8 | PASS | Important proof that the snapshot path is not Mamba2-only |
| Mamba-Codestral-7B | INCOMPATIBLE today | Pure Mamba2 model class exists, but full runtime compatibility is still incomplete |

Critical compatibility finding:
- The snapshot stack works for models whose recurrent state flows through SGLang's Mamba-style cache path, not only classic Mamba2 hybrids

## Current validation snapshot

Tests are organized in `test/phases/`, with current rollup in `test/phases/results/INDEX.md`.

| Phase | Status |
|-------|--------|
| 0 | FAIL (test bugs, environment still usable) |
| 1 | PASS |
| 2 | PASS |
| 3 | PASS |
| 4 | PASS |
| 5 | PASS |
| 6 | PARTIAL |
| 7 | PASS |
| 8 | PASS |
| 9 | PASS |
| 10a-10f | mixed but mostly PASS; see results index |
| Compat runs | Granite-small PASS, Nemotron-3-Super PASS, Qwen3-Coder-Next PASS |

Use the results index, not this skill, for the exact up-to-the-minute pass/fail matrix.

## Operational gotchas

- `Granite 4.0-H-small` is a base model and should be tested through `/generate`, not `/v1/chat/completions`
- `Qwen3-Coder-Next FP8` requires `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- The web UI resends full history, so it is a poor proof harness for true snapshot persistence
- `conversation_id` is the durable grouping key, but `rid` is still the live request handle for some snapshot operations
- `restore_snapshot` behavior differs between restore-only flows and restore-plus-generate flows

## State health and poisoning guidance

State-poisoning risk is still real, but the repo no longer has only structural checks.

Current state:
- structural validation exists in the snapshot path
- `StateHealthMonitor` exists for norm-based anomaly detection
- scheduler integration exists
- health monitoring is configurable and may be disabled in a given deployment

Implication for agents:
- do not assume "no value-level checks exist"
- do assume health monitoring is partial, policy-driven, and still an active correctness concern for persistent state systems

## 3-tier memory details

Tier transitions are driven by server configuration:
- ACTIVE -> WARM after `conversation_active_timeout`
- WARM -> COLD after `conversation_warm_timeout`
- COLD retention controlled by `conversation_cold_retention`

`TierManager` handles promotion and eviction. `MambaHostPool` backs the warm tier. Disk snapshots use safetensors and atomic file writes.

## Snapshot metadata

Common metadata fields include:
- `conversation_id`
- `turn_number`
- `branch_name`
- `timestamp`
- `token_count`
- `model_name`
- `fill_ids`

Do not assume every response includes a rigid or exhaustive metadata schema; consult the live spec for exact behavior.

## References

- **[Structured Generation Guide](references/structured-generation.md)** — upstream SGLang structured output background
- **[RadixAttention Deep Dive](references/radix-attention.md)** — upstream prefix-caching background
- **[Production Deployment](references/deployment.md)** — upstream deployment background
