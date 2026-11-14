---
name: engram-sglang
description: Engram (formerly sglang-mamba), snapshot save and restore for Mamba and hybrid recurrent-state models on SGLang. Covers the snapshot HTTP and Python APIs, server flags, the 3-tier memory hierarchy, where the snapshot code lives, and how to check what currently passes. Use when working on the Engram repo (formerly sglang-mamba), debugging the snapshot pipeline, or launching a Mamba model with snapshots.
version: 2.2.0
author: Clarit.ai
license: Apache-2.0
tags: [Engram, SGLang, Mamba, Mamba2, SSM, State Snapshots, Persistent State, Stateful Inference, Inference Serving, 3-Tier Memory, Conversation Tracking, Edge AI, RadixAttention, MambaRadixCache, Hybrid Models, GLA, DeltaNet]
dependencies: [sglang, torch, transformers, safetensors, flashinfer]
---

# Engram (formerly sglang-mamba): Persistent Stateful Inference

> **Agent Note**: This repo is **Engram**, formerly **sglang-mamba**. It is a fork of SGLang that adds persistent recurrent-state infrastructure for Mamba-family and related models. The old name still appears in local directory names, git history, some docs, and test artifacts. Treat `Engram` and `sglang-mamba` as the same project unless a document explicitly says otherwise.

## When to use this skill

Use this skill when:
- Working in or on the Engram repository
- Debugging the snapshot save/restore pipeline
- Launching Mamba, hybrid, or other recurrent-state models with persistent state
- Working with `MambaRadixCache`, `MambaPool`, `TierManager`, `StateHealthMonitor`, or `SnapshotManager`
- Investigating snapshot metadata, state health, tiering, or restore behavior
- Configuring Mamba-specific server flags

## This skill holds no status

Do not use this skill to answer "does X work" or "which release is good". It was last revised on 2026-11-14 and goes stale like any document.

- **What passes** is recorded in `results.tsv` in the release-test harness (`Clarit-AI/engram-harness`, maintainers only). Each row names the commit SHA, tag, model, GPU, package cutoff date and log folder.
- **What exists** is recorded in git. Engram releases are tagged `engram-vX.Y.Z`. A bare `vX.Y.Z` tag is an upstream SGLang tag.
- **Policy** is in `AGENTS.md` under "Rules That Come First": Engram pins to SGLang release tags, does not merge `upstream/main`, and never pushes to upstream.

For orientation only: on 2026-11-14 the harness had passing rows for `engram-v0.5.1`, `engram-v0.5.2` and `main` at `7e34b5c6cf`, on Granite 4.0-H-tiny and Mamba-Codestral-7B, on one NVIDIA L4, installed with packages dated to each ref. Read the file before repeating that sentence.

## Key facts

- **Repo**: `https://github.com/Clarit-AI/Engram`
- **Upstream**: `https://github.com/sgl-project/sglang` (read-only for Engram work)
- **Primary feature flag**: `--enable-snapshot-persistence` (not `--enable-mamba-snapshots`, which does not exist)
- **Standard API compatibility**: Engram preserves standard SGLang serving behavior and OpenAI-compatible generation endpoints, then adds snapshot endpoints on top
- **Snapshot format**: safetensors with atomic tmp-rename writes
- **Snapshot size**: roughly constant across context lengths in the spring 2026 runs, because the state is a fixed-size recurrent representation
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
```

Three things the install line does not tell you:

1. **A fresh install of `main` at `7e34b5c6cf` fails at import.** `python/pyproject.toml` requires `kernels` with no version, and `kernels` 0.18.3 raises while `transformers` 5.8.1 imports. The passing harness rows were installed with `kernels` 0.14.1. Check whether the pin has landed on your ref before debugging anything else.
2. **`sgl-kernel` must match the commit.** The harness installs an `sgl-kernel` wheel built from the ref's own `sgl-kernel/` tree, and the passing rows were produced that way. A wheel from PyPI or from another GPU architecture may not load.
3. **Refs from `engram-v0.4.1` on compile a Rust extension** (`rust/sglang-grpc`) during the source install. They need Rust 1.90 and `protoc` on the machine.

Launch flags the harness uses, shown here for the pure Mamba2 gate model:

```bash
python -m sglang.launch_server \
  --model-path mistralai/Mamba-Codestral-7B-v0.1 \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --snapshot-trigger-policy every_turn \
  --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --port 30000
```

Codestral does not start without `--disable-piecewise-cuda-graph`. Hybrid models such as Granite 4.0-H-tiny start without it.

## Server flags (Engram-specific)

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-snapshot-persistence` | Enable snapshot persistence | `false` |
| `--snapshot-dir <path>` | Directory for snapshot storage | `./sglang_snapshots` when enabled |
| `--mamba-scheduler-strategy <strategy>` | Mamba scheduler strategy (`auto`, `no_buffer`, `extra_buffer`) | `auto` |
| `--snapshot-retention-count <N>` | Max snapshots retained per conversation | `10` |
| `--snapshot-trigger-policy <policy>` | `every_turn`, `every_n_turns`, `on_tool_call`, `manual_only` | `every_turn` |
| `--snapshot-min-interval-seconds <S>` | Minimum time between automatic snapshots of one conversation | `1.0` |
| `--enable-memory-tiers` / `--no-enable-memory-tiers` | VRAM, then RAM, then disk tiering | on |
| `--max-warm-conversations <N>` | Max conversations kept in warm RAM tier | `100` |
| `--enable-cross-session-refs` | Allow cross-conversation restore references | on |
| `--snapshot-health-check-interval <N>` | Run state health checks every N snapshots | `0` (disabled) |
| `--snapshot-health-failure-policy <policy>` | `log_and_continue` or `skip_snapshot` | `log_and_continue` |

Defaults are read from `python/sglang/srt/server_args.py`. If this table and that file disagree, the file is right.

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
- The snapshot endpoints answer HTTP 200 with `"success": false` for some failures. Check `success`, not only the status code.
- `restore_snapshot` has two real modes:
  - restore state for a later request under the same `rid` or `conversation_id`
  - restore and immediately continue generation with `create_new_request=true` and `continuation_ids`

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
| `python/sglang/srt/managers/scheduler_snapshot_handlers.py` | Save, list, info, restore and delete handlers; pending-restore registry |
| `python/sglang/srt/managers/scheduler_snapshot_system.py` | Snapshot system startup; the callback that takes automatic snapshots; tiering and health-check wiring |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | Where a finished request triggers the snapshot callback |
| `python/sglang/srt/managers/scheduler.py` | Thin delegators to the files above (the snapshot code moved out on 2026-05-20) |
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

## Models

**Mamba-Codestral-7B** (pure Mamba2) is the gate model: a release is judged on it. **Granite 4.0-H-tiny** was dropped as a product target in November 2026 and stays only as the small test fixture the older releases were built against.

The table below is a record from April 2026, at `engram-v0.2.0` on A100 and H200 hardware. The release-test harness has since run the first and third models, for restore equivalence only. It has measured none of the numbers in this table. Do not quote the table as current.

| Model | Arch | Recorded result | Token Reduction | Snapshot Size | Notes |
|-------|------|-----------------|-----------------|---------------|-------|
| Granite 4.0-H-tiny (4B) | Mamba2+Attn+MoE | PASS | 93.8% | 56MB (TP1), 14MB (TP4) | Test fixture only |
| Granite 4.0-H-small | Mamba2+Attn+MoE | PASS | n/a | n/a | Base model; use `/generate`, not chat-completions |
| Mamba-Codestral-7B | Pure Mamba2 | PASS | 67.1% | 260MB | Gate model; requires `--disable-cuda-graph --disable-piecewise-cuda-graph` |
| Qwen3-Next-80B-A3B-Thinking | Mamba2+Attn+MoE | PASS | 82.0% | 19MB (TP4) | 80B total/3B active |
| Nemotron-Cascade-2-30B | Mamba2+Attn+MoE | PASS | n/a | n/a | Hybrid/MoE validation path |
| Nemotron-3-Super-120B FP8 | Mamba2+Attn+MoE | BLOCKED | n/a | n/a | Requires SM89+ (H100/H200), fails on A100 |
| Qwen3-Coder-Next FP8 | Different hybrid | PASS | n/a | n/a | Recorded as proof that the snapshot path is not Mamba2-only |

Pure Mamba2 support (Codestral) is Engram's own. Stock SGLang 0.5.12 could not load Codestral in the harness's control run (`'Mamba2Config' object has no attribute 'num_attention_heads'`), and the upstream PR that offered the support (sgl-project/sglang#22327) was closed unmerged.

## Operational gotchas

- **A plain `/generate` after a restore uses the restored state only from PR #91 on** (KHA-390, merged 2026-05-23 as `f0b617c56c`). That fix is in `main` and not in the `engram-v0.5.2` tag. On the tag, use restore-and-continue (`create_new_request=true` with `continuation_ids`).
- `Granite 4.0-H-small` is a base model and should be tested through `/generate`, not `/v1/chat/completions`
- Codestral has no chat template, so `/v1/chat/completions` returns HTTP 400 for it. Use `/generate`.
- `Qwen3-Coder-Next FP8` requires `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- The web UI resends full history, so it is a poor proof harness for true snapshot persistence
- `conversation_id` is the durable grouping key, but `rid` is still the live request handle for some snapshot operations
- `restore_snapshot` behavior differs between restore-only flows and restore-plus-generate flows
- `test/phases/` and `test/phases/results/INDEX.md` are the March 2026 test program. They are history, not the current pass/fail record.

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
- ACTIVE to WARM after `conversation_active_timeout`
- WARM to COLD after `conversation_warm_timeout`
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

- **[Structured Generation Guide](references/structured-generation.md)**: upstream SGLang structured output background
- **[RadixAttention Deep Dive](references/radix-attention.md)**: upstream prefix-caching background
- **[Production Deployment](references/deployment.md)**: upstream deployment background
