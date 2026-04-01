<div align="center">

<img src="assets/engram-logo-banner.png" alt="Engram" width="400"></img>

### Stateful inference for Mamba SSM models

[![Built on SGLang](https://img.shields.io/badge/built%20on-SGLang-blue)](https://github.com/sgl-project/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](LICENSE)

Save, restore, and persist Mamba hidden state across sessions.
Zero token waste. Sub-millisecond restore. Constant-size snapshots.

[Quick Start](#quick-start) | [Benchmarks](#benchmarks) | [API Reference](#api-extensions) | [Architecture](#how-it-works)

</div>

---

## What is Engram?

Engram adds **persistent state infrastructure** to [SGLang](https://github.com/sgl-project/sglang), the high-performance LLM serving engine. It targets Mamba and Mamba2 hybrid models specifically, turning their hidden state from disposable inference overhead into a first-class memory asset.

Standard serving infrastructure throws away model state after every session. For transformer KV caches, that's a design trade-off. For Mamba SSMs, it's a design flaw — the hidden state *is* the model's compressed understanding of the conversation. Discarding it means re-processing every token from scratch on every turn.

Engram fixes this. Save a snapshot after any turn, restore it later in ~2ms, skip the entire prefill. The snapshot is constant-size regardless of context length — a 128K-token conversation restores just as fast as a 2K-token one.

## Built on SGLang

Engram is a maintained fork of [sgl-project/sglang](https://github.com/sgl-project/sglang), the serving engine that powers inference across 400,000+ GPUs worldwide. Everything SGLang does — RadixAttention, continuous batching, tensor parallelism, OpenAI-compatible APIs, broad model and hardware support — Engram inherits and stays current with.

The fork adds a focused set of extensions for Mamba SSM state persistence. Upstream SGLang compatibility is maintained through automated sync tooling that merges non-conflicting upstream changes twice weekly.

## Benchmarks

Validated on H200 across the full test suite (77/82 pass, 0 regressions):

| Metric | Result |
|--------|--------|
| **Token reduction** | 93.8% average across 271 requests, zero failures |
| **Snapshot restore** | ~2ms warm restore (vs. 6.5s full prefill at 128K context) |
| **Speedup at 128K** | ~3,250x faster than full prefill |
| **Snapshot size** | ~56MB constant, regardless of context length (2K–128K) |
| **Memory stability** | Zero leaks, stable VRAM under sustained load |
| **Snapshot size scaling** | Constant — 2K tokens and 128K tokens produce identical snapshot sizes |

### Tested Models

All models validated on H200 with stateful recall confirmed (PR #20).

| Model | Vendor | Params | Architecture | Stateful Recall | Notes |
|-------|--------|--------|-------------|-----------------|-------|
| Granite 4.0-H-tiny | IBM | 4B | Dense Mamba2 hybrid | ✓ PASS (4/4) | Primary validation model — full phases 0–10 |
| Granite 4.0-H-small | IBM | 32B | Dense Mamba2 hybrid | ✓ PASS | Base model (no chat template) — use `/v1/completions` |
| Nemotron-3-Super-120B-A12B | NVIDIA | 120B / 12B active | LatentMoE Mamba2 hybrid | ✓ PASS (4/4) | FP8; requires H200 (133 GB VRAM) |
| Nemotron-Cascade-2-30B | NVIDIA | 30B / 3B active | MoE Mamba2 hybrid | ✓ PASS | BF16 |
| Qwen3-Coder-Next | Alibaba | ~75B / 3.9B active | GLA + MoE hybrid | ✓ PASS (4/4) | FP8; set `SGLANG_ENABLE_JIT_DEEPGEMM=0` |
| Codestral Mamba 7B | Mistral | 7B | Pure Mamba2 | — Blocked | Needs native SGLang model class ([upstream issue](https://github.com/sgl-project/sglang/issues/7429)) |

## Quick Start

```bash
# Install
pip install -e "python/"

# Start server with snapshot persistence
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port 30000

# Chat normally via OpenAI-compatible API
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-4.0-h-tiny",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 256
  }'

# Save conversation state
curl http://localhost:30000/save_snapshot \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1"}'

# Restore it later — ~2ms, skip entire prefill
curl http://localhost:30000/restore_snapshot \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1"}'
```

## How It Works

Mamba SSM layers maintain a fixed-size hidden state that compresses the entire conversation history. Unlike transformer KV caches (which grow linearly with context), this state stays constant regardless of how long the conversation gets.

Engram adds four things to SGLang to exploit this property:

**Snapshot persistence** — Save and restore complete Mamba hidden state to disk. The snapshot captures the model's full compressed understanding of a conversation at any point, and can be restored in a future session without re-processing any of the original tokens.

**3-tier memory hierarchy** — VRAM, host RAM, and disk, with configurable promotion and eviction. Hot sessions stay in VRAM for sub-millisecond restore. Warm sessions sit in host RAM. Cold sessions persist on disk. The tier manager handles movement automatically.

**Retention policies** — Configurable rules for how long snapshots live at each tier, when they get promoted or evicted, and which sessions are prioritized. Think of it as memory management for conversations.

**Agent tool framework** — Built-in tools that let agents leverage persistent state: save checkpoints, branch conversations, restore prior context. REST and WebSocket interfaces for integration.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                 OpenAI-Compatible API            │
├─────────────────────────────────────────────────┤
│              SGLang Serving Engine               │
│    (scheduler, batching, tensor parallelism)     │
├──────────────┬──────────────────────────────────┤
│  Engram      │  Snapshot Manager                 │
│  Extensions  │  ├─ save / restore / list / delete│
│              │  ├─ Conversation Tracker           │
│              │  └─ Snapshot Hooks & Policies      │
│              ├──────────────────────────────────┤
│              │  Tier Manager                     │
│              │  ├─ VRAM  (hot, ~2ms restore)     │
│              │  ├─ RAM   (warm, ~10ms restore)   │
│              │  └─ Disk  (cold, ~50ms restore)   │
│              ├──────────────────────────────────┤
│              │  Mamba State Extensions            │
│              │  ├─ MambaRadixCache (dual LRU)    │
│              │  ├─ HybridReqToTokenPool          │
│              │  └─ Mamba2Metadata                │
├──────────────┴──────────────────────────────────┤
│              Mamba / Mamba2 Models                │
│    (pure SSM, dense hybrid, MoE hybrid)          │
└─────────────────────────────────────────────────┘
```

## API Extensions

Engram extends SGLang's API with snapshot management endpoints. All existing SGLang endpoints work unchanged.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/save_snapshot` | POST | Save current conversation state |
| `/restore_snapshot` | POST | Restore a previously saved state |
| `/list_snapshots` | GET | List all saved snapshots |
| `/delete_snapshot` | DELETE | Remove a saved snapshot |

### Server Flags

| Flag | Description |
|------|-------------|
| `--enable-snapshot-persistence` | Enable the snapshot system |
| `--snapshot-dir <path>` | Directory for snapshot storage |
| `--mamba-scheduler-strategy <strategy>` | Scheduler strategy (`no_buffer`, `extra_buffer`) |

## Key Source Files

| File | Purpose |
|------|---------|
| `python/sglang/srt/snapshot/mamba_snapshot.py` | Core snapshot save/load logic |
| `python/sglang/srt/snapshot/tier_manager.py` | 3-tier memory hierarchy (VRAM/RAM/disk) |
| `python/sglang/srt/snapshot/snapshot_policy.py` | Retention and trigger policies |
| `python/sglang/srt/snapshot/conversation_tracker.py` | Session state tracking |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | Dual-LRU cache with COW support |
| `python/sglang/srt/mem_cache/memory_pool.py` | Hybrid memory pools for Mamba state |
| `python/sglang/srt/agents/` | Agent tool framework |
| `python/sglang/snapshot.py` | Public API (SnapshotManager) |

## Development

```bash
# Run unit tests (no GPU required)
pytest test/sglang/snapshot/ -v
pytest python/sglang/test/srt/test_mamba_metadata.py -v
pytest python/sglang/test/srt/test_mamba_pool_extended.py -v

# Run server integration tests (GPU required)
# See test/phases/ for the full phased test suite

# Lint
pre-commit run --all-files
```

## License

Same license as SGLang. See [LICENSE](LICENSE).

## Acknowledgments

Engram is built on [SGLang](https://github.com/sgl-project/sglang) by the [LMSYS](https://lmsys.org/) organization. SGLang is a high-performance serving framework trusted across 400,000+ GPUs worldwide, backed by contributions from xAI, AMD, NVIDIA, Intel, and dozens of research institutions. We are grateful to the SGLang team for building the foundation that makes this work possible.

We also build on the broader ecosystem that SGLang acknowledges: [vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Guidance](https://github.com/guidance-ai/guidance), [LightLLM](https://github.com/ModelTC/lightllm), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

The Mamba architecture was developed by Albert Gu and Tri Dao. Mamba2 was developed by Tri Dao and Albert Gu at Carnegie Mellon University.