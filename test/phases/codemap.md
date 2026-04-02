# Engram Codemap

Navigation reference for all phases. Each entry gives the approximate file, line range, and what lives there.

---

## Memory Pool Layer

### `HybridReqToTokenPool` + `MambaPool`
**File**: `python/sglang/srt/mem_cache/memory_pool.py`

GPU tensor storage for per-request Mamba conv/temporal states. `req_index_to_mamba_index_mapping` provides O(1) lookup. Key methods:
- `alloc([req])` — allocates req slot + mamba slot, sets `req.mamba_pool_idx`
- `free(req)` — frees req slot only
- `free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=None)` — frees mamba slot
- `get_mamba_indices(req_pool_idx)` — returns mamba index tensor for a req
- `available_size()` — req pool free slots
- `mamba_pool.available_size()` — mamba pool free slots

**Existing test coverage**: `test/registered/radix_cache/test_mamba_unittest.py` lines 36–141

### `HybridLinearKVPool`
**File**: `python/sglang/srt/mem_cache/memory_pool.py`

Hybrid KV pool: stores attention-layer KV cache, exposes `mamba_pool` reference to radix cache.
- `_transfer_full_attention_id(layer_id)` — maps layer_id → attention slot index

**Existing test**: `test_hybrid_linear_kv_pool` in `test_mamba_unittest.py` lines 36–67

---

## Radix Cache Layer

### `MambaRadixCache`
**File**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` lines 371–422 (class definition)

Hybrid radix tree with dual LRU lists. Constructor via `CacheInitParams`.

Key methods and their locations:
| Method | Lines | Notes |
|--------|-------|-------|
| `reset()` | ~415 | Reinitializes root, LRU lists, size counters |
| `insert(InsertParams)` | — | Inserts KV + mamba state |
| `match_prefix(MatchPrefixParams)` | — | Returns `device_indices`, `last_device_node`, `mamba_branching_seqlen` |
| `inc_lock_ref(node)` | 788–812 | Locks full_lock_ref up to root; locks mamba_lock_ref on node |
| `dec_lock_ref(node)` | 813–843 | Inverse of inc_lock_ref |
| `evict_mamba(mamba_num)` | 728–761 | Evicts mamba states from LRU; creates tombstones on internal nodes |
| `cache_unfinished_req(req, chunked)` | 556–672 | Caches in-progress request; handles ping-pong buffer path |
| `sanity_check()` | 844–848 | Validates full and mamba LRU list integrity end-to-end |
| `full_evictable_size()` | 854 | |
| `mamba_evictable_size()` | 857 | |
| `full_protected_size()` | 869 | |
| `mamba_protected_size()` | 872 | |

### `TreeNode`
**File**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` lines 63–115

Key fields:
- `value` — KV cache indices (`None` = fully evicted)
- `mamba_value` — Mamba state tensor (`None` = tombstone)
- `full_lock_ref` — protected from full eviction when > 0
- `mamba_lock_ref` — protected from mamba eviction when > 0
- `prev/next` — full LRU list links
- `mamba_prev/mamba_next` — mamba LRU list links

**Invariant**: `full_lock_ref >= mamba_lock_ref` always. If `mamba_lock_ref` is locked, `full_lock_ref` must also be locked. `full_lock_ref` propagates up to root; `mamba_lock_ref` only locks the node itself.

### `mamba_branching_seqlen`
**File**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` lines 984–1000

Computed in `_match_post_processor()` when KV hit extends past the last valid Mamba state (tombstone gap). Value is chunk-aligned (`// mamba_cache_chunk_size * mamba_cache_chunk_size`). `None` when no tombstone gap exists.

### `cache_unfinished_req` ping-pong path
**File**: `python/sglang/srt/mem_cache/mamba_radix_cache.py` lines 556–672

When `enable_mamba_extra_buffer=True`, reads from `mamba_ping_pong_track_buffer` (the "other" buffer index). Clears `req.mamba_last_track_seqlen` to `None` after caching. Updates `req.prefix_indices` and `req.last_node`.

---

## Forward Metadata

### `ForwardMetadata`
**File**: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` lines 26–42

Base dataclass for all forward passes:
- `query_start_loc` — cumulative seq lengths tensor
- `mamba_cache_indices` — per-req mamba pool indices
- `track_conv_indices`, `track_ssm_h_src/dst`, `track_ssm_final_src/dst` — prefill radix cache tracking

### `Mamba2Metadata`
**File**: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` lines 44–248

Extends `ForwardMetadata` with per-batch counts and `MixedMetadata`.

Key static/class methods:
| Method | Lines | Notes |
|--------|-------|-------|
| `prepare_decode(forward_metadata, seq_lens, ...)` | 152–174 | Decode-only path; `num_prefills=0`, `mixed_metadata=None` |
| `prepare_mixed(forward_metadata, chunk_size, forward_batch)` | 175–248 | Mixed/prefill path; populates `MixedMetadata` |
| `_query_start_loc_to_chunk_indices_offsets(query_start_loc, chunk_size, total_seqlens)` | 68–150 | Static; docstring has worked example: `[0,5,10]`, chunk=8 → `[0,0,1]`, `[0,5,0]` |

`MixedMetadata` fields (frozen dataclass):
- `has_initial_states` — bool tensor, True where `extend_prefix_lens > 0`
- `prep_initial_states` — True if any sequence has initial states
- `chunk_size`, `seq_idx`, `chunk_indices`, `chunk_offsets`
- `extend_seq_lens_cpu`

---

## Attention / SSM Backend

### `MambaMixer2` + `hybrid_linear_attn_backend`
**File**: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` lines 1–44 (imports)

Mamba2 SSM kernel integration. Key kernels:
- `causal_conv1d_fn` — prefill conv
- `causal_conv1d_update` — decode conv update
- GDN/Lightning hybrid variants via `fused_gdn_gating`, `fused_recurrent_gated_delta_rule_update`

---

## Snapshot System

### `MambaSnapshotManager`
**File**: `python/sglang/srt/snapshot/mamba_snapshot.py`

HTTP endpoints: `POST /save_snapshot`, `POST /restore_snapshot`. Safetensors-based disk persistence of conv and temporal states. Request body: `{"rid": "<request_id>"}`. Response: `{"success": bool}`.

### `TierManager`
**File**: `python/sglang/srt/snapshot/tier_manager.py`

Manages GPU → host (`MambaHostPool`) → disk tier promotion/demotion.

### `MambaHostPool`
**File**: `python/sglang/srt/snapshot/mamba_host_pool.py`

Pinned host memory buffer for snapshot staging between GPU and disk.

---

## Server Args

**File**: `python/sglang/srt/server_args.py`

Key Mamba-specific args:
- `--mamba-scheduler-strategy {no_buffer,extra_buffer}` — default: `no_buffer`
- `--disable-radix-cache` — bypasses `MambaRadixCache` entirely
- `--mamba-cache-chunk-size` — chunk size for `mamba_branching_seqlen` alignment
- Snapshot flags: inspect `server_args.py` directly for current names (flag names may vary — verify in `--help` output)

---

## Model

**Primary**: `granite-4.0-h-tiny` at `/home/jeanclawdai/models/granite-4.0-h-tiny`
- Architecture: `GraniteMoeHybridForCausalLM` — 40-layer Mamba/attention hybrid (Mamba2 SSM layers + sparse attention layers)
- Size: ~3 safetensors shards, hidden size 1536

**Fallback models** (see `config.sh`):
- `Nemotron-4B` — `/home/jeanclawdai/models/NVIDIA-Nemotron-3-Nano-4B-BF16` (FP16, use if granite OOMs)
- `Granite-Q4` — `/home/jeanclawdai/models/granite-4.0-h-tiny-gguf/granite-4.0-h-tiny-Q4_K_M.gguf` (GGUF Q4, comparison pass)

**Status**: Primary model already downloaded, no `huggingface-cli download` needed.

Server launch (used in Phases 1, 4, 6, 7, 8):
```bash
# Default (granite-4.0-h-tiny)
python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 [other flags]

# For Nemotron fallback:
# MODEL_PATH=$NEMOTRON_MODEL_PATH python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 [other flags]

# For Q4 GGUF comparison (args TBD):
# MODEL_PATH=$GRANITE_Q4_MODEL_PATH python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 [other flags]
```

The chatbot web UI routes `sglang/local` → `SGLANG_MODEL_ID=default` → `http://localhost:30000/v1`.

---

## HITL Chat Web UI

**URL**: `http://localhost:3000`
**Type**: Next.js chatbot (Vercel AI Chatbot template, v16.2.0)
**Running as**: `jeanclawdai` (process: `next-server`, port 3000)
**Source**: `/home/jeanclawdai/chatbot/`

The default model is `sglang/local`, which routes to the local SGLang server at `http://localhost:30000` via an OpenAI-compatible API. Select **SGLang Local** in the model picker.

Phases that use this UI: **Phase 1** (3-turn), **Phase 4** (5-turn), **Phase 7** (snapshot branch/restore).

The chatbot has auth (Auth.js) — use the guest login path or an existing account. Check `proxy.ts` for the auth flow; unauthenticated requests are redirected to `/api/auth/guest`.

---

## CI Registration

**File**: `python/sglang/test/ci/ci_register.py`

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=<seconds>, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=<seconds>, suite="stage-b-test-small-1-gpu-amd")
```

Must appear at **module top level** (before the `import unittest` block), before the test class definition.

Test runner: `python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu`

---

## Existing Test Files

| File | Tests | Notes |
|------|-------|-------|
| `test/registered/radix_cache/test_mamba_unittest.py` | 3 | `test_hybrid_linear_kv_pool`, `test_mamba_pool`, `test_mamba_radix_cache_1` |
| `test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py` | 10 | Tombstones, LRU, lock refs, COW, eviction, edge cases |

**Setup pattern** (used in both files): construct `Mamba2StateShape` → `Mamba2CacheParams` → `HybridReqToTokenPool` → `TokenToKVPoolAllocator` → `HybridLinearKVPool` → `CacheInitParams` → `MambaRadixCache`. See `test_mamba_radix_cache_comprehensive.py` `setUpClass` for the canonical pattern.

---

## Phase 3 Plan Reference

**File**: `phase3/test/test_plan.md`

Status legend: ✅ = written and passing, ⬜ = planned but not yet written.

As of 2026-02-16, integration tests for scheduler, ModelRunner, and memory management are all ⬜ (planned). Check this file before Phase 8 to pick up any newly completed items.

**File**: `phase3/tests/test_coverage_report.md`

Coverage targets for `MambaRadixCache` components as of 2026-02-16.
