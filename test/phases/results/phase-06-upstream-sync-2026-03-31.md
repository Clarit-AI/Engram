# Phase 06 Results — Server and HTTP Drift (Upstream Sync)

**Date:** 2026-03-31
**Branch:** `upstream-sync-2026-Q1`
**Base tag:** `phase-05-pass` (implicit; no explicit tag — continued from prior Phase 05 work)
**Result tag:** `phase-06-pass`
**Result:** PASS (import verification)

---

## Objective

Merge lower-risk `server_args.py` and `http_server.py` drift from upstream, then re-apply our snapshot REST endpoints and CLI flags on top of the updated substrate.

---

## Commits Integrated

| # | Upstream Commit | Description | Conflicts |
|---|-----------------|-------------|-----------|
| 1 | `d5307ce02` | Add metadata field to `UpdateWeightFromDiskReqInput` (#18821) | 0 |
| 2 | `cc451671b` | Anthropic-compatible API endpoint (#18630) | 0 |
| 3 | `581bf53e0` | Whisper model support & `/v1/audio/transcriptions` (#16983) | 1 |
| 4 | `198381d9c` | SSL/TLS support for HTTP and gRPC servers (#18973) | 3 |
| 5 | `fc7f9c1de` | Rename `--stream-output` to `--incremental-streaming-output` (#20614) | 0 |
| 6 | `f0458e0b4` | Move network/socket utils from `common.py` to `network.py` (#20646) | 12 |
| 7 | `135af6dc9` | EPD/VLM video/audio input support (#17824) | 5 |
| 8 | `8a4cdcd53` | Simplify flush_cache: reject concurrent, remove client-side retry (#21490) | 4 |
| 9 | `2e65c27b2` | Add flush_cache timeout API plumbing (#21413) | 4 |
| 10 | `ced69c9f8` | Whisper CUDA graph and timestamp support (#21190) | 1 |

**Total:** 10 upstream commits cherry-picked, 30 conflicts resolved.

**Fork-specific fixup commits (3):**
- `54fb0959f` — Port `configure_ipv6` to `network.py` (missed by upstream network refactor)
- `f22e2bb9b` — Add `is_mps()` and `get_numa_node()` stubs to `common.py`
- `fce9bce14` — Resolve SSL/TLS conflicts (3 files: pyproject.toml, server_args.py, http_server.py)

---

## Conflicts Resolved

### 1. FastAPI import merge (`581bf53e0`)
- **File:** `http_server.py`
- **Nature:** HEAD had `Query` import; upstream added `File, Form, UploadFile`
- **Resolution:** Kept both — merged import lists

### 2. SSL/TLS integration (`198381d9c`) — 3 conflicts
- **pyproject.toml:** Kept our `xgrammar==0.1.32` (newer); added upstream's `grpcio>=1.78.0` deps
- **server_args.py:** Kept PD-prefill backward compat block; added `_handle_ssl_validation()` method
- **http_server.py:** Accepted upstream's SSL-aware uvicorn launch (`reserved_socket`, SSL config)

### 3. Network refactor (`f0458e0b4`) — 12 conflicts
- **5 modify/delete:** Files deleted in HEAD, modified in upstream → accepted upstream versions (`encode_grpc_server.py`, `expert_backup_client.py`, `expert_backup_manager.py`, `network.py`, `test_socket_utils.py`)
- **7 content:** All import-line conflicts (`from sglang.srt.utils.common import X` → `from sglang.srt.utils.network import X`) → accepted upstream's new import paths

### 4. EPD/VLM video/audio (`135af6dc9`) — 5 conflicts
- **1 modify/delete:** `embedding_cache_controller.py` → accepted upstream
- **4 content:** `encode_receiver.py` (6 regions), `encode_server.py` (5 regions), `tokenizer_manager.py` (3 regions), test file → accepted upstream's EPD multimodal additions

### 5. flush_cache simplification (`8a4cdcd53`) — 4 conflicts
- **scheduler.py:** `return_health_check_ct` → `return_health_check_ipcs` + `_pending_flush`; old `flush_cache_wrapped` → new `_check_pending_flush` + timeout-aware `flush_cache_wrapped`
- **http_server.py:** flush_cache endpoint now takes `timeout` Query param
- **test_utils.py:** Removed `flush_cache_with_retry` (superseded by server-side timeout)

### 6. flush_cache timeout plumbing (`2e65c27b2`) — 4 conflicts (out-of-order cherry-pick)
- **Design decision:** `8a4cdcd53` (applied first) used `Optional` for `_pending_flush` (reject concurrent); `2e65c27b2` (applied second) used `Deque` (allow multiple). Kept `8a4cdcd53`'s `Optional` design since it was the intentional simplification.
- **tokenizer_communicator_mixin.py:** Accepted upstream's `timeout_s` parameter addition (needed by http_server)

### 7. Whisper CUDA graph (`ced69c9f8`) — 1 conflict
- **server_args.py:** Accepted upstream's new `_get_default_attn_backend()` method

---

## Design Decisions

### 1. Cherry-pick order
Applied commits in approximate chronological order (by author date), not the order listed in the phase doc. This minimized conflicts for most commits, though `8a4cdcd53` and `2e65c27b2` (related flush_cache commits) had to be reconciled due to deliberate out-of-order application.

### 2. flush_cache: Optional vs Deque
Kept `8a4cdcd53`'s `Optional[Tuple[FlushCacheReqInput, float]]` design over `2e65c27b2`'s `Deque`. The commit message "reject concurrent requests" was intentional — one pending flush at a time, reject any new request while one is pending.

### 3. `configure_ipv6` — fork-specific utility
This function existed only in our fork's `common.py`. The network refactor (`f0458e0b4`) removed network functions from `common.py` but upstream never had `configure_ipv6`. Ported it to `network.py` where it logically belongs.

### 4. `is_mps()` / `get_numa_node()` stubs
These were added to upstream's `common.py` in commits not part of Phase 06 but were imported by the updated `scheduler.py`. Added `is_mps()` (real impl via `torch.backends.mps`) and `get_numa_node()` (stub returning `None`). Full NUMA support will be ported in Phase 07+.

---

## Test Results

### Import Verification (automated, no GPU required)

| Module | Result |
|--------|--------|
| `sglang.srt.server_args` | PASS |
| `sglang.srt.managers.io_struct` | PASS |
| `sglang.srt.managers.scheduler` | PASS |
| `sglang.srt.entrypoints.http_server` | PASS |
| `sglang.srt.utils.network` | PASS |
| `sglang.srt.snapshot.mamba_snapshot` | PASS |
| `sglang.srt.snapshot.tier_manager` | PASS |

### Snapshot Code Integrity

| Check | Result |
|-------|--------|
| REST endpoints in `http_server.py` (5 endpoints) | Present |
| CLI flags in `server_args.py` (8 flags) | Present |
| IO structs in `io_struct.py` (6 request/response pairs) | Present |
| `enable_snapshot_persistence` default value | `False` ✓ |

### Live Server Validation
Not performed — Phase 06 is a merge/import phase. Live validation deferred to Phase 06B (bulk merge) and Phase 08 (full validation).

---

## Issues Discovered

### 1. Import breakage from network refactor
- `configure_ipv6` was not in upstream's `common.py` or `network.py` — it was fork-specific
- `is_port_available` and `is_valid_ipv6_address` were left in `server_args.py`'s `common` import despite being moved to `network.py`
- Both fixed in post-merge fixup commits

### 2. `get_numa_node` and `is_mps` missing from fork
- Upstream added these between our fork point and current HEAD
- `scheduler.py` imports them but doesn't use them yet (future use)
- Stubbed for now; will be fully implemented when NUMA auto-configuration is ported

### 3. Dolt push warning
- `bd` (beads) dolt auto-push fails with "no common ancestor" — likely a Dolt remote configuration issue
- Non-blocking; beads state tracked locally

---

## Files Modified (summary)

**New files added from upstream:**
- `python/sglang/srt/entrypoints/anthropic/` (3 files — Anthropic API)
- `python/sglang/srt/entrypoints/ssl_utils.py` (SSL cert refresh)
- `python/sglang/srt/entrypoints/openai/serving_transcription.py` (Whisper)
- `python/sglang/srt/models/whisper.py` + `multimodal/processors/whisper.py`
- `python/sglang/srt/utils/network.py` (network utilities from common.py)
- `python/sglang/srt/elastic_ep/expert_backup_{client,manager}.py` (restored)
- `python/sglang/srt/disaggregation/encode_grpc_server.py` (restored)
- `python/sglang/srt/mem_cache/storage/mooncake_store/embedding_cache_controller.py`
- Various test files

**Key files modified:**
- `python/sglang/srt/server_args.py` — SSL/TLS flags, Whisper flags, network import restructuring
- `python/sglang/srt/entrypoints/http_server.py` — Anthropic/Whisper/SSL routes, flush_cache timeout
- `python/sglang/srt/managers/scheduler.py` — flush_cache deferred execution, network imports
- `python/sglang/srt/managers/io_struct.py` — manifest field, timeout_s field
- `python/sglang/srt/utils/common.py` — network functions removed (moved to network.py), stubs added
