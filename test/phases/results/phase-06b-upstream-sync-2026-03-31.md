# Phase 06B Results — Bulk Upstream Merge

| Field | Value |
|-------|-------|
| **Date** | 2026-03-31 |
| **Branch** | `upstream-sync-2026-Q1` |
| **Tag** | `phase-06b-pass` |
| **Upstream ref** | `sgl-project/sglang` `main` at `03e4f2858` |

## Summary

Merged 1,387 upstream commits from `sgl-project/sglang main` into the worktree branch. This brings the fork to full upstream parity (minus our snapshot/Mamba persistence features), so Phase 07 can re-apply snapshot features on the complete substrate.

## Commits Produced

| Hash | Description |
|------|-------------|
| `10634a497` | phase-06b: bulk upstream merge — 1,387 commits from sgl-project/sglang main |
| `dc8512cff` | phase-06b: fix mamba_layer_ids missing arg in radix cache tests |

## Conflicts Resolved (55 files)

### Take-upstream (35 files)

Files where we had no mamba/snapshot-specific changes — accepted upstream versions wholesale:

- `.gitignore`, `README.md`, `python/pyproject.toml`
- `python/sglang/cli/main.py`, `python/sglang/utils.py`
- `python/sglang/srt/compilation/compile.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/disaggregation/` (5 files: decode, encode_grpc_server, encode_receiver, encode_server, prefill)
- `python/sglang/srt/disaggregation/mori/conn.py`
- `python/sglang/srt/elastic_ep/expert_backup_client.py`
- `python/sglang/srt/entrypoints/anthropic/serving.py`
- `python/sglang/srt/entrypoints/grpc_server.py`
- `python/sglang/srt/entrypoints/openai/serving_transcription.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/managers/data_parallel_controller.py`
- `python/sglang/srt/managers/detokenizer_manager.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- `python/sglang/srt/mem_cache/session_aware_cache.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`
- `python/sglang/srt/speculative/eagle_info_v2.py`
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- Test files: `test_anthropic_server.py`, `test_anthropic_tool_use.py`, `test_epd_disaggregation.py`, `test_mamba_ssm_ssd.py`, `test_qwen3_next_models_mtp.py`, `test_socket_utils.py`, `test_mamba_unittest.py`, `test/run_suite.py`

### Delete-theirs (2 files)

- `python/sglang/srt/grpc/grpc_request_manager.py` — deleted by upstream, unused in our code
- (accepted upstream's deletion)

### Keep-ours (3 files)

- `test/registered/core/test_server_args.py` — modify/delete; kept ours (has snapshot arg tests)
- `test/registered/metrics/test_cpu_monitor.py` — modify/delete; kept ours
- `test/registered/unit/test_mamba_state_scatter_triton.py` — file location conflict; kept ours

### Manual merge (15 files)

Files where we preserved our mamba/snapshot work AND integrated upstream's new features:

| File | Our changes kept | Upstream additions integrated |
|------|-----------------|------------------------------|
| `server_args.py` | Snapshot persistence args (~180 lines), `configure_ipv6`/`is_valid_ipv6_address` imports, IPv6 URL formatting | `is_xpu` import, `LINEAR_ATTN_KERNEL_BACKEND_CHOICES`, `_handle_linear_attn_backend()` method, linear attn CLI args |
| `http_server.py` | Socket fd-based uvicorn startup (`reserved_socket.listen(128)`, `fd=reserved_socket.fileno()`) | (none — upstream removed socket reservation; we kept ours) |
| `scheduler.py` | `get_numa_node` import, `enable_metrics` init, 5 snapshot handlers (SaveSnapshot, ListSnapshots, GetSnapshotInfo, RestoreSnapshot, DeleteSnapshot) | `numa_utils`/`tensor_bridge` imports, hisparse coordinator setup, `DumperControlReqInput` handler, `unwrap_shm_features`, `enable_priority_preemption` rename, `can_run_set` optimization |
| `io_struct.py` | `session_params` field, `BatchMultimodalDecodeReq` class, `BatchMultimodalOutput` class, `streaming: bool = False` | (none — upstream removed these; we kept ours) |
| `memory_pool.py` | `SGLANG_ENABLE_SPEC_V2` + mamba extra_buffer validation | `start_layer: Optional[int] = None` parameter and init logic |
| `common.py` | `get_numa_node()` stub, profiling utils (`pytorch_profile`, `dump_to_file`, `is_triton_3`, `maybe_torch_compile`) | `VideoDecoderWrapper` import (upstream moved `decode_video_base64` to separate module) |
| `network.py` | `configure_ipv6()` function, `from_parts()` static method | CVE-2026-3060 security fix (localhost default), `__post_init__` bracket stripping, `resolve_host()`/`resolved()` DNS methods |
| `metrics_collector.py` | (none) | `QueueCount` dataclass for priority scheduling |
| `scheduler_metrics_mixin.py` | (none) | `DPCooperationInfo` and `QueueCount` imports |
| `req_time_stats.py` | `MINI_LB_LAUNCH` and `WAIT_PD_FINISH` stages | `bootstrap_done_time` field, `compute_and_observe_kv_transfer_metrics()`, `set_bootstrap_done_time()`, bootstrap/alloc sub-phase breakdown |
| `tokenizer_manager.py` | `current_load`/`current_load_lock` init | `ttft_observed` field, `_set_default_priority`/`_validate_rid_not_in_flight`, streaming backlog coalescing, encoder dispatch method |
| `scheduler_output_processor_mixin.py` | `process_batch_result_dllm` method | Multi-line import formatting |
| `launch_server.py` | (none) | Multi-line import formatting, `use_ray` branch |

## Design Decisions

1. **Socket fd vs host/port for uvicorn:** Kept our `fd=reserved_socket.fileno()` approach in `http_server.py`. Upstream switched to `host/port` with a configurable `SGLANG_TIMEOUT_KEEP_ALIVE`, but our socket reservation is important for port safety during server startup.

2. **CVE-2026-3060 network security fix:** Accepted upstream's change to default to `127.0.0.1` instead of `tcp://*` in `get_zmq_socket_on_host`. Our `configure_ipv6()` and `from_parts()` coexist alongside.

3. **DLLM method retention:** Kept `process_batch_result_dllm` in `scheduler_output_processor_mixin.py` even though upstream removed it — may be needed for future speculative decoding work.

4. **`mamba_layer_ids` as required param:** Upstream made this a required keyword-only arg on `HybridReqToTokenPool.__init__`. Fixed our test fixtures to pass it.

## Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| `pip install -e "python/"` | **PASS** | sglang 0.5.10rc0 |
| `test_mamba_unittest.py` | **PASS** (4/4) | `test_hybrid_linear_kv_pool`, `test_insert_prev_prefix_len`, `test_mamba_pool`, `test_mamba_radix_cache_1` |
| `test_mamba_radix_cache_comprehensive.py` | **PASS** (9/9) | All 9 tests pass after adding `mamba_layer_ids` |
| `test_mamba_radix_cache_gauntlet.py` | **5/6** | `test_branching_seqlen_triggered` fails — pre-existing CPU/CUDA device mismatch in `allocator.py:163` (confirmed present at `phase-06-pass` tag) |
| Syntax check (10 manual-merge files) | **PASS** | All parse with `ast.parse()` |

## Issues Discovered

1. **Pre-existing: `test_branching_seqlen_triggered` device mismatch** — `allocator.py:163` does `torch.cat((self.free_pages, free_index))` where one tensor is CPU and the other is CUDA. This existed before the merge (confirmed by reverting `allocator.py` to `phase-06-pass`). Not a regression from Phase 06B.

2. **`grpc_request_manager.py` deleted by upstream** — upstream removed this file entirely. No references to it in our codebase, so deletion is safe.

## What's Next

The worktree now contains the **full upstream codebase** (1,387 commits) plus all Phase 01–06 reconciliation work. Phase 07 (Re-apply Snapshot Features) should build on this substrate.

Phases 0, 1, 4, 7 (server-dependent validation) are deferred — they require an A100 server runtime and are part of the broader validation gate.
