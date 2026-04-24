<!-- ENGRAM_MODIFIED — Fork file manifest: canonical registry of all Engram changes -->
# Engram Fork Manifest

> **Engram:** v0.2.0 (pre-merge; target v0.3.0)
> **Last updated:** 2026-04-24
>
> Auto-generated from `git diff upstream/main...HEAD` and ENGRAM_MODIFIED markers.
> Commit: 2d047c4996f13fc1bfc08a7fb6f1eedd7f2b2906 (+ pre-merge snapshot.py addition)

## How to use this file

- Before an upstream sync, read this to understand what Engram has changed
- Files marked with high block counts have the most complex changes
- "Status" column: M = modified upstream file, A = added by Engram, D = deleted from upstream, R = renamed
- The 39 workflow files all share the same CI guard pattern and are listed separately for completeness

## Statistics

| Category | Count |
|----------|-------|
| Modified files (M) | 79 |
| Added files (A) | 177 |
| Deleted files (D) | 7 |
| Renamed files (R) | 4 |
| Total ENGRAM_MODIFIED headers | 86 |
| Total BEGIN/END ENGRAM blocks | 322 |

## High Conflict Risk — Modified Upstream Files ⚠️

These are the files where upstream changes will most likely conflict with Engram additions.

| File | Blocks | +/- | Description |
|------|--------|-----|-------------|
| `python/sglang/srt/managers/scheduler.py` | 15 | +1223 | Mamba state management, snapshot save/restore, conversation tracking |
| `python/sglang/srt/managers/tokenizer_manager.py` | 9 | +98 | Snapshot and agent token management |
| `python/sglang/srt/server_args.py` | 3 | +262 | Snapshot/Mamba CLI args, memory tiers, agent tools, SSL validation, IPv6 URL |
| `python/sglang/srt/entrypoints/http_server.py` | 3 | +208 | Snapshot HTTP endpoints, agent API routes, socket pre-binding |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | 3 | +22 | Mamba radix cache extensions |
| `python/sglang/srt/managers/io_struct.py` | 3 | +138 | Snapshot and agent request/response data structures |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 2 | +113 | KV cache Mamba state integration |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | 3 | +113 | Snapshot output processing hooks |
| `python/sglang/srt/managers/schedule_batch.py` | 2 | +8 | Snapshot batch field |
| `python/sglang/srt/configs/model_config.py` | 2 | +37 | Mamba config adaptations (safe architecture access, multimodal guards) |

### Core Engine — Other Modified Files (subtotal: 24 blocks)

| File | Blocks | +/- | Description |
|------|--------|-----|-------------|
| `python/sglang/srt/mem_cache/memory_pool.py` | 1 | +9 | Mamba memory pool hooks |
| `python/sglang/srt/observability/trace.py` | 0 | +2 | Safer exception handling in host ID detection |
| `python/sglang/srt/model_executor/model_runner.py` | 0 | +15 | Mamba2Config support, safe architecture access |
| `python/sglang/srt/entrypoints/openai/protocol.py` | 2 | — | Snapshot fields in OpenAI-compatible API protocol |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 1 | — | Snapshot passthrough in chat |
| `python/sglang/srt/entrypoints/openai/serving_completions.py` | 1 | — | Snapshot passthrough in completions |
| `python/sglang/srt/observability/req_time_stats.py` | 1 | — | Snapshot timing statistics |
| `python/sglang/srt/models/nemotron_h.py` | 0 | — | Mamba state hooks for Nemotron-H |
| `python/sglang/srt/utils/hf_transformers_utils.py` | 2 | — | Mamba2Config registry entry |
| `python/sglang/srt/utils/network.py` | 2 | — | Network utilities for snapshot system |
| `python/sglang/srt/utils/common.py` | 2 | — | Snapshot utility functions |
| `python/sglang/__init__.py` | 2 | — | Snapshot module import |
| `python/sglang/lang/backend/runtime_endpoint.py` | 1 | — | Snapshot API client |
| `python/sglang/lang/interpreter.py` | 1 | — | Snapshot interpreter support |
| `python/sglang/jit_kernel/fused_metadata_copy.py` | 3 | — | Mamba metadata kernel extensions |
| `python/sglang/test/run_eval.py` | 0 | — | Test runner adaptation |
| `benchmark/hicache/bench_multiturn.py` | 0 | — | Benchmark adaptation for Mamba |
| `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py` | 0 | — | Dashboard fork tweak |
| `scripts/ci/utils/slash_command_handler.py` | 0 | — | CI handler fork tweak |
| `sgl-kernel/benchmark/bench_fp4_gemm.py` | 0 | — | Benchmark fork tweak |
| `sgl-kernel/python/sgl_kernel/flash_mla.py` | 0 | — | Kernel fork tweak |
| `sgl-model-gateway/bindings/golang/go.mod` | 0 | — | Go module (not fork-differentiated) |
| `sgl-model-gateway/bindings/golang/go.sum` | 0 | — | Go module (not fork-differentiated) |
| `sgl-model-gateway/bindings/golang/examples/oai_server/go.mod` | 0 | — | Go example (not fork-differentiated) |
| `sgl-model-gateway/bindings/golang/examples/oai_server/go.sum` | 0 | — | Go example (not fork-differentiated) |
| `docs/references/production_request_trace.md` | 0 | — | Snapshot trace documentation |
| `README.md` | 0 | — | Fork README: Engram project description, benchmarks, documentation links |
| `.gitignore` | 3 | — | Fork-specific ignore patterns |
| `.pre-commit-config.yaml` | 2 | — | Fork pre-commit configuration |

## CI Workflows (39 files — uniform fork guard)

All 39 modified workflow files share the same pattern:
```yaml
if: github.repository == 'sgl-project/sglang'
```

This disables upstream CI on the Engram fork. New/changed workflows added by Engram are listed in Added Files.

| Workflow File |
|---------------|
| `.github/workflows/amd-aiter-scout.yml` |
| `.github/workflows/amd-ci-job-monitor.yml` |
| `.github/workflows/auto-tune.yml` |
| `.github/workflows/bot-bump-flashinfer-version.yml` |
| `.github/workflows/bot-bump-kernel-version-to-sglang.yml` |
| `.github/workflows/bot-bump-kernel-version.yml` |
| `.github/workflows/bot-bump-sglang-version.yml` |
| `.github/workflows/cancel-unfinished-pr-tests.yml` |
| `.github/workflows/ci-coverage-overview.yml` |
| `.github/workflows/ci-failure-monitor.yml` |
| `.github/workflows/execute-notebook.yml` |
| `.github/workflows/list-active-pr-runs.yml` |
| `.github/workflows/nightly-release-gateway.yml` |
| `.github/workflows/nightly-test-amd-rocm720.yml` |
| `.github/workflows/nightly-test-amd.yml` |
| `.github/workflows/nightly-test-npu.yml` |
| `.github/workflows/open-pr-copy-from-oss.yml` |
| `.github/workflows/open-pr-copy-to-oss.yml` |
| `.github/workflows/pr-benchmark-rust.yml` |
| `.github/workflows/pr-test-amd-rocm720.yml` |
| `.github/workflows/pr-test-amd.yml` |
| `.github/workflows/pr-test-jit-kernel.yml` |
| `.github/workflows/pr-test-multimodal-gen.yml` |
| `.github/workflows/pr-test-npu.yml` |
| `.github/workflows/pr-test-rust.yml` |
| `.github/workflows/pr-test-sgl-kernel.yml` |
| `.github/workflows/pr-test-xeon.yml` |
| `.github/workflows/pr-test-xpu.yml` |
| `.github/workflows/pr-test.yml` |
| `.github/workflows/release-branch-cut.yml` |
| `.github/workflows/release-docker-amd-nightly.yml` |
| `.github/workflows/release-docker-npu-nightly.yml` |
| `.github/workflows/release-docker-npu.yml` |
| `.github/workflows/release-pypi-gateway.yml` |
| `.github/workflows/release-pypi-nightly.yml` |
| `.github/workflows/release-pypi-pr.yml` |
| `.github/workflows/release-whl-kernel.yml` |
| `.github/workflows/runner-utilization.yml` |
| `.github/workflows/slash-command-handler.yml` |

## Added Files — Engram-Only

### Snapshot System (9 files)

| File | Purpose |
|------|---------|
| `python/sglang/snapshot.py` | **Public API** — top-level SnapshotManager + exception hierarchy |
| `python/sglang/srt/snapshot/__init__.py` | Snapshot module exports |
| `python/sglang/srt/snapshot/conversation_tracker.py` | Multi-turn conversation state tracking |
| `python/sglang/srt/snapshot/mamba_host_pool.py` | Multi-host Mamba state pool management |
| `python/sglang/srt/snapshot/mamba_snapshot.py` | Core snapshot save/restore implementation |
| `python/sglang/srt/snapshot/snapshot_hooks.py` | Integration hooks into scheduler |
| `python/sglang/srt/snapshot/snapshot_policy.py` | Snapshot retention and eviction policy |
| `python/sglang/srt/snapshot/state_health.py` | State validation and health checks |
| `python/sglang/srt/snapshot/tier_manager.py` | Memory tier management for snapshots |

### Agent Framework (10 files)

| File | Purpose |
|------|---------|
| `python/sglang/srt/agents/__init__.py` | Agent module exports |
| `python/sglang/srt/agents/agent_loop.py` | Main agent execution loop |
| `python/sglang/srt/agents/api/__init__.py` | Agent API module |
| `python/sglang/srt/agents/api/handlers.py` | HTTP/WebSocket agent handlers |
| `python/sglang/srt/agents/api/models.py` | Agent API request/response models |
| `python/sglang/srt/agents/api/websocket.py` | WebSocket endpoint for agents |
| `python/sglang/srt/agents/builtin_tools.py` | Built-in agent tools (calculator, etc.) |
| `python/sglang/srt/agents/tool_execution.py` | Tool execution engine |
| `python/sglang/srt/agents/tool_parser.py` | Tool call parsing from model output |
| `python/sglang/srt/agents/tool_registry.py` | Tool registration and discovery |

### Mamba2 Model Support (2 files)

| File | Purpose |
|------|---------|
| `python/sglang/srt/models/mamba2.py` | Mamba2 model implementation |
| `python/sglang/srt/configs/mamba2.py` | Mamba2 configuration registry |

### Test Suite — Snapshot & Agent Tests (16 files)

| File |
|------|
| `test/sglang/agents/api/test_api_models.py` |
| `test/sglang/agents/test_tool_execution.py` |
| `test/sglang/agents/test_tool_registry.py` |
| `test/sglang/snapshot/__init__.py` |
| `test/sglang/snapshot/test_conversation_id_propagation.py` |
| `test/sglang/snapshot/test_conversation_tracker.py` |
| `test/sglang/snapshot/test_host_pool.py` |
| `test/sglang/snapshot/test_mamba_snapshot.py` |
| `test/sglang/snapshot/test_snapshot_policy.py` |
| `test/registered/core/test_server_args.py` |
| `test/registered/core/test_ssl_cert_refresher.py` |
| `test/registered/observability/test_cpu_monitor.py` |
| `test/registered/radix_cache/test_mamba_baseline_inference.py` |
| `test/registered/radix_cache/test_mamba_extra_buffer.py` |
| `test/registered/radix_cache/test_mamba_gauntlet_stress.py` |
| `test/registered/radix_cache/test_mamba_metadata.py` |

### Test Suite — Registered (40 files)

Large test suite under `test/registered/` covering radix cache, stateful inference, Mamba pool, and snapshot end-to-end tests. Not enumerated here — see `test/registered/` directory.

### Phase Test System (55 files)

Phased test protocol with individual phase runbooks, results, and scripts:
- `test/phases/phase-00-environment-verification.md` through `phase-11a-codestral-mamba-7b.md`
- `test/phases/results/` containing markdown and JSON result files
- `test/phases/scripts/` containing Python test scripts

### Documentation — Stateful Mamba (9 files)

| File | Purpose |
|------|---------|
| `docs/stateful_mamba/INDEX.md` | Documentation index |
| `docs/stateful_mamba/api_guide.md` | User-facing HTTP API guide |
| `docs/stateful_mamba/http_api_spec.md` | Technical HTTP contract |
| `docs/stateful_mamba/architecture.md` | System architecture overview |
| `docs/stateful_mamba/user_guide.md` | End-user guide |
| `docs/stateful_mamba/examples.md` | Usage examples |
| `docs/stateful_mamba/troubleshooting.md` | Common issues and fixes |
| `docs/stateful_mamba/state_validation.md` | State validation methodology |
| `docs/stateful_mamba/migration_guide.md` | Upstream migration guide |

### Configuration & Policy (15 files)

| File | Purpose |
|------|---------|
| `AGENTS.md` | Multi-agent framework instructions |
| `CLAUDE.md` | Claude Code project instructions |
| `SECURITY.md` | Security policy and disclosure process |
| `engram.json` | Engram project metadata |
| `funding.json` | Funding/platform manifest |
| `NEMOTRON3_SUPER_RESULTS.md` | Benchmark results |
| `QWEN3_CODER_NEXT_RESULTS.md` | Benchmark results |
| `.engram/SYNC_PLAYBOOK.md` | Upstream sync playbook |
| `.engram/policy/protected-paths.json` | Protected path policy |
| `docs/wiki-pages/Agent-Framework.md` | Wiki documentation |
| `docs/wiki-pages/Stateful-Mamba-Guide.md` | Wiki documentation |
| `docs/wiki-pages/Pure-Mamba2-Support.md` | Wiki documentation |
| `docs/wiki-pages/Upstream-Sync-Q1-2026.md` | Wiki documentation |
| `docs/wiki-pages/Phase-3-Audit-Archive.md` | Archived documentation |
| `docs/wiki-pages/Project-Reports-Archive.md` | Archived documentation |

### CI & Scripts (8 files)

| File | Purpose |
|------|---------|
| `.github/workflows/upstream-sync.yml` | Upstream sync automation |
| `.github/workflows/slash-command-handler.yml` | Slash command handler |
| `scripts/policy/check_protected_paths.py` | Protected path checker |
| `scripts/validate_funding_json.py` | Funding JSON validator |
| `scripts/validate-sync.sh` | Sync validation script |
| `test/manual/test_sglang_auto_benchmark_skill_configs.py` | Manual test |
| `test/manual/test_streaming_session_leak.py` | Manual test |
| `benchmark/hicache/bench_multiturn.py` | Multi-turn benchmark |

### Assets & Branding (6 files)

| File | Purpose |
|------|---------|
| `assets/engram-logo-banner.png` | Banner logo |
| `assets/engram-logo-square.png` | Square logo |
| `assets/archive/logo.png` | Archived logo (renamed) |
| `assets/archive/logo.svg` | Archived logo (renamed) |
| `assets/archive/logo_square.png` | Archived square logo (renamed) |
| `assets/archive/logo_square.svg` | Archived square logo (renamed) |

## Deleted Files (7)

These skill files were removed from `.claude/skills/`:

| File | Reason |
|------|--------|
| `.claude/skills/add-jit-kernel/SKILL.md` | Obsolete — replaced by sgl-kernel approach |
| `.claude/skills/add-sgl-kernel/SKILL.md` | Obsolete |
| `.claude/skills/ci-workflow-guide/SKILL.md` | Obsolete |
| `.claude/skills/debug-cuda-crash/SKILL.md` | Obsolete |
| `.claude/skills/generate-profile/SKILL.md` | Obsolete |
| `.claude/skills/sglang-bisect-ci-regression/SKILL.md` | Obsolete |
| `.claude/skills/write-sglang-test/SKILL.md` | Obsolete |

## Renamed Files (4)

| Old Path | New Path |
|----------|----------|
| `assets/logo.png` | `assets/archive/logo.png` |
| `assets/logo.svg` | `assets/archive/logo.svg` |
| `assets/logo_square.png` | `assets/archive/logo_square.png` |
| `assets/logo_square.svg` | `assets/archive/logo_square.svg` |

## Intentionally Unmarked

The following are not fork-differentiated and accept upstream changes verbatim:
- `sgl-model-gateway/bindings/golang/**` — Go module Dependabot bumps (not Engram-specific)
- `docs/` — Vanilla upstream documentation (no ENGRAM_MODIFIED marker needed)

## Marking Coverage

| Metric | Count |
|--------|-------|
| ENGRAM_MODIFIED headers | 86 |
| BEGIN/END ENGRAM blocks | 322 |
| Modified files without headers | 8 (all small/cosmetic: go.mod, go.sum, examples, etc.) |
| Added files without headers | ~100+ (pure additions, no markers needed) |

## Block Count by Subsystem

### High Conflict Risk (45 total)
| Subsystem | Files | Blocks |
|-----------|-------|--------|
| Scheduler | 1 | 15 |
| Tokenizer Manager | 1 | 9 |
| HTTP Server | 1 | 3 |
| Server Args | 1 | 3 |
| Mamba Radix Cache | 1 | 3 |
| IO Struct | 1 | 3 |
| Model Runner KV Cache Mixin | 1 | 2 |
| Scheduler Output Processor | 1 | 3 |
| Schedule Batch | 1 | 2 |
| Model Config | 1 | 2 |

### Other Modified (24 total)
| File | Blocks |
|------|--------|
| OpenAI protocol | 2 |
| OpenAI serving_chat | 1 |
| OpenAI serving_completions | 1 |
| req_time_stats | 1 |
| hf_transformers_utils | 2 |
| network | 2 |
| common | 2 |
| __init__.py | 2 |
| runtime_endpoint | 1 |
| interpreter | 1 |
| fused_metadata_copy | 3 |
| memory_pool | 1 |
| .gitignore | 3 |
| .pre-commit-config.yaml | 2 |
| All others (13 files) | 0 |

**Grand total: 69 blocks** (45 high conflict + 24 other modified = 69 blocks in modified Python files; 322 total blocks across all file types per global grep)

### Per-file verification
Run `grep -c "BEGIN ENGRAM" <filepath>` on any file listed above to verify.