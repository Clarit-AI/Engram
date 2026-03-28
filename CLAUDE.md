# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **SGLang with stateful Mamba inference** — a fork of [upstream SGLang](https://github.com/sgl-project/sglang) that adds snapshot persistence capabilities for Mamba SSM (State Space Model) hidden states. The key innovation is saving/restoring Mamba's internal memory to enable fast multi-turn conversations (25x+ speedup on subsequent turns).

**Upstream**: https://github.com/sgl-project/sglang

## Common Commands

### Installation
```bash
pip install -e "python[all]"       # Install with all dependencies
pip install -e "python[test]"     # Install with test dependencies
```

### Running the Server
```bash
python -m sglang.launch_server --model-path <model_name> [--port 30000]
```

Key server flags for this fork:
- `--enable-snapshot-persistence` — Enable Mamba state snapshot save/restore
- `--snapshot-dir <path>` — Directory for snapshot storage
- `--enable-memory-tiers` — Enable VRAM/RAM/Disk tier management
- `--enable-agent-tools` — Enable agent tool calling framework

### Running Tests
```bash
# Unit tests for snapshot and agent framework (this fork's additions)
cd test/sglang
pytest snapshot/ agents/ -v                    # Run all snapshot/agent tests
pytest snapshot/test_mamba_snapshot.py -v     # Run specific test file
pytest snapshot/ -k "test_metadata" -v        # Run tests matching pattern

# SRT (runtime) tests
cd test/srt
python test_srt_endpoint.py                   # Run SRT endpoint tests
python run_suite.py --suite per-commit        # Run CI test suite

# Manual tests
cd test/manual
python test_*.py                              # Various manual tests
```

### Linting
```bash
pre-commit run --all-files                     # Run all pre-commit hooks
pre-commit run ruff --files <files>            # Run ruff linter on specific files
```

### Building Documentation
```bash
# Docs are in docs/ directory - check specific doc for build requirements
```

## Architecture

### Key Directory Structure

```
python/sglang/
├── __init__.py              # Public API exports (Runtime, function, gen, SnapshotManager)
├── snapshot.py              # High-level snapshot API (user-facing)
├── launch_server.py         # Server entry point
├── lang/                    # Frontend Language API
│   ├── api.py              # @function, @gen, Runtime, etc.
│   ├── interpreter.py      # Statement interpretation
│   └── ir.py               # Intermediate representation
├── srt/                     # SGLang Runtime (core inference engine)
│   ├── entrypoints/        # HTTP/gRPC server entrypoints
│   ├── server_args.py      # Server configuration
│   ├── snapshot/           # Mamba snapshot implementation
│   │   ├── mamba_snapshot.py       # Core snapshot save/restore logic
│   │   ├── mamba_host_pool.py      # Host memory pool management
│   │   └── snapshot_policy.py      # Snapshot eviction policies
│   ├── agents/             # Agent framework
│   │   ├── agent_loop.py          # Agent execution loop
│   │   ├── tool_registry.py       # Tool registration
│   │   ├── builtin_tools.py       # Calculator, memory tools
│   │   └── api/                   # REST/WebSocket handlers
│   ├── models/            # Model implementations
│   │   └── registry.py    # Model registry
│   ├── layers/
│   │   └── attention/mamba/   # Mamba attention layer
│   └── mem_cache/
│       └── mamba_radix_cache.py # Mamba-specific radix cache
└── jit_kernel/            # CUDA kernel implementations
    └── hadamard.py        # Fast Hadamard transform for Mamba

test/
├── sglang/                 # Tests for this fork's additions
│   ├── snapshot/          # Snapshot save/restore tests
│   └── agents/           # Agent framework tests
├── srt/                   # SRT runtime tests
└── registered/            # CI registry tests
```

### Core Components

**Snapshot System** (`python/sglang/srt/snapshot/`):
- `MambaSnapshotManager` — Low-level serialization to safetensors + JSON
- `MambaHostPool` — Host memory pool for snapshot staging between GPU and disk
- `snapshot_policy.py` — LRU eviction policies for tier management

**Agent Framework** (`python/sglang/srt/agents/`):
- `AgentLoop` — Tool-calling agent execution with max iterations
- `ToolRegistry` — Registry for built-in (calculator, memory_store/recall/search) and custom tools
- REST API at `/v1/agent/*` endpoints

**Mamba Integration**:
- `sglang/srt/layers/attention/mamba/mamba.py` — Mamba layer implementation
- `sglang/srt/mem_cache/mamba_radix_cache.py` — Radix cache for Mamba
- `sglang/srt/configs/mamba_utils.py` — Mamba-specific utilities

### Key Data Structures

- `MambaSnapshotMetadata` — Dataclass tracking conversation_id, turn_number, token_count, mamba_pool_idx, layer_config
- `MambaHostPool` — Manages host (RAM) staging area for snapshots
- `SnapshotPolicy` — Enum controlling when snapshots are created (eager/lazy/effective_size)

## This Fork's Key Features

1. **Snapshot Persistence**: Save Mamba SSM hidden states to disk, restore later (10-50ms save, 5-30ms restore)
2. **3-Tier Memory Management**: Active (VRAM) → Warm (RAM) → Cold (Disk) with automatic LRU eviction
3. **Agent Framework**: Tool calling with 4 built-in tools (calculator, memory store/recall/search)
4. **Fast Hadamard Transform**: Custom CUDA kernels for Mamba's structured state space operations

## Testing Notes

- This fork adds tests in `test/sglang/snapshot/` and `test/sglang/agents/`
- Tests use `pytest` with `asyncio_mode = auto` (see `test/pytest.ini`)
- SRT tests use `unittest` framework directly
- Integration tests in `test/registered/` use the CI registry system

## Development Phases

Phase plans and validation reports are in `phase3/` directory:
- `PHASE_3_PLAN.md` - Full development plan with 4 phases
- `MAMBA_SNAPSHOT_RESTORATION_PLAN.md` - Original snapshot restoration plan
- `phase3/oversight/validation_reports/` - Phase validation reports
- `phase3/PERFORMANCE_ANALYSIS.md` - Static analysis with optimization opportunities

### Phase Status
- Phase 3.1 ✅ Complete - Foundation
- Phase 3.2 ✅ Complete - Core Implementation (MambaRadixCache was already implemented!)
- Phase 3.3 ✅ Complete - Static Analysis (optimizations identified)
- Phase 3.4 ⬜ Pending - Final Audit

## GCloud Testing Instance

> **Credentials**: See the private runbook for access credentials and instance details.

**Instance:** `<INSTANCE_NAME>`
**Zone:** `<ZONE>`
**Project:** `<PROJECT_ID>`
**GPU:** `<GPU_TYPE>`

Connect via:
```bash
# Template — fill in values from runbook
gcloud compute ssh --zone "<ZONE>" "<INSTANCE_NAME>" --project "<PROJECT_ID>"
```

Clone at: `<CLONE_PATH>`
Cloudflare tunnel: `<TUNNEL_HOST>` → `localhost:30000`

### Available Models

| Model | Path | Type | Use Case |
|-------|------|------|----------|
| granite-4.0-h-tiny | `<MODEL_DIR>/granite-4.0-h-tiny` | FP16, 40-layer hybrid | Primary (try first) |
| Nemotron-4B | `<MODEL_DIR>/NVIDIA-Nemotron-3-Nano-4B-BF16` | FP16, 4B | Backup if granite OOMs |
| Granite-Q4 | `<MODEL_DIR>/granite-4.0-h-tiny-gguf/granite-4.0-h-tiny-Q4_K_M.gguf` | GGUF Q4 | Quantized comparison (args TBD) |

**Testing priority:** granite-4.0-h-tiny → Nemotron (if OOM) → Granite-Q4 (comparison pass)

## Known Issues

1. **Mamba model config bug**: `architectures: None` causes crash in model_config.py line 149 *(fixed)*
2. **Granite GGUF**: `granitehybrid` architecture not supported by transformers GGUF loader
3. **V100 memory**: Full Granite MoE models require >16GB GPU memory — use Nemotron fallback

## Memory Context

For session persistence, important project context is stored in:
- Global memory: `~/.claude/projects/<PROJECT_PATH>/memory/`
- Session ID for this project: `sglang-mamba-session-001`