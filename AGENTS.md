# SGLang-Mamba - Agent Instructions

## Project Overview

This is **SGLang with stateful Mamba inference** — a fork of [upstream SGLang](https://github.com/sgl-project/sglang) that adds snapshot persistence for Mamba SSM hidden states. The key innovation: saving/restoring Mamba's internal memory to enable fast multi-turn conversations (25x+ speedup on subsequent turns).

**Status:** Core implementation complete. Startup restore (Gap 3) merged. Server-phase testing blocked by sm75+ GPU requirement. Performance optimizations identified but not yet implemented.

**Upstream:** https://github.com/sgl-project/sglang
**Fork:** https://github.com/KHAEntertainment/sglang-mamba

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** SGLang (forked from upstream)
- **Key dependencies:** PyTorch 2.9.1, transformers 4.57.1, flashinfer 0.6.3, sgl-kernel 0.3.21
- **Build:** `pip install -e "python[all]"`
- **Testing:** pytest + 9-phase test plan (test/phases/)
- **Linting:** ruff via pre-commit hooks
- **GPU requirement:** sm75+ (A100, T4, A10G, RTX 30xx+) for server phases

## Project Structure

```
python/sglang/
├── snapshot.py              # High-level SnapshotManager API (user-facing)
├── srt/
│   ├── snapshot/               # Mamba snapshot implementation
│   │   ├── mamba_snapshot.py       # Core save/restore (safetensors + JSON)
│   │   ├── mamba_host_pool.py      # Host RAM staging pool (Tier 2)
│   │   ├── tier_manager.py        # 3-tier VRAM/RAM/Disk orchestration
│   │   ├── conversation_tracker.py # Tier state tracking
│   │   ├── snapshot_hooks.py      # Hook manager (post_forward, pre_eviction, on_demand)
│   │   └── snapshot_policy.py     # Trigger policies, retention, branching
│   ├── agents/                  # Agent framework
│   │   ├── agent_loop.py         # Tool-calling agent execution
│   │   ├── tool_registry.py      # Tool registration
│   │   ├── builtin_tools.py      # Calculator, memory store/recall/search
│   │   └── api/                  # REST + WebSocket handlers
│   ├── mem_cache/
│   │   └── mamba_radix_cache.py  # Dual radix cache (KV + Mamba), with tombstones + COW
│   ├── managers/
│   │   └── scheduler.py          # Snapshot save/restore handlers, startup restore
│   └── layers/attention/mamba/   # Mamba layer implementation (13 files)
test/
├── sglang/                     # Fork-specific unit tests
│   ├── snapshot/               # 4 test files (46 pass, 1 skip)
│   └── agents/                 # 3 test files (37 pass)
├── registered/radix_cache/     # MambaRadixCache comprehensive + gauntlet tests
├── phases/                    # 9-phase test plan (0-8), config.sh, codemap
└── phases/results/            # Phase result reports
docs/
├── stateful_mamba/            # Snapshot system docs
├── agent_framework/           # Agent framework docs
└── migration-prep/            # VM context export (operational, not linked)
skills/mamba-sglang/            # Gemini skill + 3 reference docs
.beads/                         # Beads issue tracking (on main branch)
```

## Development Commands

```bash
# Install
pip install -e "python[all]"
pip install -e "python[test]"

# Run server
python -m sglang.launch_server --model-path <model> --enable-snapshot-persistence --snapshot-dir ./snapshots

# Key server flags
--enable-snapshot-persistence    # Enable Mamba snapshot save/restore
--enable-memory-tiers            # Enable VRAM/RAM/Disk tier management
--enable-agent-tools             # Enable agent tool calling framework
--mamba-scheduler-strategy [no_buffer | extra_buffer]
--snapshot-dir <path>

# Run tests
cd test/sglang && pytest snapshot/ agents/ -v
cd test/registered/radix_cache && pytest test_mamba_radix_cache_comprehensive.py -v

# Run test phases (on GPU instance)
source test/phases/config.sh
# Follow phase docs in test/phases/phase-XX-*.md

# Lint
pre-commit run --all-files
```

## Current Status

### What's working
- Snapshot save/restore/list/get/delete fully implemented and wired
- MambaRadixCache with dual LRU, tombstone nodes, COW (1239 lines)
- Agent framework with 4 built-in tools (REST + WebSocket)
- 3-tier memory (VRAM/RAM/Disk) with automatic transitions
- Startup snapshot warm restore (Gap 3) — merged in PR #6
- `create_new_request` restore flow — merged in PR #4
- `fill_ids` sync on restore — merged in PR #4
- Unit tests: snapshot (46 pass), agents (37 pass)
- Test phases 0/2/3/5 PASS (16/16 radix cache tests)

### What's blocked
- Server test phases (1/4/6/7/8) blocked by sm75+ GPU requirement
- Phase 7 is the critical one — validates all 3 gap fixes from PRs #4 and #6

### What's next (priority order)
1. Provision sm75+ GPU instance → run server test phases
2. Phase 7 (snapshot system e2e) — validates gap fixes
3. Phase 3.4 final audit (KHA-6)
4. Performance optimizations (KHA-7 through KHA-11) — after server validation
5. Module documentation (KHA-12, KHA-13)
6. Phase3 docs cleanup (KHA-14)

## Architecture Notes

- Mamba SSM state is NOT a cache optimization — it's the model's compressed representation of everything processed. Persisting it restores the model's *understanding*, not just performance.
- `MambaRadixCache` has dual trees: full KV (attention) and Mamba state. Tombstone nodes keep KV but evict Mamba state.
- `TierManager` handles VRAM (active) → RAM (warm) → Disk (cold) transitions with configurable timeouts.
- Snapshot metadata now includes `fill_ids` (token ID array) — critical for correct restore. Snapshots without fill_ids are rejected.
- The scheduler's `create_new_request=True` flow enables stateless client pattern — client doesn't need to hold a request ID across sessions.

## Known Issues / Tech Debt

1. **sm75+ blocker**: V100 = sm70. FLA Mamba2 CUDA kernels + FlashInfer require sm75+. All server-phase testing blocked without appropriate GPU.
2. **`docs/stateful_mamba/README.md` stale**: Still says Phase 2 "Coming Soon" but restoration is implemented.
3. **~14 server args undocumented** in user-facing docs (snapshot_retention_count, snapshot_trigger_policy, agent_tool_timeout, etc.)
4. **License ambiguity**: SKILL.md says MIT; upstream SGLang is Apache 2.0. Fork license should be explicitly stated.
5. **Author inconsistency**: SKILL.md says "Orchestra Research"; GitHub org is "KHAEntertainment".
6. **`fix/mamba-model-config` branch diverged** from its remote (behind 1).
7. **Test phase line numbers** in `codemap.md` will drift as code changes.

## Agent Guidelines

- **Flag**: `--enable-snapshot-persistence` (NOT `--enable-mamba-snapshots` — that flag does NOT exist)
- Use `config.sh` in `test/phases/` as single source of truth for paths/ports when running phases
- Tombstone nodes only created on INTERNAL nodes (not leaves). Must insert extension first, then evict base.
- Sequences must be >= 64 tokens for `mamba_branching_seqlen` (chunk_size=64)
- `query_start_loc`: use `torch.tensor([0, L, 2L, ..., N*L])`, NOT `arange(N+1)`
- Match result field: `device_indices` (not `value`)
- dtype attribute: `cache_params.dtype.temporal` (not `.ssm_state_dtype`)
- Free req pool slots after inserting into radix cache

## Related Projects / Dependencies

- **Upstream SGLang**: https://github.com/sgl-project/sglang
- **Linear project**: [SGLang - Mamba](https://linear.app/khaentertainment/project/sglang-mamba-e14f2152be8d)
- **Clarit.ai**: Parent product — "Mamba models were built to remember. Clarit is the infrastructure that lets them."
- **GCloud GPU instance**: See CLAUDE.md for connection details (sanitized). Core-memory has actual values.

## Context & Project Management Access

**Do not use the Linear CLI or native Linear MCP directly.**

All Linear interaction goes through **core-memory MCP** via `execute_integration_action`. This builds accumulated project memory over time.

**Session start pattern:**
```
# 1. Search memory for prior context
memory_search("sglang-mamba")

# 2. Check Linear backlog via core-memory
execute_integration_action(
  accountId: "0b4764e3-a793-4537-89b7-b26eff7b7675",
  action: "linear_search_issues",
  params: { query: "sglang-mamba", first: 20 }
)
```

**Available Linear actions through core-memory:**
- `linear_search_issues` — search/filter by project, label, state, text
- `linear_create_issue` — create with projectId, priority, labels, parent
- `linear_update_issue` — update state, project, labels (requires internal UUID, not KHA-XX)
- `linear_create_project` / `linear_update_project` — manage projects
- `linear_create_label` — create team labels
- `linear_list_cycles` — list sprints/cycles
- `linear_get_viewer` — get authenticated user

**Linear accountId:** `0b4764e3-a793-4537-89b7-b26eff7b7675`
**Linear projectId (SGLang - Mamba):** `f7f1cb8c-c4cd-4b63-83f6-58b9ddba6ce8`
**Linear teamId (KHAEntertainment):** `1ee12f51-86fc-4bce-8cad-845d8a67bfa9`

Additional context (decisions, notes, test results) lives in **Linear project docs** and **core-memory**.
Local ephemeral context (session notes, VM-specific config) lives in `docs/migration-prep/` — gitignored, not linked from docs.
