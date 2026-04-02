# CLAUDE.md — Engram

Fork of [SGLang](https://github.com/sgl-project/sglang) adding **Mamba SSM snapshot persistence** for fast multi-turn inference.
Repo: `github.com/clarit-ai/engram`

---

## Session Start (REQUIRED)

Run these before touching code:

```bash
# 1. Surface project state + history from Linear and prior sessions
memory_search("sglang mamba")
memory_search("sglang mamba backlog issues")
# 2. Read this file  ← you are here
# 3. Check .agent/ for local session notes (gitignored, may not exist)
```

---

## Current Machine — RunPod A100

| | |
|---|---|
| **Path** | `/home/jeanclawdai/sglang-mamba/` |
| **GPU** | NVIDIA A100-SXM4-80GB (sm80) |
| **Primary model** | `/mnt/models/granite-4.0-h-tiny` |
| **Fallback model** | `/home/jeanclawdai/models/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| **Install** | `pip install -e "python/"` (system-wide, already done) |

Testing priority: granite → Nemotron (if OOM) → granite-q4 (comparison)

---

## Key Commands

```bash
# Start server (primary model)
source test/phases/config.sh
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --enable-snapshot-persistence \
  --snapshot-dir $SNAPSHOT_DIR \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port $SERVER_PORT

# Run a test phase (example: phase 1)
source test/phases/config.sh
bash test/phases/phase-01-stateless-inference-baseline.md  # follow phase doc

# Unit tests (no server needed)
pytest python/sglang/test/srt/test_mamba_pool_extended.py -v
pytest python/sglang/test/srt/test_mamba_radix_cache_comprehensive.py -v
pytest python/sglang/test/srt/test_mamba_metadata.py -v

# Lint
pre-commit run --all-files
```

**Correct server flags:** `--enable-snapshot-persistence` (NOT `--enable-mamba-snapshots`)

---

## Test Phase Status

| Phase | Description | Result |
|-------|-------------|--------|
| 0 | Environment verification | **PASS** |
| 1 | Stateless inference baseline | INCOMPLETE — run first on A100 |
| 2 | MambaPool unit tests | **PASS** (5/5) |
| 3 | MambaRadixCache gauntlet | **PASS** (16/16) |
| 4 | Live server — no_buffer strategy | INCOMPLETE |
| 5 | Mamba2Metadata integrity | **PASS** (5/5) |
| 6 | extra_buffer strategy | INCOMPLETE |
| 7 | Snapshot system e2e | INCOMPLETE — validates Gap fixes PRs #4 #6 |
| 8 | Gauntlet stress tests | INCOMPLETE |

Resume order: **1 → 4 → 7 → 6 → 8**. Stop at first failure and diagnose.
Phase docs + config: `test/phases/` | Results: `test/phases/results/`

---

## Open Work

- **PR #7** (open): docs resync — update agent instructions, create AGENTS.md, fix stale docs
  `https://github.com/clarit-ai/engram/pull/7`
- **KHA-6** (Backlog): Phase 3.4 — Final Audit
  `https://linear.app/khaentertainment/issue/KHA-6`
- **KHA-5 / KHA-15 / KHA-16** — show Backlog in Linear but work is shipped in PR #6 (needs status update)

---

## Linear & GitHub Access

**All Linear and GitHub access goes through Core Memory MCP — never the native Linear MCP, Linear CLI, or direct gh calls for project tracking.**

```python
# Linear — list/search issues
execute_integration_action(
  accountId="0b4764e3-a793-4537-89b7-b26eff7b7675",
  action="linear_search_issues",
  parameters={"query": "...", "first": 50}
)

# GitHub — list PRs
execute_integration_action(
  accountId="a1b3e383-1a13-4e4c-ace4-55797d394674",
  action="list_pull_requests",
  parameters={"owner": "clarit-ai", "repo": "engram", "state": "all"}
)
```

Linear project ID: `f7f1cb8c-c4cd-4b63-83f6-58b9ddba6ce8`

---

## Key Source Files

| File | Purpose |
|------|---------|
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | MambaRadixCache, dual LRU, tombstone nodes, COW |
| `python/sglang/srt/mem_cache/memory_pool.py` | HybridReqToTokenPool, MambaPool, HybridLinearKVPool |
| `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` | ForwardMetadata, Mamba2Metadata |
| `python/sglang/srt/snapshot/mamba_snapshot.py` | MambaSnapshotMetadata, save/load |
| `python/sglang/srt/snapshot/tier_manager.py` | WARM tier preload on startup |
| `python/sglang/srt/managers/scheduler.py` | handle_save/restore_snapshot, create_new_request |
| `test/phases/config.sh` | Single source of truth for MODEL_PATH, PORT, SNAPSHOT_DIR |


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
