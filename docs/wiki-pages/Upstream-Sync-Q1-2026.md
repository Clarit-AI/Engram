# Upstream Sync Q1 2026

> This page consolidates documentation originally in `docs/migration-prep/` in the repo.

## Context

- **Fork:** `clarit-ai/engram`
- **Upstream:** `sgl-project/sglang`
- **Delta:** 1,327 commits behind upstream, 97 ahead
- **Merge risk:** CRITICAL
- **Validation baseline:** Phases 0–8 must pass after the sync

Upstream has introduced overlapping abstractions (`SessionController`, `HiCache`, `HybridCacheController`, `HiMambaRadixCache`), but **does not** implement our product differentiators: disk snapshots, restore APIs, startup warm restore, or server-side Mamba state persistence.

## Strategy

Treat upstream's new Mamba/HiCache/session work as the new substrate, then re-port our snapshot persistence feature on top of it after explicit design reviews. Blind branch merging would damage restore semantics or upstream cache invariants.

## Phase Sequence

| Phase | Document | Complexity |
|-------|----------|------------|
| 1 | `PHASE_01_DEPENDENCY_BASELINE.md` | LOW-MEDIUM |
| 2 | `PHASE_02_OBSERVABILITY_IMPORT_REBASE.md` | LOW |
| 3 | `PHASE_03_SESSION_CONTROLLER_PORT.md` | HIGH |
| 4 | `PHASE_04_MAMBA_CACHE_ARCHITECTURE_RECONCILE.md` | HIGH |
| 5 | `PHASE_05_SCHEDULER_IDLE_AND_POOL_FIXES.md` | MEDIUM |
| 6 | `PHASE_06_SERVER_AND_HTTP_DRIFT.md` | MEDIUM |
| 6B | `PHASE_06B_BULK_UPSTREAM_MERGE.md` | MEDIUM |
| 7 | `PHASE_07_REAPPLY_SNAPSHOT_FEATURES.md` | HIGH |
| 8 | `PHASE_08_FULL_VALIDATION_AND_STRESS_TEST.md` | MEDIUM |

**Critical Path:** `PHASE_01 → PHASE_02 → PHASE_03 → PHASE_04 → PHASE_05 → PHASE_06 → PHASE_06B → PHASE_07 → PHASE_08`

No parallel phases — each builds on substrate stability. Within HIGH phases, Research and Validation agents can run sub-tasks in parallel.

**GPU provisioning note:** Phases 01 and 02 (dependency resolution and import fixups) can run without a GPU. Phase 03 onward requires an A100 for validation steps.

## Key Upstream Findings (as of 2026-03-28)

### Critical Answer

Upstream is not building disk-backed state persistence, snapshot APIs, or conversation restore for Mamba models. There are no upstream matches for `save_snapshot`, `restore_snapshot`, `SnapshotManager`, `--enable-snapshot-persistence`, or snapshot-specific server args in `upstream/main`.

### OVERLAP Areas

| Upstream Commit | Overlap Area | Risk |
|-----------------|-------------|------|
| `0986bed8e` | Mamba state offloading + `HybridCacheController` | HIGH |
| `5867c3fa8` | HiCache support for `MambaRadixCache` | HIGH |
| `197f80713` | Upstream now owns `mamba_radix_cache.py` | HIGH |
| `dfd0a77a9` | Fixes `HiMambaRadixCache` prefix insertion | HIGH |
| `07ef5f7be` | Mamba cache/prefill cudagraph plumbing changes | HIGH |
| `c6cb0c964` | Streaming sessions + `SessionAwareCache` | HIGH |
| `5acb45cf3` | Extracts `SessionController` from `Scheduler` | HIGH |

### Per-File Churn

| File | Upstream Commits |
|------|-----------------|
| `scheduler.py` | 61 |
| `server_args.py` | 105 |
| `memory_pool.py` | 17 |
| `io_struct.py` | 12 |
| `http_server.py` | 18 |
| `mamba2_metadata.py` | 0 |

## Snapshot Restoration Plan

### The Problem: Gap 3

`restore_snapshots_on_startup()` in `scheduler.py` was a stub — only logs found snapshots, does not restore any state.

### Solution: Pre-load to WARM Tier

After server restart, pre-load the latest snapshot for each conversation into the **WARM tier** (MambaHostPool / host RAM) so subsequent requests can fast-restore without disk I/O.

**Implementation:**
```python
def restore_snapshots_on_startup(self):
    """Pre-load the latest snapshots for all conversations into WARM tier (host RAM)."""
    if self.snapshot_manager is None:
        return

    conversations = self.snapshot_manager.list_conversations()
    if not conversations:
        return

    tier_manager = getattr(self.snapshot_manager, 'tier_manager', None)
    if tier_manager is None:
        logger.warning("TierManager not available, skipping startup restore")
        return

    restored_count = 0
    for conv_id in conversations:
        latest = self.snapshot_manager.get_latest_snapshot(conv_id)
        if not latest:
            continue

        turn_number, metadata = latest
        try:
            result = tier_manager.restore_conversation(conv_id, turn_number=turn_number)
            if result:
                logger.info(f"  Pre-loaded conv_id={conv_id} turn={turn_number} to WARM tier")
                restored_count += 1
        except Exception as e:
            logger.error(f"  Failed to pre-load conv_id={conv_id}: {e}")

    logger.info(f"Startup restore complete: {restored_count}/{len(conversations)} conversations pre-loaded")
```

**Key components:**
- `MambaSnapshotManager.list_conversations()` — Returns list of conversation IDs with snapshots
- `MambaSnapshotManager.get_latest_snapshot()` — Returns `(turn_number, MambaSnapshotMetadata)` for latest snapshot
- `TierManager.restore_conversation()` — Restores from WARM tier first, falls back to COLD; auto-promotes to WARM

## Gap Fixes Summary (PR #4)

### Gap 1: `fill_ids` sync after restore
`MambaSnapshotMetadata` now stores `fill_ids`, captured at save time and restored to `req.fill_ids` + `req.origin_input_ids` after injection.

### Gap 2: `create_new_request=True`
`POST /restore_snapshot` with `create_new_request: true` creates a new request backed by restored state, returning a fresh `rid`.

### Gap 3: Startup Restore (Gap 3)
Pre-load latest snapshots to WARM tier on server restart. Implemented in PR #6.

## Branch Strategy

- Worktree: `worktrees/upstream-sync-2026-Q1` branched from `main`
- Each phase ends as a discrete commit tagged `phase-N-pass`
- If a phase fails validation, reset worktree to previous `phase-(N-1)-pass` tag
- Do not push or merge to `main` until `PHASE_08` passes fully

## Related Documentation

- [Stateful Mamba Guide](./Stateful-Mamba-Guide.md) - Snapshot persistence system
- [Agent Framework](./Agent-Framework.md) - Tool-calling system
- [GitHub Repo](https://github.com/clarit-ai/engram)
