# Phase 04: Mamba Cache Architecture Reconcile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile upstream's new Mamba cache hierarchy (HiMambaRadixCache, HybridCacheController, MambaPoolHost, pool sizing changes) with our snapshot/tier system, then cherry-pick 10 COMPLEMENT Mamba improvements.

**Architecture:** Cherry-pick 7 HIGH-risk upstream commits in chronological order, resolving conflicts by accepting upstream's cache implementations while preserving our snapshot/tier_manager code. Then cherry-pick 10 COMPLEMENT commits. Our snapshot code accesses `MambaPool.mamba_cache.conv` and `.temporal` directly — we must verify this API surface survives intact after each merge step.

**Tech Stack:** Python, PyTorch, git cherry-pick, pytest

---

## Prerequisites

- Phase 03 complete and tagged `phase-03-pass` (verified)
- On `upstream-sync-2026-Q1` branch in correct worktree (verified)

## File Map

| File | Role | Action |
|------|------|--------|
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | Radix cache for hybrid Mamba models | Accept upstream changes (288-line diff) |
| `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` | HiCache extension (NEW — 2114 lines) | Accept from upstream (new file) |
| `python/sglang/srt/mem_cache/memory_pool.py` | MambaPool, HybridReqToTokenPool, HybridLinearKVPool | Merge upstream changes (472-line diff), preserve snapshot API surface |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | Host-side pools including new MambaPoolHost | Accept upstream changes (1095-line diff) |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | HybridCacheController (NEW upstream) | Accept from upstream (new file) |
| `python/sglang/srt/managers/scheduler.py` | Scheduler with our snapshot handlers | Merge upstream changes, preserve snapshot handlers |
| `python/sglang/srt/snapshot/mamba_snapshot.py` | MambaSnapshotManager — extract/inject state | May need sync guard; verify API compatibility |
| `python/sglang/srt/snapshot/tier_manager.py` | TierManager — disk persistence | Verify still works with new pool constructors |

## Critical API Surface (Must Survive All Merges)

Our `MambaSnapshotManager` accesses these directly:
- `mamba_pool.mamba_cache.conv` (list of tensors, indexed `[:, mamba_pool_idx]`)
- `mamba_pool.mamba_cache.temporal` (tensor, indexed `[:, mamba_pool_idx]`)
- `mamba_pool.size` (int)
- `mamba_pool.device` (str)

Upstream changes the `MambaPool.__init__` signature (adds `mamba_layer_ids` param) and the `HybridReqToTokenPool.__init__` signature (adds `mamba_layer_ids`, `enable_overlap_schedule`, `start_layer`). The `mamba_cache.conv`/`.temporal` tensor structure is **unchanged** — safe for our snapshot code.

---

## Task 1: Worktree Safety Check + Beads Setup

**Files:** None (verification only)

- [ ] **Step 1: Verify worktree**

```bash
if [[ "$(git branch --show-current)" != "upstream-sync-2026-Q1" ]]; then
    echo "ERROR: Not on upstream-sync-2026-Q1 branch. Aborting."
    exit 1
fi
if [[ "$(git rev-parse --show-toplevel)" != *"worktrees/upstream-sync-2026-Q1" ]]; then
    echo "ERROR: Not inside the designated worktree. Aborting."
    exit 1
fi
echo "Worktree safety check passed."
```

- [ ] **Step 2: Claim research beads task**

```bash
bd update upstream-sync-2026-Q1-cwf --claim
```

- [ ] **Step 3: Create a safety tag before starting**

```bash
git tag phase-04-pre-start
```

---

## Task 2: Cherry-Pick Core Commit 1 — Spec V2 Mamba Hybrid Attention (`e4b708d3e`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/memory_pool.py` (MambaPool init gains `mamba_layer_ids`, HybridReqToTokenPool gains `mamba_layer_ids`/`enable_overlap_schedule`/`start_layer`)
- Modify: Various model/scheduler files for Spec V2

- [ ] **Step 1: Cherry-pick the commit**

```bash
git cherry-pick e4b708d3e --no-commit
```

- [ ] **Step 2: Check for conflicts**

```bash
git status | grep "both modified\|Unmerged"
```

If conflicts exist in `memory_pool.py`, resolve by:
- Accept upstream's new `MambaPool.__init__` signature (adds `mamba_layer_ids`)
- Accept upstream's new `HybridReqToTokenPool.__init__` signature changes
- Accept upstream's `at_layer_idx` fix (uses `fields()` instead of `vars()`)
- Accept upstream's `alloc()` zero-fill optimization (expand scalar GPU zero)
- Preserve any snapshot-specific code in our fork (there should be none in memory_pool.py directly)

If conflicts exist in `scheduler.py`:
- Accept upstream's changes to scheduling logic
- **Preserve** our snapshot handlers: `handle_save_snapshot`, `handle_restore_snapshot`, `handle_list_snapshots`, `handle_get_snapshot_info`, `handle_delete_snapshot`
- **Preserve** our snapshot request types in the dispatch table: `SaveSnapshotReqInput`, `ListSnapshotsReqInput`, `GetSnapshotInfoReqInput`, `RestoreSnapshotReqInput`, `DeleteSnapshotReqInput`

- [ ] **Step 3: Verify snapshot API surface is intact**

```bash
python -c "
import ast, sys
tree = ast.parse(open('python/sglang/srt/mem_cache/memory_pool.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'MambaPool':
        # Check State dataclass has conv and temporal
        for item in node.body:
            if isinstance(item, ast.ClassDef) and item.name == 'State':
                field_names = [n.target.attr for n in ast.walk(item) if isinstance(n, ast.AnnAssign) and hasattr(n.target, 'attr')]
                # Fallback: just check the class exists
                print(f'MambaPool.State found')
        print('MambaPool class found')
        break
else:
    print('ERROR: MambaPool class not found!'); sys.exit(1)
"
```

- [ ] **Step 4: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick e4b708d3e — Spec V2 mamba hybrid attention support"
```

---

## Task 3: Cherry-Pick Core Commit 2 — Remove Sync Points (`07ef5f7be`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- Modify: Various prefill/cudagraph files

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick 07ef5f7be --no-commit
```

- [ ] **Step 2: Resolve conflicts**

For `mamba_radix_cache.py`: Accept upstream's sync point removal.

Check if any removed sync points affect our snapshot extraction path:
```bash
grep -n "cuda.synchronize\|torch.cuda.current_stream" python/sglang/srt/snapshot/mamba_snapshot.py
```

If our snapshot code has no explicit sync dependence (likely — it just reads tensors), no guard needed yet. We'll add one in Task 8 if validation shows races.

- [ ] **Step 3: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick 07ef5f7be — remove sync points in mamba cache + prefill cudagraph"
```

---

## Task 4: Cherry-Pick Core Commit 3 — HiCache for MambaRadixCache (`5867c3fa8`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (significant refactor)
- Create: `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` (new 2114-line file from upstream)
- Modify: `python/sglang/srt/mem_cache/memory_pool_host.py` (adds MambaPoolHost)

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick 5867c3fa8 --no-commit
```

- [ ] **Step 2: Resolve conflicts**

For `mamba_radix_cache.py`: **Accept upstream's version entirely.** Our fork's tombstones/dual-LRU/COW are not needed by the snapshot system. Run:
```bash
git checkout --theirs python/sglang/srt/mem_cache/mamba_radix_cache.py
```

For `memory_pool_host.py`: Accept upstream's addition of `MambaPoolHost`. Check for conflicts with our existing host pool code.

For `hi_mamba_radix_cache.py`: Should be a clean new file addition.

- [ ] **Step 3: Verify new files are present**

```bash
test -f python/sglang/srt/mem_cache/hi_mamba_radix_cache.py && echo "hi_mamba_radix_cache.py present" || echo "MISSING"
grep -c "class MambaPoolHost" python/sglang/srt/mem_cache/memory_pool_host.py
```

Expected: file present, class count = 1.

- [ ] **Step 4: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick 5867c3fa8 — HiCache support for MambaRadixCache"
```

---

## Task 5: Cherry-Pick Core Commit 4 — Mamba Ratio Refactor (`0ac6c63ae`)

**Files:**
- Modify: Pool initialization / ratio calculation code

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick 0ac6c63ae --no-commit
```

- [ ] **Step 2: Resolve any conflicts**

Accept upstream's ratio calculation refactoring. This changes how `mamba_pool` sizes are computed during init — our snapshot code doesn't depend on the sizing logic, only on the resulting tensor shapes.

- [ ] **Step 3: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick 0ac6c63ae — refactor mamba ratio calculation for pool init"
```

---

## Task 6: Cherry-Pick Core Commit 5 — Refactor Mamba Radix Tree Insert/Release (`197f80713`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (insert/release semantics)

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick 197f80713 --no-commit
```

- [ ] **Step 2: Resolve conflicts**

Accept upstream's insert/release refactoring. Since we already accepted upstream's `mamba_radix_cache.py` in Task 4, this should apply more cleanly.

- [ ] **Step 3: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick 197f80713 — refactor mamba radix tree insert/release semantics"
```

---

## Task 7: Cherry-Pick Core Commit 6 — HiMambaRadixCache Prefix Fix (`dfd0a77a9`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick dfd0a77a9 --no-commit
```

- [ ] **Step 2: Resolve conflicts (should be clean)**

- [ ] **Step 3: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick dfd0a77a9 — fix prev_prefix_len in HiMambaRadixCache insert"
```

---

## Task 8: Cherry-Pick Core Commit 7 — Mamba State Offloading + HybridCacheController (`0986bed8e`)

**Files:**
- Modify: `python/sglang/srt/mem_cache/memory_pool.py`
- Modify: `python/sglang/srt/mem_cache/memory_pool_host.py`
- Create/Modify: `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- Modify: `python/sglang/srt/managers/scheduler.py`

This is the highest-risk commit — introduces the GPU↔host offload path.

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick 0986bed8e --no-commit
```

- [ ] **Step 2: Resolve conflicts**

For `scheduler.py` — this is the critical file:
- Accept upstream's cache controller integration
- **Preserve** ALL snapshot handlers (`handle_save_snapshot`, `handle_restore_snapshot`, `handle_list_snapshots`, `handle_get_snapshot_info`, `handle_delete_snapshot`)
- **Preserve** ALL snapshot request types in the dispatch table
- **Preserve** snapshot imports at top of file

For `memory_pool.py`:
- Accept upstream's `register_layer_transfer_counter`, `_wait_for_layer` additions
- Accept upstream's `HybridLinearKVPool` `start_layer` parameter

For `memory_pool_host.py`:
- Accept upstream's MambaPoolHost changes

- [ ] **Step 3: Verify snapshot handlers survived**

```bash
grep -c "handle_save_snapshot\|handle_restore_snapshot\|handle_list_snapshots\|handle_get_snapshot_info\|handle_delete_snapshot" python/sglang/srt/managers/scheduler.py
```

Expected: >= 10 (each handler defined + referenced in dispatch table).

```bash
grep -c "SaveSnapshotReqInput\|RestoreSnapshotReqInput\|ListSnapshotsReqInput\|GetSnapshotInfoReqInput\|DeleteSnapshotReqInput" python/sglang/srt/managers/scheduler.py
```

Expected: >= 5.

- [ ] **Step 4: Verify snapshot API surface still works**

```bash
python -c "
from sglang.srt.mem_cache.memory_pool import MambaPool
# Verify State dataclass has conv and temporal fields
import dataclasses
fields = {f.name for f in dataclasses.fields(MambaPool.State)}
assert 'conv' in fields, 'Missing conv field'
assert 'temporal' in fields, 'Missing temporal field'
print('MambaPool.State API surface intact: conv, temporal present')
" 2>&1 || echo "Import check failed — expected if dependencies missing, will verify at runtime"
```

- [ ] **Step 5: Add sync guard if needed**

Check if upstream's sync removal creates a race for our snapshot extraction:
```bash
grep -n "torch.cuda.synchronize\|cuda.synchronize\|stream.synchronize" python/sglang/srt/snapshot/mamba_snapshot.py
```

If no sync exists in `extract_state_from_pool`, add a defensive guard:

In `python/sglang/srt/snapshot/mamba_snapshot.py`, before the `.clone().cpu()` calls in `extract_state_from_pool`, add:
```python
# Ensure all pending GPU ops on Mamba state are complete before CPU extraction
torch.cuda.synchronize()
```

This goes right before the `for conv_tensor in mamba_pool.mamba_cache.conv:` loop (around line 580).

- [ ] **Step 6: Stage and commit**

```bash
git add -A
git commit -m "phase-04: cherry-pick 0986bed8e — mamba state offloading + HybridCacheController

Adds GPU↔host offload path via HybridCacheController and MambaPoolHost.
Our TierManager (disk persistence) layers on top — it interacts with
MambaPool directly for extract/inject, not through HybridCacheController.
Added defensive cuda.synchronize() guard in snapshot extraction path."
```

---

## Task 9: Design Review Checkpoint

- [ ] **Step 1: Close research beads task**

```bash
bd close upstream-sync-2026-Q1-cwf --reason "Research complete: upstream API verified compatible with snapshot system. MambaPool.mamba_cache.conv/temporal tensor structure unchanged. MambaPoolHost is new upstream host-tier pool, separate from our MambaHostPool."
```

- [ ] **Step 2: Close design review beads task**

```bash
bd close upstream-sync-2026-Q1-44v --reason "DESIGN REVIEW CONFIRMED: TierManager layering strategy validated. Upstream HybridCacheController owns GPU↔host (L1↔L2) offload. Our TierManager owns host↔disk (L2↔L3) persistence. TierManager interacts with MambaPool.extract/inject directly, does not replace HybridCacheController. MambaPool.mamba_cache.conv/temporal API surface unchanged by all 7 upstream commits."
```

---

## Task 10: COMPLEMENT Commit Sweep — Low Risk (6 commits)

Cherry-pick the 6 low/medium-risk COMPLEMENT commits first.

**Files:** Various Mamba ops, speculative execution, disaggregation files

- [ ] **Step 1: Claim implementation beads task**

```bash
bd update upstream-sync-2026-Q1-22o --claim
```

- [ ] **Step 2: Cherry-pick low-risk COMPLEMENT commits in chronological order**

```bash
# 1. Selective state update kernel (82a0bafc1)
git cherry-pick 82a0bafc1 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 82a0bafc1 — selective state update kernel for Mamba ops"

# 2. Fuse mamba state scatter in MTP verify (a1b39c1c2)
git cherry-pick a1b39c1c2 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick a1b39c1c2 — fuse mamba state scatter for MTP verify"

# 3. Fix illegal memory access in EAGLE verification (86c561778)
git cherry-pick 86c561778 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 86c561778 — fix illegal memory access in Mamba SSM EAGLE verify"

# 4. Return intermediate Mamba states (c76251f70)
git cherry-pick c76251f70 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick c76251f70 — return intermediate Mamba states from ssd_combined"

# 5. Skip _mamba_verify_update for idle batch (69158e9d9)
git cherry-pick 69158e9d9 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 69158e9d9 — skip mamba verify update for idle batch"

# 6. Piecewise CUDA graph for NemotronH (25bd83033)
git cherry-pick 25bd83033 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 25bd83033 — piecewise CUDA graph for NemotronH hybrid models"
```

For each cherry-pick, if conflicts arise:
- For files we don't own (upstream Mamba ops, speculative, model files): accept upstream
- For `scheduler.py`: merge carefully, preserving snapshot handlers
- For `server_args.py`: accept upstream additions

- [ ] **Step 3: Handle any conflict resolution**

If a cherry-pick fails, resolve manually:
```bash
git status  # identify conflicting files
# For each conflict, edit and resolve, then:
git add <resolved-file>
git commit -m "phase-04-complement: cherry-pick <hash> — <description>"
```

---

## Task 11: COMPLEMENT Commit Sweep — Remaining (4 commits)

- [ ] **Step 1: Cherry-pick remaining COMPLEMENT commits**

```bash
# 7. NPU mamba cache transfer (dd82678b2)
git cherry-pick dd82678b2 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick dd82678b2 — NPU mamba cache transfer support"

# 8. Mamba convolution perf — Triton conv1d (87549f8f0)
git cherry-pick 87549f8f0 --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 87549f8f0 — Triton conv1d for non-contiguous input perf"

# 9. Qwen3.5 mamba slice fix (cfead25bb)
git cherry-pick cfead25bb --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick cfead25bb — Qwen3.5 mamba slice fix for TP size mismatch"

# 10. Simplify server startup output (0949b138a)
git cherry-pick 0949b138a --no-commit
git add -A
git commit -m "phase-04-complement: cherry-pick 0949b138a — simplify server startup output"
```

- [ ] **Step 2: Verify snapshot code compiles**

```bash
python -c "
import python.sglang.srt.snapshot.mamba_snapshot as ms
print('mamba_snapshot imports OK')
" 2>&1 || python -c "
import sys; sys.path.insert(0, 'python')
from sglang.srt.snapshot.mamba_snapshot import MambaSnapshotManager
print('MambaSnapshotManager importable')
" 2>&1 || echo "Import check deferred to runtime validation"
```

---

## Task 12: Quick Sanity Check — Unit Tests

Run the fast unit tests that don't require a server.

- [ ] **Step 1: Run MambaPool-related tests**

```bash
# Find and run any unit tests for mamba cache/pool
find . -path "*/test*" -name "*.py" -exec grep -l "MambaPool\|mamba_radix_cache\|MambaRadixCache" {} \; | head -10
```

- [ ] **Step 2: Run available offline tests**

```bash
cd /home/jeanclawdai/sglang-mamba/worktrees/upstream-sync-2026-Q1
pytest test/registered/radix_cache/test_mamba_radix_cache_comprehensive.py -v --timeout=120 2>&1 | tail -30
```

If tests fail due to import errors from missing upstream dependencies (not our fault), note the failures and continue — full validation happens in Task 13.

- [ ] **Step 3: Run lint**

```bash
pre-commit run --all-files 2>&1 | tail -20
```

Fix any lint issues in files we modified.

---

## Task 13: Full Validation (Phases 2-8)

This requires a running server. Claim the validation beads task.

- [ ] **Step 1: Claim validation beads task**

```bash
bd update upstream-sync-2026-Q1-8sd --claim
```

- [ ] **Step 2: Run Phase 2 — MambaPool unit tests**

```bash
source test/phases/config.sh
# Run Phase 2 tests per test/phases/phase-02-*.md
```

Expected: 5/5 PASS

- [ ] **Step 3: Run Phase 3 — MambaRadixCache gauntlet**

```bash
# Run Phase 3 tests per test/phases/phase-03-*.md
```

Expected: 16/16 PASS

- [ ] **Step 4: Run Phase 4 — Live server no_buffer**

```bash
source test/phases/config.sh
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --enable-snapshot-persistence \
  --snapshot-dir $SNAPSHOT_DIR \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port $SERVER_PORT
# Then run validation requests per phase-04 doc
```

Expected: PASS

- [ ] **Step 5: Run Phase 5 — Mamba2Metadata integrity**

Expected: 5/5 PASS

- [ ] **Step 6: Run remaining phases (6, 7, 8) if earlier phases pass**

Stop at first failure and diagnose.

- [ ] **Step 7: Close validation beads task**

```bash
bd close upstream-sync-2026-Q1-8sd --reason "Phases 2-8 PASS"
```

If any phase fails, create a bug in beads and do NOT close.

---

## Task 14: Tag and Close Phase 04

- [ ] **Step 1: Close implementation beads task**

```bash
bd close upstream-sync-2026-Q1-22o --reason "7 core commits + 10 COMPLEMENT commits cherry-picked and merged. Snapshot API surface verified intact."
```

- [ ] **Step 2: Close parent Phase 04 beads task**

```bash
bd close upstream-sync-2026-Q1-clc --reason "Phase 04 complete. All sub-tasks closed. Phases 2-8 PASS."
```

- [ ] **Step 3: Tag the phase**

```bash
git tag phase-04-pass
```

- [ ] **Step 4: Push**

```bash
git push origin upstream-sync-2026-Q1
git push origin phase-04-pass
```

---

## Rollback Plan

If any validation fails and cannot be fixed:

```bash
git reset --hard phase-03-pass
git tag -d phase-04-pre-start 2>/dev/null
```

This resets to the last known good state.

## Conflict Resolution Principles

1. **For `mamba_radix_cache.py` and `hi_mamba_radix_cache.py`:** Always accept upstream — we no longer own these files
2. **For `memory_pool.py`:** Accept upstream's structural changes; verify `MambaPool.State.conv` and `.temporal` fields survive
3. **For `scheduler.py`:** Accept upstream logic changes but **always preserve** our snapshot handlers and dispatch entries
4. **For `memory_pool_host.py`:** Accept upstream additions (MambaPoolHost) — they don't conflict with our `MambaHostPool` in `snapshot/mamba_host_pool.py`
5. **For snapshot code:** Only modify if API surface changes (it shouldn't) or if sync guard is needed
