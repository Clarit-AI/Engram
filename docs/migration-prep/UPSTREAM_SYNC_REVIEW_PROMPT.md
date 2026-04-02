# Upstream SGLang Sync Review — Commit Triage Prompt

## Your Role

You are reviewing ~1,400 commits from upstream `sgl-project/sglang` that our fork `clarit-ai/engram` has fallen behind on. Your job is to triage these commits into actionable categories so we can plan a safe merge that preserves our custom Mamba state persistence features.

This is a READ-ONLY investigation. Do not modify any files. Produce a structured report.

---

## Step 0: Establish the Commit Range

```bash
# Add upstream if not already present
git remote add upstream https://github.com/sgl-project/sglang.git 2>/dev/null || true
git fetch upstream main

# Find the merge base (where our fork diverged or last synced)
MERGE_BASE=$(git merge-base main upstream/main)
echo "Merge base: $MERGE_BASE"

# Count commits behind
git rev-list --count $MERGE_BASE..upstream/main
echo "commits behind upstream"

# Count our commits ahead
git rev-list --count $MERGE_BASE..main
echo "commits ahead (our custom work)"
```

---

## Step 1: Identify Commits That Touch OUR Files (HIGH PRIORITY)

These are the files we've heavily modified or created. Any upstream commit touching these is a potential conflict.

### Files we OWN (created by us, not in upstream):
```
python/sglang/srt/snapshot/                          # entire directory — our snapshot system
python/sglang/srt/mem_cache/mamba_radix_cache.py      # 1233 lines, our MambaRadixCache
python/sglang/snapshot.py                             # high-level SnapshotManager API
python/sglang/srt/agents/                             # our agent framework
test/phases/                                          # our test infrastructure
test/registered/radix_cache/test_mamba_radix_cache_*.py
test/registered/radix_cache/test_mamba_unittest.py
test/sglang/snapshot/
test/sglang/agents/
```

### Files we MODIFIED (exist upstream, we added to them):
```
python/sglang/srt/managers/scheduler.py               # handle_save_snapshot, handle_restore_snapshot, create_new_request restore flow
python/sglang/srt/managers/io_struct.py                # RestoreSnapshotReqOutput, RestoreSnapshotReqInput, session_params, lora normalization
python/sglang/srt/mem_cache/memory_pool.py             # MambaPool, HybridReqToTokenPool, HybridLinearKVPool
python/sglang/srt/server_args.py                       # --enable-snapshot-persistence, --snapshot-dir, --mamba-scheduler-strategy, etc.
python/sglang/srt/layers/attention/mamba/mamba2_metadata.py  # ForwardMetadata, Mamba2Metadata
python/sglang/srt/entrypoints/http_server.py           # snapshot REST endpoints
```

Run this to find upstream commits touching our modified files:

```bash
MERGE_BASE=$(git merge-base main upstream/main)

# High-conflict files (we modified these)
git log --oneline $MERGE_BASE..upstream/main -- \
    python/sglang/srt/managers/scheduler.py \
    python/sglang/srt/managers/io_struct.py \
    python/sglang/srt/mem_cache/memory_pool.py \
    python/sglang/srt/server_args.py \
    python/sglang/srt/layers/attention/mamba/mamba2_metadata.py \
    python/sglang/srt/entrypoints/http_server.py
```

For each commit that touches these files, record:
- Commit hash + one-line summary
- Which of our files it touches
- Whether it's additive (new feature), a refactor (renames/moves), a bugfix, or a deletion
- Conflict risk: HIGH (changes same functions we modified) / MEDIUM (changes same file, different area) / LOW (nearby but no overlap)

---

## Step 2: Mamba/SSM-Specific Commits (STRATEGIC PRIORITY)

This is the intelligence-gathering phase. We need to know if upstream is building features that overlap with ours.

```bash
MERGE_BASE=$(git merge-base main upstream/main)

# Search for Mamba-related commits
git log --oneline $MERGE_BASE..upstream/main --grep="mamba" -i
git log --oneline $MERGE_BASE..upstream/main --grep="ssm" -i
git log --oneline $MERGE_BASE..upstream/main --grep="hybrid" -i
git log --oneline $MERGE_BASE..upstream/main --grep="state.space" -i
git log --oneline $MERGE_BASE..upstream/main --grep="granite" -i
git log --oneline $MERGE_BASE..upstream/main --grep="jamba" -i
git log --oneline $MERGE_BASE..upstream/main --grep="nemotron" -i
git log --oneline $MERGE_BASE..upstream/main --grep="snapshot" -i
git log --oneline $MERGE_BASE..upstream/main --grep="persistent" -i

# Also search for changes to Mamba-related source paths
git log --oneline $MERGE_BASE..upstream/main -- \
    'python/sglang/srt/layers/attention/mamba/' \
    'python/sglang/srt/models/*mamba*' \
    'python/sglang/srt/models/*hybrid*' \
    'python/sglang/srt/models/*granite*' \
    'python/sglang/srt/models/*jamba*'
```

For each Mamba/SSM-related commit, classify:
- **OVERLAP**: Implements something we already have (e.g., state caching, session persistence)
- **COMPLEMENT**: Enhances Mamba support in ways we can leverage (e.g., new model support, kernel improvements)
- **CONFLICT**: Changes Mamba internals in ways that break our assumptions
- **NEUTRAL**: Mentions Mamba but doesn't affect our work

Pay special attention to:
1. Any changes to `MambaPool`, `MambaCacheParams`, or Mamba state management
2. Any new radix cache strategies or cache eviction changes
3. Any session/conversation persistence features
4. Any new Mamba/hybrid model registrations
5. Changes to how `fill_ids` or `origin_input_ids` are handled in the scheduler

---

## Step 3: Structural/Refactor Commits (MERGE RISK)

Large refactors are the most dangerous for merge conflicts even when they don't touch Mamba code.

```bash
MERGE_BASE=$(git merge-base main upstream/main)

# Find commits with large diffs (>500 lines changed)
git log --oneline --shortstat $MERGE_BASE..upstream/main | awk '/files? changed/ { split($0, a, ","); for(i in a) { if(a[i] ~ /insertion/) { gsub(/[^0-9]/, "", a[i]); ins=a[i] } if(a[i] ~ /deletion/) { gsub(/[^0-9]/, "", a[i]); del=a[i] } } if(ins+del > 500) print prev" | +"ins" -"del } { prev=$0 }'

# Find file renames/moves that could break our imports
git log --oneline --diff-filter=R $MERGE_BASE..upstream/main -- 'python/sglang/'

# Find deleted files
git log --oneline --diff-filter=D $MERGE_BASE..upstream/main -- 'python/sglang/'
```

For major refactors, note:
- What was renamed/moved/deleted
- Whether it affects any of our import paths
- Whether it changes class hierarchies we inherit from or interfaces we implement

---

## Step 4: Dependency & Build Changes

```bash
MERGE_BASE=$(git merge-base main upstream/main)

git log --oneline $MERGE_BASE..upstream/main -- \
    'pyproject.toml' \
    'setup.py' \
    'setup.cfg' \
    'requirements*.txt' \
    'python/sglang/version.py'
```

Note any changes to:
- flashinfer version requirements
- torch version requirements
- triton version
- New dependencies added
- Python version constraints

---

## Step 5: Output Format

Produce a single markdown report with these sections:

### 1. Executive Summary
- Total commits behind
- Total commits touching our files (count)
- Total Mamba/SSM-related commits (count)
- Overall merge risk assessment: LOW / MEDIUM / HIGH / CRITICAL
- One paragraph on whether upstream is moving toward or away from our feature set

### 2. Direct Conflict Table
| Commit | File(s) | Type | Risk | Notes |
|--------|---------|------|------|-------|

### 3. Mamba/SSM Intelligence
For each Mamba-related commit, one paragraph explaining what it does and how it relates to our work. Group by classification (OVERLAP / COMPLEMENT / CONFLICT / NEUTRAL).

**Critical question to answer:** Is upstream building state persistence, snapshot systems, or session continuity for Mamba models? If yes, how does their approach compare to ours?

### 4. Dangerous Refactors
List any commits that rename, move, or delete files/classes/functions that we import or extend.

### 5. Dependency Changes
Table of version bumps and new dependencies.

### 6. Recommended Merge Strategy
Based on your findings, recommend one of:
- **REBASE**: If conflicts are minimal and mostly additive
- **MERGE COMMIT**: If conflicts are moderate but manageable
- **CHERRY-PICK**: If upstream has diverged significantly and we should selectively take commits
- **REVIEW-THEN-MERGE**: If there are overlapping features that need architectural decisions first

Include a suggested ordering if cherry-picking is recommended.

---

## Context: What Our Fork Does (for reference)

Our fork adds **Mamba SSM state snapshot persistence** to SGLang:

1. **Snapshot System** (`python/sglang/srt/snapshot/`): Save/restore Mamba hidden states to disk via safetensors. Configurable retention policies. Hooks in scheduler lifecycle.

2. **3-Tier Memory Hierarchy**: VRAM (MambaPool, instant) → Host RAM (MambaHostPool, 10-50ms) → Disk (snapshots, 100-500ms). LRU eviction between tiers.

3. **MambaRadixCache** (`mamba_radix_cache.py`): Dual-tree radix cache with both KV cache + Mamba state. Copy-on-write for states. Tombstone nodes (KV kept, Mamba evicted).

4. **Scheduler Extensions**: `handle_save_snapshot`, `handle_restore_snapshot`, `create_new_request` with restored Mamba state injection. `fill_ids` sync on restore.

5. **Startup Restore**: Server preloads latest snapshots into WARM tier on boot.

6. **Server Args**: `--enable-snapshot-persistence`, `--snapshot-dir`, `--mamba-scheduler-strategy`, `--snapshot-retention-count`, `--snapshot-trigger-policy`, etc.

7. **Agent Framework** (`python/sglang/srt/agents/`): Tool registry, multi-format parser, execution engine, built-in tools.

Our server flags, snapshot REST endpoints, and test infrastructure are additions — they don't replace upstream code, they extend it. The main conflict surfaces are `scheduler.py`, `io_struct.py`, `memory_pool.py`, and `server_args.py` where we added methods and fields to existing upstream classes.

---

## Important Notes

- Do NOT modify any files. This is read-only analysis.
- If the commit count is truly ~1,400, focus your detailed analysis on Steps 1-2 (our files + Mamba commits). For Step 3, only flag commits with >500 lines changed.
- Time-box: spend ~60% of effort on Step 1-2, ~25% on Step 3, ~15% on Steps 4-5.
- Save the report to `docs/migration-prep/UPSTREAM_SYNC_REPORT.md`
