# Engram Upstream Sync Playbook

> **Engram:** v0.2.0 (pre-merge; target v0.3.0)
> **Last updated:** 2026-04-24

> This document enables any coding agent to resolve upstream merge conflicts
> in the Engram fork without prior project context. Read this before touching
> any merge conflict.

## What is Engram?

Engram is a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang) that adds:
- **Mamba state snapshot persistence** — save/restore SSM hidden states across server restarts
- **3-tier memory management** — VRAM → Host RAM → Disk for Mamba conversation states
- **Agent tool-calling framework** — built-in tool execution loop with WebSocket streaming
- **Pure Mamba2 model support** — Engram can run pure SSM models (e.g., Codestral Mamba 7B) that upstream cannot

## How to identify Engram changes

### ENGRAM_MODIFIED headers
Every file the fork has modified has a header comment:
```
# ENGRAM_MODIFIED — <description>
```
Search: `grep -rl "ENGRAM_MODIFIED" .`

### BEGIN/END ENGRAM blocks
Every fork modification within a file is wrapped:
```
# --- BEGIN ENGRAM: <what this block does> ---
<fork-specific code>
# --- END ENGRAM ---
```
Search: `grep -rn "BEGIN ENGRAM" .`

### ENGRAM_CHANGED inline markers
Single-line modifications to upstream code:
```
# ENGRAM_CHANGED: <why this line was modified>
<modified line>
```

### Protected paths policy
`.engram/policy/protected-paths.json` lists all protected file patterns.
`scripts/policy/check_protected_paths.py` validates protection coverage.

## Conflict resolution rules

### Rule 1: Never delete ENGRAM blocks
If upstream changes code near a BEGIN/END ENGRAM block, preserve the Engram block intact. Accept the upstream change outside the block boundaries.

### Rule 2: Upstream changes INSIDE an ENGRAM block = escalate
If upstream modifies code that falls inside a BEGIN/END ENGRAM block, this is a semantic conflict. Do NOT auto-resolve. Flag for human review with:
- The file path
- The Engram block description
- What upstream changed and why (check the upstream commit message)
- Your recommendation for resolution

### Rule 3: ENGRAM_CHANGED lines need manual review
If upstream modifies a line marked with ENGRAM_CHANGED, compare the upstream change against the Engram modification. Usually the Engram change is a small addition (extra import, extra field) that can be re-applied on top of upstream's new version.

### Rule 4: CI workflow files — keep the fork guard
All workflow files have: `if: github.repository == 'sgl-project/sglang'`
During a sync, accept all upstream workflow changes BUT re-apply the fork guard. It's always a single `if:` line added under a job definition.

### Rule 5: Accept upstream for unmodified files
Any file WITHOUT an ENGRAM_MODIFIED header should accept upstream changes verbatim. No judgment needed.

### Rule 6: Added files are ours entirely
Files that exist only in Engram (not on upstream) are never touched by a sync. If upstream creates a file at the same path, treat it as Rule 2 (escalate).

## Typical conflict patterns

### server_args.py
Engram adds ~29 dataclass fields and ~170 argparse arguments for snapshot/tier/agent features. Upstream frequently adds new args too. Resolution: keep both — Engram's args go in their own BEGIN/END block, upstream's args go wherever upstream put them.

### scheduler.py
Engram adds ~1200 lines of Mamba state management methods. Upstream modifies the scheduler for performance/features. Resolution: Engram's methods are appended (not interleaved), so conflicts only happen if upstream changes the class signature or method ordering. Usually safe to rebase Engram's additions on top of upstream's changes.

### io_struct.py
Engram adds ~130 lines of snapshot/agent data structures. Same pattern as scheduler — pure additions, rarely conflicts with upstream content.

### model_runner_kv_cache_mixin.py
Engram adds ~31 lines for Mamba KV cache integration. This is close to upstream's memory management code. Conflicts here need careful review.

## Validation after merge

Run on a single GPU (A100 recommended):
```bash
scripts/validate-sync.sh
```
This tests: server startup, model loading, basic inference, snapshot save/restore, stateful recall, and KV cache behavior. If it passes, the merge is safe to push.

## Post-merge checklist

1. Run validation script — must pass
2. Close superseded sync issues on GitHub
3. Tag release per convention (v0.MINOR.PATCH)
4. Update test/phases/MODEL_MATRIX.md if model behavior changed
5. Push to origin
