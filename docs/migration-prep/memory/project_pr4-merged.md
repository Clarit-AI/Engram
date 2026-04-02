---
name: PR #4 snapshot restore state sync — merged
description: What PR #4 fixed, what CodeRabbit found, and the 3 final fixes applied before merge
type: project
---

## PR #4: fix/snapshot-restore-state-sync

**Merged**: 2026-03-28T03:25:03Z into `main` at clarit-ai/engram

### What it fixed

1. **Gap 1 (fill_ids sync)**: `MambaSnapshotMetadata` now stores `fill_ids` (token ID array). Captured at save, restored to `req.fill_ids` + `req.origin_input_ids` after injection. Prevents token stream desync after restore.

2. **Gap 2 (create_new_request)**: `POST /restore_snapshot` accepts `create_new_request: true`. Creates a new request backed by restored Mamba state — enables the stateless client pattern where the client doesn't hold a request ID across sessions.

### Final 3 fixes applied before merge (2026-03-28)

1. **`pool_freed` NameError** (`scheduler.py`): Moved `pool_freed = False` to right after `new_pool_idx = mamba_pool.alloc(1)` (before `inject_state_to_pool`). Previously, if `inject_state_to_pool` threw, the outer `except` would crash with `NameError` on `not pool_freed` — masking the real exception and skipping cleanup.

2. **inject-before-validate** (`scheduler.py` in-place restore path): Moved `fill_ids is None` check to before `inject_state_to_pool`. Previously a snapshot without fill_ids would overwrite the live request's Mamba state, then return failure — leaving the request corrupted.

3. **Hardcoded model name** (`phase-01-stateless-inference-baseline.md`): Changed heredoc from `'EOF'` to `EOF` and replaced hardcoded `granite-4.0-h-tiny` with `${MODEL_NAME}`.

### CodeRabbit review volume
59 total review comments across multiple cycles. All addressed in 15 commits. Key patterns found:
- Hardcoded `/home/bbrenner/sglang-mamba` paths in phase docs → fixed to `REPO_ROOT=$(git rev-parse --show-toplevel)`
- Hardcoded port 30000 → fixed to `$SERVER_PORT`
- `EmbeddingReqInput` fields were removed but consumers still used them → restored
- `device='cuda'` hardcoded → `device=self.device`

**Why:** Useful context if we revisit the snapshot system or need to understand why these patterns exist.
**How to apply:** If snapshot restore behaves unexpectedly, check fill_ids presence in metadata and pool_freed guard.
