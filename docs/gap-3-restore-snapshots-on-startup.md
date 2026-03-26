# Gap 3: `restore_snapshots_on_startup` — Full Implementation Plan

## Status

**Stub only** — currently only logs found snapshots, does not restore any state.

---

## Context

After PR [#4](https://github.com/KHAEntertainment/sglang-mamba/pull/4) merged two critical fixes:

1. **Gap 1 (FIXED)**: `fill_ids` sync after restore — `MambaSnapshotMetadata` now stores `fill_ids`, captured at save time and restored to `req.fill_ids` + `req.origin_input_ids` after injection ([scheduler.py:1439-1443](python/sglang/srt/managers/scheduler.py#L1439-L1443))
2. **Gap 2 (FIXED)**: `create_new_request=True` — `POST /restore_snapshot` with `create_new_request: true` creates a new request backed by restored state, returning a fresh `rid` ([scheduler.py:1311-1385](python/sglang/srt/managers/scheduler.py#L1311-L1385))

---

## The Problem: Gap 3

### Current State

`restore_snapshots_on_startup()` in `scheduler.py` lines 1040-1074:

```python
def restore_snapshots_on_startup(self):
    """
    Restore the latest snapshots for all conversations on server startup.

    This enables continuity across server restarts.
    """
    if self.snapshot_manager is None:
        return

    logger.info("Attempting to restore snapshots from previous sessions...")

    conversations = self.snapshot_manager.list_conversations()

    if not conversations:
        logger.info("No previous snapshots found")
        return

    logger.info(f"Found {len(conversations)} conversation(s) with snapshots")

    # For now, we'll just log what we found
    # Full restoration requires request recreation which is complex
    # and should be implemented when the agent loop is added
    for conv_id in conversations:
        latest = self.snapshot_manager.get_latest_snapshot(conv_id)
        if latest:
            turn_number, metadata = latest
            logger.info(
                f"  - Conversation {conv_id}: latest snapshot at turn {turn_number}, "
                f"{metadata.token_count} tokens"
            )

    logger.info(
        "Snapshot restoration logged. Full state restoration will be "
        "implemented with agent loop in Phase 3."
    )
```

### What It Does

- Lists all conversations with snapshots on disk
- Logs the latest snapshot metadata for each
- **Does nothing else**

### What It Should Do

After a server restart, pre-load the latest snapshot for each conversation into the **WARM tier** (MambaHostPool / host RAM) so subsequent requests can fast-restore without disk I/O. This is a **partial implementation** — full cross-session request recreation requires more infrastructure.

---

## Deepwiki Analysis (Source)

From the Deepwiki code analysis:

> **`restore_snapshots_on_startup` is a stub.** It only **logs** what snapshots exist; the actual request recreation is explicitly deferred. The `get_latest_snapshot()` method only reads metadata from disk but never calls `load_snapshot()` or `inject_state_to_pool()`. No request object is created, no Mamba state is restored.

**Citation**: `python/sglang/srt/managers/scheduler.py:1058-1072`

---

## Architecture Context

### Relevant Components

| Component | File | Role |
|-----------|------|------|
| `MambaSnapshotManager` | `python/sglang/srt/snapshot/mamba_snapshot.py` | Disk serialization; `list_conversations()`, `get_latest_snapshot()`, `load_snapshot()`, `inject_state_to_pool()` |
| `TierManager` | `python/sglang/srt/snapshot/tier_manager.py` | 3-tier management: GPU → WARM (host RAM) → COLD (disk); `restore_conversation()`, `save_to_warm_tier()` |
| `MambaHostPool` | `python/sglang/srt/snapshot/mamba_host_pool.py` | Host memory staging area for snapshots |
| `MambaSnapshotManager.list_conversations()` | `mamba_snapshot.py` | Returns list of conversation IDs with snapshots |
| `MambaSnapshotManager.get_latest_snapshot()` | `mamba_snapshot.py` | Returns `(turn_number, MambaSnapshotMetadata)` for latest snapshot |
| `TierManager.restore_conversation()` | `tier_manager.py:235-281` | Restores from WARM tier first, falls back to COLD; auto-promotes to WARM after cold restore |

### TierManager Restore Logic

```python
# tier_manager.py:255-281 — restore_conversation
with self._lock:
    tier = self.conversation_tracker.get_tier(conversation_id)

    # Try WARM tier first (fastest)
    if tier == ConversationTier.WARM or self.host_pool.has_state(conversation_id):
        result = self.restore_from_warm_tier(conversation_id)
        if result:
            return result

    # Fall back to COLD tier
    if tier == ConversationTier.COLD or turn_number is not None:
        result = self.restore_from_cold_tier(conversation_id, turn_number)
        if result:
            conv_states, temporal_states, metadata = result
            # Promote to WARM tier for future fast access
            self.save_to_warm_tier(conversation_id, conv_states, temporal_states, metadata.to_dict())
            return conv_states, temporal_states, metadata.to_dict()

    logger.warning(f"Failed to restore conversation: {conversation_id}")
    return None
```

---

## Implementation Strategy

### Phase A: Pre-load Latest Snapshots to WARM Tier (This PR)

**Scope**: Load the latest snapshot for each conversation into host RAM (WARM tier) on startup.

**Why this matters**: After server restart, if a client reconnects and wants to continue a conversation, the snapshot is already in WARM tier — no disk I/O needed for the first restore. Fast restore from ~5-30ms (warm) vs ~50-200ms (cold disk read).

**Implementation approach**:

```python
def restore_snapshots_on_startup(self):
    """
    Pre-load the latest snapshots for all conversations into WARM tier (host RAM).
    """
    if self.snapshot_manager is None:
        return

    conversations = self.snapshot_manager.list_conversations()
    if not conversations:
        return

    # Get TierManager (self.tier_manager or self.snapshot_manager.tier_manager)
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
            # Use TierManager.restore_conversation to load from disk (COLD)
            # and auto-promote to WARM
            result = tier_manager.restore_conversation(conv_id, turn_number=None)
            if result:
                logger.info(f"  Pre-loaded conv_id={conv_id} turn={turn_number} to WARM tier")
                restored_count += 1
        except Exception as e:
            logger.error(f"  Failed to pre-load conv_id={conv_id}: {e}")

    logger.info(f"Startup restore complete: {restored_count}/{len(conversations)} conversations pre-loaded to WARM tier")
```

### Phase B: Conversation Tracker Registration

The `TierManager.conversation_tracker` needs to know about restored conversations so it can track which tier they're in. This may require registering each restored conversation in `conversation_tracker` after the WARM load.

### Phase C: Full Request Recreation (Future)

This is **not** in scope for this PR. Full request recreation would mean:
1. After loading snapshot to WARM, create a `Req` object in a "dormant" state
2. When a new request comes in with that `conversation_id`, attach it to the dormant `Req`
3. This enables true cross-session continuation without client re-sending all history

Phase C is complex and depends on request lifecycle management infrastructure.

---

## Files to Modify

| File | Change |
|------|--------|
| `python/sglang/srt/managers/scheduler.py` | Implement `restore_snapshots_on_startup()` to pre-load latest snapshots to WARM tier via `TierManager.restore_conversation()` |

---

## Testing

### Unit test approach (no server needed)

```python
def test_restore_snapshots_on_startup_pre_loads_to_warm(self):
    """After restore_snapshots_on_startup, conversations are in WARM tier."""
    # Create snapshots for conv_1 and conv_2
    # Call scheduler.restore_snapshots_on_startup()
    # Assert tier_manager.conversation_tracker.get_tier(conv_1) == WARM
    # Assert tier_manager.conversation_tracker.get_tier(conv_2) == WARM
```

### Integration test approach

```python
def test_startup_restore_fast_followed_by_request(self):
    """Server starts, pre-loads snapshots, first request with conv_id restores from WARM (fast)."""
    # 1. Server running, create conv with snapshots
    # 2. Kill server
    # 3. Restart server
    # 4. Call restore_snapshots_on_startup()
    # 5. Assert conv in WARM tier (no disk read on next request)
    # 6. Send request with conv_id — should fast-restore from WARM
```

---

## Verification

After implementation:

```bash
# 1. Start server with snapshots
python -m sglang.launch_server --model-path $MODEL_PATH --port 30000 \
    --enable-snapshot-persistence --snapshot-dir /tmp/snapshots

# 2. Create conversations with snapshots (via API)

# 3. Kill and restart server
# 4. Check logs for pre-load:
grep "Startup restore complete" /tmp/server.log
# Expected: "Startup restore complete: 2/2 conversations pre-loaded to WARM tier"

# 5. Check tier manager:
# tier_manager.conversation_tracker should show WARM for restored convs
```

---

## Branch Strategy

- Branch from: `main`
- Branch name: `fix/startup-snapshot-restore`
- Files: `python/sglang/srt/managers/scheduler.py`
- Tests: `test/sglang/snapshot/` (new test file or additions)

---

## Dependencies

- `MambaSnapshotManager.list_conversations()` — already implemented
- `MambaSnapshotManager.get_latest_snapshot()` — already implemented
- `TierManager.restore_conversation()` — already implemented (handles WARM/COLD restore with auto-promotion)
- `TierManager.conversation_tracker` — exists, needs `register()` or similar for restored convs

---

## Open Questions

1. **TierManager accessibility**: Is `self.snapshot_manager.tier_manager` accessible from `Scheduler.restore_snapshots_on_startup()`? Or is there a separate `self.tier_manager` reference?
2. **Conversation tracker registration**: Does `restore_conversation()` automatically register the conversation in `conversation_tracker`, or does that need to be done separately?
3. **Error handling**: If pre-loading fails for one conversation (disk corruption, etc.), should we continue with others or abort?
4. **Snapshots created during runtime**: If a snapshot is saved while server is running, it should be registered in the WARM tier immediately (not just on restart). This is a separate runtime concern.

---

## References

- **PR #4** (Gaps 1 & 2): https://github.com/KHAEntertainment/sglang-mamba/pull/4
- **Scheduler**: `python/sglang/srt/managers/scheduler.py:1040-1074` (current stub)
- **TierManager**: `python/sglang/srt/snapshot/tier_manager.py:235-281` (`restore_conversation`)
- **MambaSnapshotManager**: `python/sglang/srt/snapshot/mamba_snapshot.py`
- **Migration Guide**: `docs/stateful_mamba/migration_guide.md`
