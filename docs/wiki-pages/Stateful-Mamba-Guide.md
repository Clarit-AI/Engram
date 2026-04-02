# Stateful Mamba Guide

> This page consolidates documentation originally in `docs/stateful_mamba/` and related files in the repo.

## Overview

The Stateful Mamba snapshot system extends SGLang's Mamba support with **opt-in** state persistence capabilities, enabling advanced use cases like multi-turn conversations, checkpoint-based inference, and state reuse across requests.

**Implementation Status:** Snapshot save/restore/list/get/delete are all available. Startup restore (Gap 3) implemented in PR #6.

## Key Features

### Phase 1 (Available Now)
- **Snapshot Saving**: Save Mamba hidden states (SSM states) at any point during inference
- **Snapshot Inspection**: List and inspect saved snapshots with metadata
- **State Persistence**: Store snapshots to disk in safetensors format
- **Zero Impact**: Existing Mamba and transformer inference workflows are unaffected

### Phase 2 (Available Now)
- **State Restoration**: Restore from saved snapshots to resume conversations (in-place and create_new_request)
- **State Reuse**: Share snapshots across multiple inference branches
- **Multi-turn Conversations**: Efficiently handle long conversations without reprocessing context
- **Startup Restore**: Automatically pre-load snapshots into WARM tier on server restart

## Quick Start

```python
from sglang import function, gen, Runtime

runtime = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,
    snapshot_dir="./my_snapshots"
)

@function
def conversation_with_snapshots(s):
    s += "User: What is machine learning?\n"
    s += "Assistant: " + gen("response1", max_tokens=100)

    # Save snapshot after first response
    snapshot_id = s.save_snapshot()

    # Continue conversation
    s += "\nUser: Can you give me an example?\n"
    s += "Assistant: " + gen("response2", max_tokens=100)

    return s

result = conversation_with_snapshots.run(runtime=runtime)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Frontend (Language API)                 │
│  Phase 1: save_snapshot(), list_snapshots()         │
│  Phase 2: restore_snapshot(), get_snapshot_info()   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Snapshot Persistence                      │
│  - State Serialization (safetensors)                │
│  - Metadata Storage (JSON)                          │
│  - Disk I/O                                         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Mamba Radix Cache                           │
│  - Existing tree-based cache                        │
│  - Snapshot references (Phase 1)                    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Memory Pool (SSM States)                    │
│  - GPU memory allocation                            │
│  - State persistence to disk                        │
└─────────────────────────────────────────────────────┘
```

## Startup Restore (Gap 3)

After server restart, `restore_snapshots_on_startup()` pre-loads the latest snapshot for each conversation into the **WARM tier** (host RAM) so subsequent requests can fast-restore without disk I/O.

**Implementation** (`scheduler.py:1040-1074`):
```python
def restore_snapshots_on_startup(self):
    """Pre-load the latest snapshots for all conversations into WARM tier."""
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

**Key components involved:**
- `MambaSnapshotManager.list_conversations()` — Returns list of conversation IDs with snapshots
- `MambaSnapshotManager.get_latest_snapshot()` — Returns `(turn_number, MambaSnapshotMetadata)` for latest snapshot
- `TierManager.restore_conversation()` — Restores from WARM tier first, falls back to COLD; auto-promotes to WARM

**Tiers:**
| Tier | Storage | Latency | Use Case |
|------|---------|---------|----------|
| Active | VRAM | <1ms | Current conversations |
| Warm | RAM | ~10ms | Recent conversations |
| Cold | Disk | ~100ms | Long-term storage |

## When NOT to Use Snapshots

1. **Single-shot Inference**: Standard completion requests don't benefit from snapshots
2. **Transformer Models**: This feature is Mamba-specific (transformer KV cache already optimized)
3. **Memory-Constrained Environments**: Snapshots require additional memory for state storage

## System Requirements

- **Model Type**: Mamba/Mamba2 architectures only
- **SGLang Version**: v0.5.0+ (with snapshot support)
- **Python**: 3.8+
- **GPU Memory**: Additional memory for snapshot storage (typically 5-10% of model size per snapshot)

## Memory Usage

Snapshots store SSM states in GPU memory. Each snapshot consumes:
- **State Size**: Depends on model architecture (typically 10-100 MB per snapshot for mid-sized models)
- **Metadata**: Minimal overhead (~1 KB per snapshot)

**Compute Overhead:**
- **Save**: O(1) - Creates a reference to existing state, minimal copying
- **Restore**: O(1) - Updates pointers, no recomputation
- **Disk I/O**: Asynchronous when enabled, doesn't block inference

## Server Configuration

```bash
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --enable-snapshot-persistence \
  --snapshot-dir $SNAPSHOT_DIR \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port $SERVER_PORT
```

**Correct server flags:** `--enable-snapshot-persistence` (NOT `--enable-mamba-snapshots`)

## Gap Fixes (PR #4)

### Gap 1: `fill_ids` sync after restore
`MambaSnapshotMetadata` now stores `fill_ids`, captured at save time and restored to `req.fill_ids` + `req.origin_input_ids` after injection. (`scheduler.py:1439-1443`)

### Gap 2: `create_new_request=True`
`POST /restore_snapshot` with `create_new_request: true` creates a new request backed by restored state, returning a fresh `rid`. (`scheduler.py:1311-1385`)

## Backward Compatibility

**Important**: The snapshot system is completely opt-in and maintains 100% backward compatibility:
- **Default Behavior**: Snapshots are disabled by default
- **Transformer Models**: Unaffected - this feature is Mamba-specific
- **Existing Mamba Inference**: Works exactly as before when snapshots are disabled
- **Performance**: Zero overhead when snapshots are not used
- **API**: All existing SGLang APIs remain unchanged

## Related Documentation

- [Agent Framework](./Agent-Framework.md) - Tool-calling system with 3-tier memory management
- [Upstream Sync](./Upstream-Sync-Q1-2026.md) - Q1 2026 upstream synchronization history
- [GitHub Repo](https://github.com/clarit-ai/engram)
- [DeepWiki](https://deepwiki.com/clarit-ai/engram)
