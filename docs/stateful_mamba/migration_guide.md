# Migration Guide: Enabling Stateful Mamba Snapshots

> **⚠️ Implementation Status:** Phase 1 (Snapshot Saving) is complete. Phase 2 (State Restoration) is in development.
>
> **Available Now:** `save_snapshot()`, `list_snapshots()`, `SnapshotManager` class (for restore/get_info)
> **Coming Soon:** Direct state object methods `s.restore_snapshot()`, `s.get_snapshot_info()`

This guide helps you enable and integrate snapshot features into existing SGLang applications.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Phase 1 Migration](#phase-1-migration)
- [Phase 1 Usage Examples](#phase-1-usage-examples)
- [Phase 2 Preview](#phase-2-preview)
- [Breaking Changes](#breaking-changes)
- [Performance Impact](#performance-impact)

## Overview

The snapshot system is **fully backward compatible**. Existing code continues to work without modification when snapshots are disabled (the default). This guide shows you how to **opt-in** to Phase 1 snapshot features.

### Key Points

- **Zero Breaking Changes**: All existing code works unchanged
- **Opt-in Feature**: Must be explicitly enabled
- **Phase 1 Only**: Currently supports snapshot saving and inspection only
- **No Performance Penalty**: Zero overhead when disabled
- **Transformer Models Unaffected**: This is Mamba-specific

## Prerequisites

### Version Requirements

- **SGLang**: Latest version with Phase 1 snapshot support
- **Python**: 3.8+
- **Model**: Mamba or Mamba2 architecture
- **GPU**: CUDA-capable GPU with sufficient memory

### Check Your Version

```python
import sglang
print(f"SGLang version: {sglang.__version__}")
```

### Memory Requirements

Snapshots consume additional disk space for persistence:
- Typically 50-100 MB per snapshot for Mamba-2.8B
- Varies by model size and sequence length
- Snapshots are saved to disk, not held in memory long-term

## Phase 1 Migration

### Step 1: Update SGLang

```bash
# Update to latest version
pip install --upgrade sglang

# Or install from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

### Step 2: Enable Snapshots

#### Before (without snapshots)

```python
from sglang import Runtime

runtime = Runtime(
    model_path="state-spaces/mamba-2.8b"
)
```

#### After (with snapshots enabled)

```python
from sglang import Runtime
import os

# Create snapshot directory
os.makedirs("./my_snapshots", exist_ok=True)

runtime = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,  # Enable snapshots
    snapshot_dir="./my_snapshots"
)
```

### Step 3: Update Your Code to Use Snapshots

#### Before (standard inference)

```python
from sglang import function, gen

@function
def my_function(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100)
    return s

result = my_function.run(prompt="Hello", runtime=runtime)
```

#### After (with snapshot support - Phase 1)

```python
from sglang import function, gen

@function
def my_function(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100)

    # Save snapshot (new feature)
    snapshot_id = s.save_snapshot()

    # List all snapshots
    snapshots = s.list_snapshots()
    print(f"Total snapshots: {len(snapshots)}")

    # Get snapshot info (Phase 2 - use SnapshotManager for now)
    # NOTE: s.get_snapshot_info() is not yet available as a direct method
    # Use SnapshotManager instead:
    # sm = sgl.SnapshotManager(runtime.endpoint)
    # info = sm.get_info(conversation_id=s.stream_executor.sid, turn_number=0)
    # print(f"Snapshot metadata: {info}")

    return s

result = my_function.run(prompt="Hello", runtime=runtime)
```

### Step 4: Test the Migration

```python
# Test basic snapshot operations
@function
def test_snapshots(s):
    # Basic generation
    s += "Test: "
    s += gen("test1", max_tokens=10)

    # Save snapshot
    try:
        snap_id = s.save_snapshot()
        print(f"✓ Snapshot saved: {snap_id}")
    except Exception as e:
        print(f"✗ Snapshot save failed: {e}")
        return s

    # List snapshots
    try:
        snapshots = s.list_snapshots()
        print(f"✓ Found {len(snapshots)} snapshots")
    except Exception as e:
        print(f"✗ List snapshots failed: {e}")

    # Get snapshot info
    try:
        info = s.get_snapshot_info(
            conversation_id=s.stream_executor.sid,
            turn_number=0
        )
        print(f"✓ Snapshot info retrieved: {info}")
    except Exception as e:
        print(f"✗ Get snapshot info failed: {e}")

    return s

result = test_snapshots.run(runtime=runtime)
```

## Compatibility Matrix

| Feature | Snapshots Disabled | Phase 1 (Current) | Phase 2 (Coming) |
|---------|-------------------|-------------------|------------------|
| Standard Mamba inference | ✓ | ✓ | ✓ |
| Transformer models | ✓ | ✓ | ✓ |
| Radix cache | ✓ | ✓ | ✓ |
| Multi-turn conversations | ✓ | ✓ | ✓ (better) |
| `save_snapshot()` | ✗ | ✓ | ✓ |
| `list_snapshots()` | ✗ | ✓ | ✓ |
| `get_snapshot_info()` | ✗ | ✓ | ✓ |
| `restore_snapshot()` | ✗ | ✗ | ✓ |
| `SnapshotManager` | ✗ | ✗ | ✓ |
| Disk persistence | ✗ | ✓ | ✓ |

## Phase 1 Usage Examples

### Example 1: Basic Snapshot Saving

```python
from sglang import function, gen, Runtime

@function
def stateful_chat(s):
    s += "User: " + s["user_input"] + "\n"
    s += "Assistant: " + gen("response", max_tokens=100)

    # Save snapshot with auto-generated ID
    snap_id = s.save_snapshot()
    print(f"Saved snapshot: {snap_id}")

    # Or with custom conversation tracking
    snap_id = s.save_snapshot(
        conversation_id="user_123_session",
        turn_number=1
    )

    return s

runtime = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,
    snapshot_dir="./snapshots"
)

result = stateful_chat.run(user_input="Hello!", runtime=runtime)
```

### Example 2: Listing and Inspecting Snapshots

```python
from sglang import function, gen

@function
def inspect_snapshots(s):
    # Generate some content with multiple turns
    for i in range(3):
        s += f"Turn {i}: "
        s += gen(f"response_{i}", max_tokens=50)
        s.save_snapshot()

    # List all snapshots for current conversation
    snapshots = s.list_snapshots()
    print(f"Total snapshots: {len(snapshots)}")

    for snap in snapshots:
        print(f"Turn {snap['turn_number']}: {snap['token_count']} tokens")
        print(f"Timestamp: {snap['timestamp']}")
        print(f"Conversation ID: {snap['conversation_id']}")

    # Get detailed info about a specific snapshot
    if snapshots:
        info = s.get_snapshot_info(
            conversation_id=snapshots[0]['conversation_id'],
            turn_number=snapshots[0]['turn_number']
        )
        print(f"Detailed info: {info}")

    return s

result = inspect_snapshots.run(runtime=runtime)
```

### Example 3: Tracking Conversation Progress

```python
from sglang import function, gen

@function
def multi_turn_conversation(s):
    turns = [
        "What is machine learning?",
        "Can you give an example?",
        "How does it differ from traditional programming?"
    ]

    for i, user_msg in enumerate(turns):
        s += f"User: {user_msg}\n"
        s += "Assistant: " + gen(f"response_{i}", max_tokens=100)

        # Save snapshot after each turn
        snap_id = s.save_snapshot(
            conversation_id="ml_conversation",
            turn_number=i
        )
        print(f"Saved turn {i} as snapshot {snap_id}")

    # After conversation, inspect all snapshots
    all_snapshots = s.list_snapshots()
    print(f"\nConversation had {len(all_snapshots)} turns")

    # Examine growth over time
    for snap in all_snapshots:
        info = s.get_snapshot_info(
            conversation_id=snap['conversation_id'],
            turn_number=snap['turn_number']
        )
        print(f"Turn {snap['turn_number']}: {info['token_count']} tokens")

    return s

result = multi_turn_conversation.run(runtime=runtime)
```

## Phase 2 Preview

Phase 2 will add state restoration and advanced snapshot management. Here's a preview of what's coming:

```python
# Phase 2: Coming Soon
from sglang import function, gen
from sglang.snapshot import SnapshotManager

@function
def chat_with_restoration(s):
    # First turn
    s += "User: Hello\n"
    s += "Assistant: " + gen("response1", max_tokens=100)
    snap_id = s.save_snapshot()

    # Second turn
    s += "\nUser: Tell me more\n"
    s += "Assistant: " + gen("response2", max_tokens=100)

    # Restore to first turn (Phase 2 feature - not yet available as direct method)
    # NOTE: s.restore_snapshot() is not yet available
    # Use SnapshotManager instead (see below)

    # Continue from restored state (without actual restore in Phase 1)
    s += "\nUser: Actually, tell me something else\n"
    s += "Assistant: " + gen("response3", max_tokens=100)

    return s

# SnapshotManager class (Available Now!)
from sglang import SnapshotManager
manager = SnapshotManager(runtime.endpoint)
# List snapshots for a conversation
snapshots = manager.list_conversation(conversation_id="conv_123")
# Get snapshot metadata
info = manager.get_info(conversation_id="conv_123", turn_number=5)
# Restore a snapshot
manager.restore(rid="req_123", conversation_id="conv_123", turn_number=5)
# Delete a snapshot
manager.delete(conversation_id="conv_123", turn_number=5)
```

## Breaking Changes

**None.** The snapshot system is fully backward compatible.

### Disk Usage

Snapshots are persisted to disk:
- Each snapshot: ~50-100 MB for Mamba-2.8B
- Ensure adequate disk space in snapshot directory
- Consider cleanup strategies for long-running applications

## Performance Impact

### When Snapshots Are Disabled (default)

- **Overhead**: Zero
- **Disk**: No change
- **Performance**: Identical to baseline

### When Snapshots Are Enabled (Phase 1)

- **Save Operation**: Fast I/O operation to write to disk
- **Overhead**: Minimal during inference
- **Disk**: ~50-100 MB per snapshot (varies by model)

### Benchmark Comparison

```python
import time
from sglang import function, gen, Runtime

# Without snapshots
runtime_baseline = Runtime(model_path="state-spaces/mamba-2.8b")

@function
def without_snapshots(s):
    for i in range(10):
        s += gen(f"text_{i}", max_tokens=50)
    return s

start = time.time()
result = without_snapshots.run(runtime=runtime_baseline)
time_baseline = time.time() - start
print(f"Without snapshots: {time_baseline:.2f}s")

# With snapshots enabled
runtime_snapshots = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,
    snapshot_dir="./snapshots"
)

start = time.time()
result = without_snapshots.run(runtime=runtime_snapshots)
time_enabled = time.time() - start
print(f"With snapshots (not used): {time_enabled:.2f}s")
print(f"Overhead: {(time_enabled - time_baseline) / time_baseline * 100:.2f}%")

# Actively saving snapshots
@function
def with_snapshots(s):
    for i in range(10):
        s += gen(f"text_{i}", max_tokens=50)
        s.save_snapshot()
    return s

start = time.time()
result = with_snapshots.run(runtime=runtime_snapshots)
time_with_saves = time.time() - start
print(f"With snapshot saves: {time_with_saves:.2f}s")
print(f"Save overhead: {(time_with_saves - time_baseline) / time_baseline * 100:.2f}%")
```

## Migration Checklist

Use this checklist when migrating to snapshot-enabled inference:

- [ ] Upgrade SGLang to latest version with Phase 1 support
- [ ] Check model compatibility (Mamba/Mamba2 only)
- [ ] Create snapshot directory
- [ ] Ensure adequate disk space
- [ ] Enable snapshots in runtime config
- [ ] Update code to use snapshot operations (optional)
- [ ] Test snapshot save/list/info operations
- [ ] Monitor disk usage
- [ ] Plan cleanup strategy for old snapshots
- [ ] Update documentation for your application

## Getting Help

If you encounter issues during migration:

- **Documentation**: [Troubleshooting Guide](troubleshooting.md)
- **GitHub Issues**: [Report bugs](https://github.com/sgl-project/sglang/issues)
- **Discussions**: [Ask questions](https://github.com/sgl-project/sglang/discussions)
- **Slack**: [Join community](https://slack.sglang.io/)

## Next Steps

After successful migration:

- Read the [User Guide](user_guide.md) for best practices
- Explore [Examples](examples.md) for real-world patterns
- Review [API Reference](api_reference.md) for detailed documentation
- Learn about [Architecture](architecture.md) for advanced usage
