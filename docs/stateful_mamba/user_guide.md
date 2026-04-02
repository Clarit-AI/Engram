# Stateful Mamba User Guide

This guide covers everything you need to know about using the snapshot system for Mamba models in SGLang.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Basic Snapshot Operations](#basic-snapshot-operations)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Performance Tuning](#performance-tuning)

## Introduction

The Stateful Mamba snapshot system allows you to save and restore the internal state of Mamba models during inference. This enables powerful patterns like:

- **Multi-turn conversations** without reprocessing context
- **Branching narratives** from a single checkpoint
- **Session persistence** across restarts
- **A/B testing** different continuations

### What is a Snapshot?

A snapshot captures the complete state of a Mamba model at a specific point in the token sequence:

- **SSM States**: Hidden states of the State Space Model layers
- **Token Sequence**: The tokens processed up to this point
- **Metadata**: User-defined key-value pairs

### Backward Compatibility

**Important**: All snapshot features are **opt-in** and have **zero impact** on existing SGLang usage:

- Snapshots are disabled by default
- Existing code works unchanged
- No performance overhead when disabled
- Transformer models are unaffected (this is Mamba-specific)

## Getting Started

### Prerequisites

1. **Mamba Model**: Snapshots only work with Mamba/Mamba2 architectures
2. **SGLang Version**: v0.5.0 or later
3. **GPU Memory**: Additional memory for snapshot storage (typically 5-10% per snapshot)

### Enabling Snapshots

Snapshots can be enabled globally or per-request.

#### Global Configuration

```python
from sglang import Engine

# Enable snapshots for all requests
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,  # Enable snapshot feature
    snapshot_config={
        "max_snapshots": 100,           # Maximum number of concurrent snapshots
        "enable_persistence": True,     # Allow saving to disk
        "storage_path": "./snapshots",  # Where to store snapshots
        "enable_cow": True,             # Enable copy-on-write optimization
    }
)
```

#### Per-Request Configuration

```python
from sglang import Engine
from sglang.lang import function, gen

engine = Engine(model_path="state-spaces/mamba-2.8b")

@function
def my_function(s):
    # Enable snapshots for this request only
    s.enable_snapshots()

    # Use snapshot features
    s += gen("text", max_tokens=100)
    snapshot_id = s.save_snapshot()

    return s
```

### Quick Example

Here's a minimal example demonstrating snapshot usage:

```python
from sglang import Engine
from sglang.lang import function, gen

engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True
)

@function
def chat_with_snapshots(s, user_message):
    # Initial context
    s += "You are a helpful assistant.\n\n"

    # First user message
    s += f"User: {user_message}\n"
    s += "Assistant: " + gen("response", max_tokens=100)

    # Save state after first exchange
    snapshot_id = s.save_snapshot()

    # Continue with second message
    s += "\nUser: Can you explain that differently?\n"
    s += "Assistant: " + gen("explanation", max_tokens=100)

    return s, snapshot_id

# Run the conversation
result, snap_id = chat_with_snapshots.run(
    user_message="What is machine learning?",
    engine=engine
)

print(f"Response: {result['response']}")
print(f"Snapshot saved: {snap_id}")
```

## Basic Snapshot Operations

### Creating Snapshots

#### Auto-generated ID

```python
@function
def my_function(s):
    s += gen("text", max_tokens=50)

    # Save with auto-generated UUID
    snapshot_id = s.save_snapshot()

    return snapshot_id
```

#### Custom ID

```python
@function
def my_function(s):
    s += gen("text", max_tokens=50)

    # Save with custom ID
    snapshot_id = s.save_snapshot("checkpoint_1")

    return snapshot_id
```

#### With Metadata

```python
@function
def my_function(s):
    s += gen("text", max_tokens=50)

    # Save with metadata
    snapshot_id = s.save_snapshot(
        "checkpoint_1",
        metadata={
            "turn": 1,
            "user_id": "alice",
            "timestamp": time.time(),
            "tags": ["important", "conversation"]
        }
    )

    return snapshot_id
```

### Restoring Snapshots

#### Basic Restore

```python
@function
def continue_from_checkpoint(s, snapshot_id):
    # Restore state from snapshot
    s.restore_snapshot(snapshot_id)

    # Continue generation from restored state
    s += "\nUser: Tell me more.\n"
    s += "Assistant: " + gen("continuation", max_tokens=100)

    return s
```

#### Branching from Snapshot

```python
@function
def explore_branch_a(s, snapshot_id):
    s.restore_snapshot(snapshot_id)
    s += "\nPath A: " + gen("branch_a", max_tokens=50)
    return s

@function
def explore_branch_b(s, snapshot_id):
    s.restore_snapshot(snapshot_id)
    s += "\nPath B: " + gen("branch_b", max_tokens=50)
    return s

# Both branches start from same state
result_a = explore_branch_a.run(snapshot_id="checkpoint_1", engine=engine)
result_b = explore_branch_b.run(snapshot_id="checkpoint_1", engine=engine)
```

### Querying Snapshots

#### Get Snapshot Info

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Get snapshot details
snapshot = manager.get_snapshot("checkpoint_1")

print(f"ID: {snapshot.snapshot_id}")
print(f"Tokens: {len(snapshot.token_ids)}")
print(f"Seq Length: {snapshot.seq_len}")
print(f"Metadata: {snapshot.metadata}")
print(f"Created: {snapshot.timestamp}")
```

#### List All Snapshots

```python
manager = SnapshotManager(engine)

# List all active snapshots
snapshots = manager.list_snapshots()

for snap in snapshots:
    print(f"ID: {snap.snapshot_id}, Tokens: {snap.seq_len}")
```

#### Filter by Metadata

```python
# Find snapshots by metadata
important_snapshots = manager.list_snapshots(
    filter_fn=lambda s: "important" in s.metadata.get("tags", [])
)
```

### Deleting Snapshots

#### Delete Single Snapshot

```python
@function
def cleanup_example(s):
    # Create snapshot
    snap_id = s.save_snapshot()

    # Use it...
    # ...

    # Clean up when done
    s.delete_snapshot(snap_id)

    return s
```

#### Delete Multiple Snapshots

```python
manager = SnapshotManager(engine)

# Delete old snapshots
cutoff_time = time.time() - 3600  # 1 hour ago
manager.delete_snapshots_before(cutoff_time)
```

#### Delete All Snapshots

```python
manager = SnapshotManager(engine)

# Delete all snapshots
manager.clear_all_snapshots()
```

## Advanced Features

### Disk Persistence

#### Saving to Disk

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Save snapshot to disk
snapshot_id = "checkpoint_1"
manager.persist_snapshot(
    snapshot_id,
    path="./saved_states/checkpoint_1.snapshot"
)
```

#### Loading from Disk

```python
manager = SnapshotManager(engine)

# Load snapshot from disk
snapshot_id = manager.load_snapshot(
    path="./saved_states/checkpoint_1.snapshot"
)

# Now use the loaded snapshot
@function
def continue_from_disk(s):
    s.restore_snapshot(snapshot_id)
    s += gen("continuation", max_tokens=100)
    return s
```

#### Auto-Persistence

```python
# Enable auto-persistence in config
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_persistence": True,
        "auto_persist": True,              # Auto-save to disk
        "storage_path": "./snapshots",
        "persist_delay_seconds": 60,       # Delay before persisting
    }
)

@function
def auto_persist_example(s):
    # This snapshot will automatically be saved to disk after 60 seconds
    snapshot_id = s.save_snapshot("auto_saved")
    return s
```

### Copy-on-Write (COW)

COW optimization allows multiple requests to share snapshot memory until modification.

#### Enabling COW

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_cow": True,  # Enable copy-on-write
    }
)
```

#### How COW Works

```python
@function
def branch_1(s, snapshot_id):
    # Restore with COW - shares memory with snapshot
    s.restore_snapshot(snapshot_id)
    # First token generated triggers copy
    s += gen("branch1", max_tokens=10)
    return s

@function
def branch_2(s, snapshot_id):
    # Also shares memory initially
    s.restore_snapshot(snapshot_id)
    # Triggers its own copy on first modification
    s += gen("branch2", max_tokens=10)
    return s

# Both share snapshot memory until first generation
# Then each gets its own copy
```

### Snapshot Chaining

Create snapshots that reference other snapshots:

```python
@function
def snapshot_chain(s):
    # Base snapshot
    s += "Context: " + gen("context", max_tokens=50)
    base_id = s.save_snapshot("base", metadata={"level": 0})

    # Child snapshot 1
    s += "\nPath 1: " + gen("path1", max_tokens=30)
    child1_id = s.save_snapshot(
        "child_1",
        metadata={"level": 1, "parent": base_id}
    )

    # Restore base and create child 2
    s.restore_snapshot(base_id)
    s += "\nPath 2: " + gen("path2", max_tokens=30)
    child2_id = s.save_snapshot(
        "child_2",
        metadata={"level": 1, "parent": base_id}
    )

    return base_id, child1_id, child2_id
```

### Snapshot Comparison

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Compare two snapshots
snap1 = manager.get_snapshot("checkpoint_1")
snap2 = manager.get_snapshot("checkpoint_2")

# Compare token sequences
common_prefix_len = 0
for t1, t2 in zip(snap1.token_ids, snap2.token_ids):
    if t1 != t2:
        break
    common_prefix_len += 1

print(f"Common prefix: {common_prefix_len} tokens")
print(f"Divergence point: {snap1.token_ids[common_prefix_len:]}")
```

## Best Practices

### Memory Management

#### 1. Clean Up Unused Snapshots

```python
@function
def good_memory_management(s):
    # Create snapshot
    snap_id = s.save_snapshot()

    # Use snapshot
    s.restore_snapshot(snap_id)
    s += gen("text", max_tokens=100)

    # Delete when no longer needed
    s.delete_snapshot(snap_id)  # ✓ Good: Free memory

    return s
```

**Bad**:
```python
@function
def bad_memory_management(s):
    # Create many snapshots without cleanup
    for i in range(100):
        s += gen(f"text_{i}", max_tokens=10)
        s.save_snapshot(f"snap_{i}")  # ✗ Bad: Memory leak

    return s
```

#### 2. Use Metadata for Tracking

```python
@function
def with_metadata(s):
    snap_id = s.save_snapshot(
        metadata={
            "created_at": time.time(),
            "purpose": "checkpoint",
            "important": True,
            "auto_delete_after": 3600,  # 1 hour
        }
    )
    return snap_id
```

#### 3. Monitor Memory Usage

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Check memory usage
total_memory = manager.get_total_snapshot_memory()
num_snapshots = len(manager.list_snapshots())

print(f"Snapshots: {num_snapshots}, Memory: {total_memory / 1e9:.2f} GB")

# Alert if memory is high
if total_memory > 10 * 1e9:  # 10 GB
    print("WARNING: High snapshot memory usage!")
    # Cleanup old snapshots
    manager.cleanup_old_snapshots(keep_recent=10)
```

### Performance Optimization

#### 1. Batch Snapshot Operations

```python
# Good: Create snapshots at strategic points
@function
def strategic_snapshots(s):
    s += gen("intro", max_tokens=100)
    snap1 = s.save_snapshot()  # After intro

    s += gen("main", max_tokens=500)
    snap2 = s.save_snapshot()  # After main content

    return snap1, snap2
```

```python
# Bad: Create snapshots too frequently
@function
def excessive_snapshots(s):
    for i in range(100):
        s += gen(f"token_{i}", max_tokens=1)
        s.save_snapshot()  # ✗ Too many snapshots!
    return s
```

#### 2. Use COW for Memory Efficiency

```python
# Enable COW for branching scenarios
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={"enable_cow": True}  # ✓ Good for branching
)
```

#### 3. Persist Only Important Snapshots

```python
@function
def selective_persistence(s):
    # Temporary snapshot (memory only)
    temp_snap = s.save_snapshot("temp")

    # Important snapshot (persist to disk)
    important_snap = s.save_snapshot("important")
    s.persist_snapshot(important_snap, "./important.snapshot")

    # Clean up temp
    s.delete_snapshot(temp_snap)

    return important_snap
```

### Error Handling

#### 1. Handle Missing Snapshots

```python
@function
def safe_restore(s, snapshot_id):
    try:
        s.restore_snapshot(snapshot_id)
    except SnapshotNotFoundError:
        print(f"Snapshot {snapshot_id} not found, using fallback")
        # Fallback: Generate from scratch
        s += gen("fallback", max_tokens=100)

    return s
```

#### 2. Handle Memory Errors

```python
@function
def safe_save(s):
    try:
        snap_id = s.save_snapshot()
        return snap_id
    except OutOfMemoryError:
        print("Out of memory, cleaning up old snapshots")
        s.cleanup_old_snapshots(keep_recent=5)
        # Retry
        snap_id = s.save_snapshot()
        return snap_id
```

#### 3. Validate Snapshot State

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

def validate_snapshot(snapshot_id):
    """Check if snapshot is valid before using."""
    try:
        snapshot = manager.get_snapshot(snapshot_id)
        if snapshot is None:
            return False

        # Check if snapshot is not too old
        age = time.time() - snapshot.timestamp
        if age > 86400:  # 24 hours
            print(f"Snapshot {snapshot_id} is too old")
            return False

        return True
    except Exception as e:
        print(f"Snapshot validation failed: {e}")
        return False
```

## Common Patterns

### Pattern 1: Multi-Turn Conversation

```python
@function
def multi_turn_chat(s, user_messages):
    # Initial context
    s += "You are a helpful assistant.\n\n"

    snapshots = []

    for i, user_msg in enumerate(user_messages):
        # User message
        s += f"User: {user_msg}\n"
        s += "Assistant: " + gen(f"response_{i}", max_tokens=100)

        # Save snapshot after each exchange
        snap_id = s.save_snapshot(
            f"turn_{i}",
            metadata={"turn": i, "user_message": user_msg}
        )
        snapshots.append(snap_id)

    return s, snapshots

# Usage
messages = [
    "What is Python?",
    "How do I use loops?",
    "Can you show an example?"
]

result, snapshots = multi_turn_chat.run(
    user_messages=messages,
    engine=engine
)

# Can restore to any turn later
```

### Pattern 2: Branching Narratives

```python
@function
def branching_story(s, base_snapshot_id, choice):
    # Restore base state
    s.restore_snapshot(base_snapshot_id)

    # Branch based on choice
    if choice == "A":
        s += "\nYou chose path A.\n"
        s += gen("path_a_story", max_tokens=200)
    elif choice == "B":
        s += "\nYou chose path B.\n"
        s += gen("path_b_story", max_tokens=200)
    else:
        s += "\nYou chose path C.\n"
        s += gen("path_c_story", max_tokens=200)

    return s

# Base story
@function
def base_story(s):
    s += "You enter a dark forest. You see three paths.\n"
    s += gen("setup", max_tokens=100)
    return s.save_snapshot("forest_entrance")

# Generate base
base_snap = base_story.run(engine=engine)

# Explore different branches
story_a = branching_story.run(base_snapshot_id=base_snap, choice="A", engine=engine)
story_b = branching_story.run(base_snapshot_id=base_snap, choice="B", engine=engine)
story_c = branching_story.run(base_snapshot_id=base_snap, choice="C", engine=engine)
```

### Pattern 3: Checkpoint-Based Generation

```python
@function
def long_generation_with_checkpoints(s, num_sections):
    checkpoints = []

    for i in range(num_sections):
        s += f"\n## Section {i+1}\n"
        s += gen(f"section_{i}", max_tokens=500)

        # Checkpoint after each section
        checkpoint_id = s.save_snapshot(
            f"section_{i}_checkpoint",
            metadata={"section": i}
        )
        checkpoints.append(checkpoint_id)

    return s, checkpoints

# Generate document
result, checkpoints = long_generation_with_checkpoints.run(
    num_sections=5,
    engine=engine
)

# If generation fails at section 3, resume from section 2 checkpoint
@function
def resume_generation(s, checkpoint_id, start_section, num_sections):
    s.restore_snapshot(checkpoint_id)

    for i in range(start_section, num_sections):
        s += f"\n## Section {i+1}\n"
        s += gen(f"section_{i}", max_tokens=500)

    return s
```

### Pattern 4: A/B Testing

```python
@function
def ab_test_generation(s, base_snapshot_id, temperature_a, temperature_b):
    # Variant A
    s.restore_snapshot(base_snapshot_id)
    s += gen("variant_a", max_tokens=100, temperature=temperature_a)
    result_a = s.text()

    # Variant B
    s.restore_snapshot(base_snapshot_id)
    s += gen("variant_b", max_tokens=100, temperature=temperature_b)
    result_b = s.text()

    return result_a, result_b

# Usage
@function
def setup_ab_test(s):
    s += "Write a product description for: "
    return s.save_snapshot("base")

base_snap = setup_ab_test.run(engine=engine)

# Test different temperatures
variant_a, variant_b = ab_test_generation.run(
    base_snapshot_id=base_snap,
    temperature_a=0.7,
    temperature_b=1.2,
    engine=engine
)

print(f"Variant A (T=0.7): {variant_a}")
print(f"Variant B (T=1.2): {variant_b}")
```

### Pattern 5: Session Persistence

```python
import os
from sglang.snapshot import SnapshotManager

class ChatSession:
    def __init__(self, session_id, engine):
        self.session_id = session_id
        self.engine = engine
        self.manager = SnapshotManager(engine)
        self.current_snapshot = None

    def save_session(self):
        """Save current session to disk."""
        if self.current_snapshot:
            path = f"./sessions/{self.session_id}.snapshot"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.manager.persist_snapshot(self.current_snapshot, path)

    def load_session(self):
        """Load session from disk."""
        path = f"./sessions/{self.session_id}.snapshot"
        if os.path.exists(path):
            self.current_snapshot = self.manager.load_snapshot(path)
            return True
        return False

    @function
    def send_message(s, self, user_message):
        # Restore previous session if exists
        if self.current_snapshot:
            s.restore_snapshot(self.current_snapshot)

        # Process new message
        s += f"\nUser: {user_message}\n"
        s += "Assistant: " + gen("response", max_tokens=100)

        # Save updated state
        self.current_snapshot = s.save_snapshot()

        return s

# Usage
session = ChatSession("user_123", engine)

# Load previous session if exists
session.load_session()

# Send messages
session.send_message("Hello!")
session.save_session()

# Later...
session.send_message("How are you?")
session.save_session()
```

## Performance Tuning

### Memory Configuration

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "max_snapshots": 50,              # Limit concurrent snapshots
        "max_snapshot_memory_gb": 10.0,   # Max memory for snapshots
        "enable_cow": True,                # Reduce memory with COW
        "eviction_policy": "lru",          # Evict least recently used
    }
)
```

### Disk I/O Configuration

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_persistence": True,
        "storage_path": "./snapshots",
        "serialization_format": "safetensors",  # Fast, safe format
        "compression_enabled": True,            # Compress on disk
        "async_io": True,                       # Non-blocking I/O
        "io_threads": 4,                        # Parallel I/O
    }
)
```

### Profiling Snapshot Operations

```python
import time
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Profile save operation
@function
def profile_save(s):
    s += gen("text", max_tokens=100)

    start = time.time()
    snap_id = s.save_snapshot()
    save_time = time.time() - start

    print(f"Save time: {save_time*1000:.2f}ms")
    return snap_id

# Profile restore operation
@function
def profile_restore(s, snapshot_id):
    start = time.time()
    s.restore_snapshot(snapshot_id)
    restore_time = time.time() - start

    print(f"Restore time: {restore_time*1000:.2f}ms")
    return s
```

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed API documentation
- Read the [Architecture](architecture.md) document to understand the internals
- Check out [Examples](examples.md) for more real-world use cases
