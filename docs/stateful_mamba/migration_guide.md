# Migration Guide: Enabling Stateful Mamba Snapshots

This guide helps you enable and integrate snapshot features into existing SGLang applications.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Migration Steps](#migration-steps)
- [Compatibility Matrix](#compatibility-matrix)
- [Common Migration Scenarios](#common-migration-scenarios)
- [Breaking Changes](#breaking-changes)
- [Performance Impact](#performance-impact)
- [Rollback Procedure](#rollback-procedure)

## Overview

The snapshot system is **fully backward compatible**. Existing code continues to work without modification when snapshots are disabled (the default). This guide shows you how to **opt-in** to snapshot features.

### Key Points

- **Zero Breaking Changes**: All existing code works unchanged
- **Opt-in Feature**: Must be explicitly enabled
- **Gradual Migration**: Can enable snapshots incrementally
- **No Performance Penalty**: Zero overhead when disabled
- **Transformer Models Unaffected**: This is Mamba-specific

## Prerequisites

### Version Requirements

- **SGLang**: v0.5.0 or later
- **Python**: 3.8+
- **Model**: Mamba or Mamba2 architecture
- **GPU**: CUDA-capable GPU with sufficient memory

### Check Your Version

```python
import sglang
print(f"SGLang version: {sglang.__version__}")

# Check if snapshot modules are available
try:
    from sglang.srt.snapshot import MambaSnapshotManager
    print("Snapshots available: True")
except ImportError:
    print("Snapshots available: False")
```

### Memory Requirements

Estimate additional memory needed for snapshots:

```python
from sglang.snapshot import estimate_snapshot_memory

# For a typical Mamba model
memory_per_snapshot = estimate_snapshot_memory(
    model_name="state-spaces/mamba-2.8b"
)

print(f"Memory per snapshot: {memory_per_snapshot / 1e6:.2f} MB")

# If you plan to have 10 concurrent snapshots
total_memory = memory_per_snapshot * 10
print(f"Total snapshot memory (10 snapshots): {total_memory / 1e9:.2f} GB")
```

## Migration Steps

### Step 1: Update SGLang

```bash
# Update to latest version
pip install --upgrade sglang

# Or install from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

### Step 2: Enable Snapshots Globally

#### Before (without snapshots)

```python
from sglang import Engine

engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    mem_fraction_static=0.9
)
```

#### After (with snapshots enabled)

```python
from sglang import Engine

engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    mem_fraction_static=0.85,  # Leave more memory for snapshots
    enable_mamba_snapshots=True,  # Enable snapshots
    snapshot_config={
        "max_snapshots": 50,
        "max_snapshot_memory_gb": 5.0,
        "enable_cow": True,
    }
)
```

### Step 3: Update Your Code to Use Snapshots

#### Before (standard inference)

```python
from sglang.lang import function, gen

@function
def my_function(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100)
    return s

result = my_function.run(prompt="Hello", engine=engine)
```

#### After (with snapshot support)

```python
from sglang.lang import function, gen

@function
def my_function(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100)

    # Save snapshot (new feature)
    snapshot_id = s.save_snapshot()

    return s, snapshot_id

result, snap_id = my_function.run(prompt="Hello", engine=engine)

# Can now restore from this snapshot
@function
def continue_from_snapshot(s, snapshot_id):
    s.restore_snapshot(snapshot_id)
    s += gen("continuation", max_tokens=100)
    return s
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

    # Restore snapshot
    try:
        s.restore_snapshot(snap_id)
        print(f"✓ Snapshot restored: {snap_id}")
    except Exception as e:
        print(f"✗ Snapshot restore failed: {e}")
        return s

    # Continue generation
    s += gen("test2", max_tokens=10)

    # Cleanup
    s.delete_snapshot(snap_id)
    print("✓ Snapshot deleted")

    return s

result = test_snapshots.run(engine=engine)
```

## Compatibility Matrix

| Feature | Before v0.5.0 | v0.5.0+ (snapshots disabled) | v0.5.0+ (snapshots enabled) |
|---------|---------------|------------------------------|----------------------------|
| Standard Mamba inference | ✓ | ✓ | ✓ |
| Transformer models | ✓ | ✓ | ✓ |
| Radix cache | ✓ | ✓ | ✓ |
| Multi-turn conversations | ✓ | ✓ | ✓ (better with snapshots) |
| `save_snapshot()` | ✗ | ✗ | ✓ |
| `restore_snapshot()` | ✗ | ✗ | ✓ |
| Disk persistence | ✗ | ✗ | ✓ |
| COW optimization | ✗ | ✗ | ✓ |

## Common Migration Scenarios

### Scenario 1: Chatbot Application

#### Before

```python
# chatbot_old.py
from sglang import Engine
from sglang.lang import function, gen

engine = Engine(model_path="state-spaces/mamba-2.8b")

@function
def chat(s, history, new_message):
    # Reprocess entire history each time (inefficient)
    for msg in history:
        s += f"{msg['role']}: {msg['content']}\n"

    s += f"user: {new_message}\n"
    s += "assistant: " + gen("response", max_tokens=100)

    return s

# Must reprocess full history on each message
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

result = chat.run(history=history, new_message="How are you?", engine=engine)
```

#### After (with snapshots)

```python
# chatbot_new.py
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True
)

class ChatSession:
    def __init__(self, engine):
        self.engine = engine
        self.current_snapshot = None

    def send_message(self, message):
        @function
        def chat_turn(s, snapshot_id, msg):
            if snapshot_id:
                # Restore previous state (efficient - no reprocessing!)
                s.restore_snapshot(snapshot_id)

            s += f"user: {msg}\n"
            s += "assistant: " + gen("response", max_tokens=100)

            # Save updated state
            new_snapshot = s.save_snapshot()
            return s, new_snapshot

        result, self.current_snapshot = chat_turn.run(
            snapshot_id=self.current_snapshot,
            msg=message,
            engine=self.engine
        )

        return result["response"]

# Usage
session = ChatSession(engine)
response1 = session.send_message("Hello")
response2 = session.send_message("How are you?")  # Efficient - uses snapshot!
```

### Scenario 2: Document Processing

#### Before

```python
# document_old.py
@function
def process_document(s, document):
    # Process in one go
    s += f"Document: {document}\n\n"
    s += "Summary: " + gen("summary", max_tokens=200)
    s += "\n\nKey points: " + gen("key_points", max_tokens=150)
    s += "\n\nQuestions: " + gen("questions", max_tokens=100)

    return s

# If generation fails partway, must start over
```

#### After (with checkpoints)

```python
# document_new.py
@function
def process_document(s, document):
    # Stage 1: Summary
    s += f"Document: {document}\n\n"
    s += "Summary: " + gen("summary", max_tokens=200)
    checkpoint1 = s.save_snapshot(metadata={"stage": "summary"})

    # Stage 2: Key points
    s += "\n\nKey points: " + gen("key_points", max_tokens=150)
    checkpoint2 = s.save_snapshot(metadata={"stage": "key_points"})

    # Stage 3: Questions
    s += "\n\nQuestions: " + gen("questions", max_tokens=100)

    return s, checkpoint1, checkpoint2

# If stage 3 fails, can resume from checkpoint2!
```

### Scenario 3: A/B Testing

#### Before

```python
# ab_test_old.py
# Must regenerate from scratch for each variant
@function
def variant_a(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100, temperature=0.7)
    return s

@function
def variant_b(s, prompt):
    s += prompt  # Redundant processing
    s += gen("response", max_tokens=100, temperature=1.2)
    return s

result_a = variant_a.run(prompt="Write a poem", engine=engine)
result_b = variant_b.run(prompt="Write a poem", engine=engine)
```

#### After (with snapshots)

```python
# ab_test_new.py
@function
def setup_ab_test(s, prompt):
    s += prompt
    snapshot_id = s.save_snapshot()
    return snapshot_id

@function
def run_variant(s, snapshot_id, temperature):
    s.restore_snapshot(snapshot_id)  # Efficient - reuse base state!
    s += gen("response", max_tokens=100, temperature=temperature)
    return s

# Setup once
base_snapshot = setup_ab_test.run(prompt="Write a poem", engine=engine)

# Run variants efficiently
result_a = run_variant.run(snapshot_id=base_snapshot, temperature=0.7, engine=engine)
result_b = run_variant.run(snapshot_id=base_snapshot, temperature=1.2, engine=engine)
```

## Breaking Changes

**None.** The snapshot system is fully backward compatible.

However, be aware of these **behavioral changes** when snapshots are enabled:

### Memory Usage

Snapshots consume additional GPU memory:

```python
# Before: 100% of allocated memory for inference
# After: ~90% for inference, ~10% for snapshots (configurable)
```

Configure memory allocation:

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    mem_fraction_static=0.85,  # Reduce from 0.9 to leave room for snapshots
    enable_mamba_snapshots=True,
    snapshot_config={
        "max_snapshot_memory_gb": 5.0,  # Limit snapshot memory
    }
)
```

### Reference Counting

Radix cache nodes referenced by snapshots won't be evicted:

```python
# This is usually beneficial, but if you create many snapshots
# without cleanup, it can reduce cache effectiveness

# Best practice: Clean up unused snapshots
@function
def with_cleanup(s):
    snap_id = s.save_snapshot()
    # Use snapshot...
    s.delete_snapshot(snap_id)  # Clean up when done
    return s
```

## Performance Impact

### When Snapshots Are Disabled (default)

- **Overhead**: Zero
- **Memory**: No change
- **Performance**: Identical to pre-v0.5.0

### When Snapshots Are Enabled But Not Used

- **Overhead**: Minimal (<1%)
- **Memory**: Small registry overhead (<1 MB)
- **Performance**: Negligible impact

### When Actively Using Snapshots

- **Save Operation**: O(1) - very fast (~0.1ms)
- **Restore Operation**: O(1) without COW, O(S) with COW (where S = state size)
- **Memory**: Additional ~50-100 MB per snapshot (varies by model)

### Benchmark Comparison

```python
# Benchmark script
import time
from sglang import Engine
from sglang.lang import function, gen

# Without snapshots
engine_old = Engine(model_path="state-spaces/mamba-2.8b")

@function
def without_snapshots(s):
    for i in range(10):
        s += gen(f"text_{i}", max_tokens=50)
    return s

start = time.time()
result = without_snapshots.run(engine=engine_old)
time_old = time.time() - start
print(f"Without snapshots: {time_old:.2f}s")

# With snapshots (but not using them)
engine_new = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True
)

start = time.time()
result = without_snapshots.run(engine=engine_new)
time_new = time.time() - start
print(f"With snapshots (not used): {time_new:.2f}s")
print(f"Overhead: {(time_new - time_old) / time_old * 100:.2f}%")

# With snapshots (actively using)
@function
def with_snapshots(s):
    snapshots = []
    for i in range(10):
        s += gen(f"text_{i}", max_tokens=50)
        snapshots.append(s.save_snapshot())
    return s, snapshots

start = time.time()
result, snaps = with_snapshots.run(engine=engine_new)
time_with_snaps = time.time() - start
print(f"With snapshots (active): {time_with_snaps:.2f}s")
print(f"Snapshot overhead: {(time_with_snaps - time_old) / time_old * 100:.2f}%")
```

## Rollback Procedure

If you need to disable snapshots after enabling them:

### Option 1: Disable Globally

```python
# Change engine config
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=False,  # Disable
)
```

### Option 2: Disable Per-Request

```python
@function
def my_function(s):
    s.disable_snapshots()  # Disable for this request
    # snapshot operations will raise SnapshotDisabledError
    return s
```

### Option 3: Remove Snapshot Code

Snapshot code gracefully handles disabled snapshots:

```python
@function
def safe_function(s):
    try:
        snap_id = s.save_snapshot()
    except SnapshotDisabledError:
        snap_id = None  # Continue without snapshot

    # Rest of code...
    return s
```

### Clean Up Persisted Snapshots

```bash
# Remove stored snapshots from disk
rm -f ./snapshots/*.snapshot
```

## Migration Checklist

Use this checklist when migrating to snapshot-enabled inference:

- [ ] Upgrade SGLang to v0.5.0+
- [ ] Check model compatibility (Mamba/Mamba2 only)
- [ ] Estimate memory requirements
- [ ] Adjust `mem_fraction_static` if needed
- [ ] Enable snapshots in engine config
- [ ] Update code to use snapshot operations (optional)
- [ ] Test snapshot save/restore
- [ ] Monitor memory usage
- [ ] Set up snapshot cleanup strategy
- [ ] Update documentation for your application
- [ ] Configure disk persistence (if needed)
- [ ] Set up monitoring/alerts for snapshot memory

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
