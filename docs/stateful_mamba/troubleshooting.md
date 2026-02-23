# Troubleshooting Guide: Stateful Mamba Snapshots

> **⚠️ Implementation Status:** Phase 1 (Snapshot Saving) is complete. Phase 2 (State Restoration) is in development.
>
> **Available Now:** `save_snapshot()`, `list_snapshots()`, `SnapshotManager` class (for restore/get_info)
> **Coming Soon:** Direct state object methods `s.restore_snapshot()`, `s.get_snapshot_info()`

Common issues and solutions for the snapshot system.

## Table of Contents

- [Phase 1 Limitations](#phase-1-limitations)
- [Common Issues](#common-issues)
- [Error Messages](#error-messages)
- [Memory Problems](#memory-problems)
- [FAQ](#faq)

## Phase 1 Limitations

The current implementation (Phase 1) provides snapshot **saving and inspection only**.

**What works:**
- ✅ `s.save_snapshot()` - Save current state (direct method)
- ✅ `s.list_snapshots()` - List saved snapshots (via endpoint)
- ✅ `SnapshotManager` class - Use `sm.restore()`, `sm.get_info()`, `sm.delete()` for full API

**What doesn't work yet (Phase 2 - direct state object methods):**
- ❌ `s.restore_snapshot()` - Use `SnapshotManager.restore()` instead
- ❌ `s.get_snapshot_info()` - Use `SnapshotManager.get_info()` instead
- ❌ Automatic snapshot management (retention policies, hooks)

## Common Issues

### Issue: "Backend does not support snapshots"

**Symptom**: Getting an error that snapshots are not supported when calling `save_snapshot()`.

**Cause**: The runtime wasn't started with snapshot support enabled.

**Solution**:

```python
# Wrong - this parameter doesn't exist
runtime = Runtime(enable_mamba_snapshots=True)  # ❌

# Correct - use this parameter
runtime = Runtime(enable_snapshot_persistence=True)  # ✅
```

Or via command line:
```bash
# Wrong
python -m sglang.launch_server --enable-mamba-snapshots  # ❌

# Correct
python -m sglang.launch_server --enable-snapshot-persistence  # ✅
```

---

### Issue: Snapshot Not Found

**Symptom**: `list_snapshots()` returns an empty list or a snapshot you expect to be there is missing.

**Cause**: Snapshot wasn't saved, or wrong conversation_id used.

**Solution**:

```python
# List all snapshots to see what's available
snapshots = s.list_snapshots()
for snap in snapshots:
    print(f"Turn: {snap['turn_number']}, Tokens: {snap['token_count']}")

# Ensure save_snapshot() was called before listing
snap_id = s.save_snapshot()
snapshots = s.list_snapshots()
print(f"Saved snap_id={snap_id}, total={len(snapshots)}")
```

> **Note:** `get_snapshot_info()` for querying individual snapshot metadata is planned for Phase 2.

---

### Issue: Slow Snapshot Operations

**Symptom**: `save_snapshot()` is slow.

**Diagnosis**:

```python
import time

@function
def benchmark_snapshot(s):
    s += gen("text", max_tokens=100)

    # Benchmark save
    start = time.time()
    snap_id = s.save_snapshot()
    save_time = time.time() - start
    print(f"Save time: {save_time*1000:.2f}ms")

    return s
```

**Solutions**:

1. **Check for disk I/O bottleneck**: Ensure snapshot directory is on fast storage (SSD)

2. **Reduce snapshot frequency**:

```python
# Don't create snapshots too frequently
@function
def efficient_snapshots(s):
    for i in range(100):
        s += gen(f"text_{i}", max_tokens=10)
        # Only snapshot every 10 iterations
        if i % 10 == 0:
            s.save_snapshot()
    return s
```

---

## Error Messages

### "Snapshot directory does not exist"

**Full Error**: `FileNotFoundError: Snapshot directory './snapshots' does not exist`

**Cause**: The specified snapshot directory hasn't been created.

**Solution**:

```python
import os

# Create snapshot directory before running
snapshot_dir = "./my_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

runtime = Runtime(
    model_path="state-spaces/mamba-2.8b",
    enable_snapshot_persistence=True,
    snapshot_dir=snapshot_dir
)
```

---

### "Failed to save snapshot"

**Full Error**: Various errors during snapshot save operation.

**Cause**: Disk space issues, permission problems, or I/O errors.

**Solutions**:

1. **Check disk space**:

```bash
df -h ./snapshots
```

2. **Check permissions**:

```bash
ls -ld ./snapshots
chmod 755 ./snapshots  # If needed
```

3. **Verify snapshot directory is writable**:

```python
import os
snapshot_dir = "./my_snapshots"
if not os.access(snapshot_dir, os.W_OK):
    print(f"Directory {snapshot_dir} is not writable")
```

---

## Memory Problems

### Diagnosing Memory Issues

```python
import torch

# Check GPU memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU allocated: {allocated:.2f} GB")
    print(f"GPU reserved: {reserved:.2f} GB")

# List snapshots to understand storage
snapshots = s.list_snapshots()
print(f"Number of snapshots: {len(snapshots)}")

for snap in snapshots:
    print(f"Snapshot - Turn: {snap['turn_number']}, Tokens: {snap['token_count']}")
```

---

## FAQ

### Q: Can I use snapshots with transformer models?

**A**: No, snapshots are Mamba-specific. Transformer models use KV cache, which is already optimized by the radix cache system.

---

### Q: How much memory does each snapshot use?

**A**: Typically 50-100 MB for Mamba-2.8B, varies by model size and sequence length. Check the snapshot metadata for details.

---

### Q: Can I restore snapshots in Phase 1?

**A**: No, Phase 1 only supports saving and inspecting snapshots. Restoration will be available in Phase 2.

---

### Q: Can I share snapshots between different models?

**A**: No, snapshots are model-specific. They contain SSM states that are only compatible with the exact model architecture.

---

### Q: Are snapshots serializable for remote storage (S3, etc.)?

**A**: Yes, the snapshot files saved to disk can be uploaded to remote storage. They are saved in safetensors format with JSON metadata.

```python
# Snapshots are saved to the configured directory
# You can upload these files to S3 or other storage
import boto3
s3 = boto3.client('s3')
s3.upload_file("./snapshots/snapshot_file.safetensors", "my-bucket", "snapshots/snapshot_file.safetensors")
```

---

### Q: Can I snapshot in the middle of a generation?

**A**: You can save a snapshot at any point, but it captures state up to the last completed token. Mid-generation snapshots capture state before the current `gen()` call completes.

---

### Q: What file format are snapshots saved in?

**A**: Snapshots are saved in safetensors format with accompanying JSON metadata files.

---

## Getting Help

If you're still experiencing issues:

1. **Check logs**: Enable debug logging to see detailed error messages
2. **GitHub Issues**: [Search or create an issue](https://github.com/sgl-project/sglang/issues)
3. **Discussions**: [Ask the community](https://github.com/sgl-project/sglang/discussions)
4. **Slack**: [Join #support channel](https://slack.sglang.io/)

When reporting issues, please include:
- SGLang version
- Model name
- Minimal reproduction code
- Full error traceback
- GPU memory usage
- Snapshot configuration

## See Also

- [User Guide](user_guide.md) - Usage patterns and best practices
- [API Reference](api_reference.md) - Detailed API documentation
- [Architecture](architecture.md) - System design and internals
- [Migration Guide](migration_guide.md) - Upgrading to snapshot support
