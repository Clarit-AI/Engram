# Stateful Mamba API Reference

Complete API documentation for the snapshot system in SGLang.

## Table of Contents

- [Frontend API](#frontend-api)
- [Snapshot Manager](#snapshot-manager)
- [Snapshot Class](#snapshot-class)
- [Snapshot Registry](#snapshot-registry)
- [Snapshot Serializer](#snapshot-serializer)
- [Configuration](#configuration)
- [Exceptions](#exceptions)
- [Utility Functions](#utility-functions)

## Frontend API

### SglContext Snapshot Methods

Methods available in the SGLang frontend context (`s` object).

#### `save_snapshot()`

Save the current state as a snapshot.

```python
def save_snapshot(
    snapshot_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters**:
- `snapshot_id` (str, optional): Custom ID for the snapshot. If None, auto-generates a UUID.
- `metadata` (dict, optional): User-defined metadata to attach to the snapshot.

**Returns**:
- `str`: The snapshot ID.

**Raises**:
- `SnapshotDisabledError`: If snapshots are not enabled.
- `OutOfMemoryError`: If insufficient GPU memory for snapshot.
- `ValueError`: If `snapshot_id` already exists.

**Example**:
```python
@function
def my_function(s):
    s += gen("text", max_tokens=100)

    # Auto-generated ID
    snap_id = s.save_snapshot()

    # Custom ID with metadata
    snap_id = s.save_snapshot(
        "checkpoint_1",
        metadata={"turn": 1, "important": True}
    )

    return snap_id
```

---

#### `restore_snapshot()`

Restore state from a previously saved snapshot.

```python
def restore_snapshot(snapshot_id: str) -> None
```

**Parameters**:
- `snapshot_id` (str): ID of the snapshot to restore.

**Returns**:
- None

**Raises**:
- `SnapshotNotFoundError`: If snapshot ID doesn't exist.
- `SnapshotInvalidError`: If snapshot state is corrupted.

**Example**:
```python
@function
def continue_from_checkpoint(s, snapshot_id):
    s.restore_snapshot(snapshot_id)
    s += gen("continuation", max_tokens=100)
    return s
```

---

#### `delete_snapshot()`

Delete a snapshot and free its resources.

```python
def delete_snapshot(snapshot_id: str) -> bool
```

**Parameters**:
- `snapshot_id` (str): ID of the snapshot to delete.

**Returns**:
- `bool`: True if deleted, False if not found.

**Raises**:
- `SnapshotInUseError`: If snapshot is currently being used by other requests.

**Example**:
```python
@function
def cleanup_example(s):
    snap_id = s.save_snapshot()
    # ... use snapshot ...
    s.delete_snapshot(snap_id)
    return s
```

---

#### `persist_snapshot()`

Save a snapshot to disk.

```python
def persist_snapshot(
    snapshot_id: str,
    path: Optional[str] = None,
    async_mode: bool = True
) -> Optional[Future]
```

**Parameters**:
- `snapshot_id` (str): ID of the snapshot to persist.
- `path` (str, optional): File path. If None, uses default storage path with snapshot_id.
- `async_mode` (bool): If True, performs I/O asynchronously. Default: True.

**Returns**:
- `Future` if async_mode=True, None otherwise.

**Raises**:
- `SnapshotNotFoundError`: If snapshot doesn't exist.
- `IOError`: If disk write fails.

**Example**:
```python
@function
def save_to_disk(s):
    snap_id = s.save_snapshot()

    # Async persist
    future = s.persist_snapshot(snap_id, "./snapshots/my_snap.bin")

    # Wait for completion if needed
    if future:
        future.result()  # Blocks until write complete

    return snap_id
```

---

#### `load_snapshot()`

Load a snapshot from disk.

```python
def load_snapshot(path: str) -> str
```

**Parameters**:
- `path` (str): Path to the saved snapshot file.

**Returns**:
- `str`: The loaded snapshot ID.

**Raises**:
- `FileNotFoundError`: If file doesn't exist.
- `SnapshotDeserializationError`: If file is corrupted or incompatible.

**Example**:
```python
@function
def restore_from_disk(s, snapshot_path):
    snap_id = s.load_snapshot(snapshot_path)
    s.restore_snapshot(snap_id)
    s += gen("continuation", max_tokens=100)
    return s
```

---

#### `enable_snapshots()`

Enable snapshots for the current request (if disabled globally).

```python
def enable_snapshots() -> None
```

**Example**:
```python
@function
def my_function(s):
    s.enable_snapshots()  # Enable for this request only
    snap_id = s.save_snapshot()
    return snap_id
```

---

#### `disable_snapshots()`

Disable snapshots for the current request.

```python
def disable_snapshots() -> None
```

**Example**:
```python
@function
def my_function(s):
    s.disable_snapshots()  # Disable for this request
    # save_snapshot() will raise SnapshotDisabledError
    return s
```

---

## Snapshot Manager

The `SnapshotManager` class provides programmatic access to snapshot operations.

### Initialization

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)
```

**Parameters**:
- `engine` (Engine): The SGLang engine instance.

---

### Methods

#### `get_snapshot()`

Retrieve a snapshot by ID.

```python
def get_snapshot(snapshot_id: str) -> Optional[Snapshot]
```

**Parameters**:
- `snapshot_id` (str): The snapshot ID.

**Returns**:
- `Snapshot` object or None if not found.

**Example**:
```python
manager = SnapshotManager(engine)
snapshot = manager.get_snapshot("checkpoint_1")

if snapshot:
    print(f"Tokens: {len(snapshot.token_ids)}")
    print(f"Metadata: {snapshot.metadata}")
```

---

#### `list_snapshots()`

List all active snapshots.

```python
def list_snapshots(
    filter_fn: Optional[Callable[[Snapshot], bool]] = None,
    sort_by: str = "timestamp",
    ascending: bool = False
) -> List[Snapshot]
```

**Parameters**:
- `filter_fn` (callable, optional): Filter function. Snapshot included if returns True.
- `sort_by` (str): Sort key. Options: "timestamp", "seq_len", "snapshot_id". Default: "timestamp".
- `ascending` (bool): Sort order. Default: False (newest first).

**Returns**:
- `List[Snapshot]`: List of snapshot objects.

**Example**:
```python
manager = SnapshotManager(engine)

# All snapshots, newest first
all_snaps = manager.list_snapshots()

# Filter by metadata
important = manager.list_snapshots(
    filter_fn=lambda s: s.metadata.get("important", False)
)

# Sort by sequence length
by_length = manager.list_snapshots(sort_by="seq_len", ascending=True)
```

---

#### `delete_snapshot()`

Delete a snapshot.

```python
def delete_snapshot(snapshot_id: str) -> bool
```

**Parameters**:
- `snapshot_id` (str): Snapshot to delete.

**Returns**:
- `bool`: True if deleted, False if not found.

**Example**:
```python
manager = SnapshotManager(engine)
deleted = manager.delete_snapshot("checkpoint_1")
```

---

#### `delete_snapshots_before()`

Delete snapshots older than a timestamp.

```python
def delete_snapshots_before(timestamp: float) -> int
```

**Parameters**:
- `timestamp` (float): Unix timestamp cutoff.

**Returns**:
- `int`: Number of snapshots deleted.

**Example**:
```python
import time
manager = SnapshotManager(engine)

# Delete snapshots older than 1 hour
cutoff = time.time() - 3600
deleted_count = manager.delete_snapshots_before(cutoff)
print(f"Deleted {deleted_count} old snapshots")
```

---

#### `clear_all_snapshots()`

Delete all snapshots.

```python
def clear_all_snapshots() -> int
```

**Returns**:
- `int`: Number of snapshots deleted.

**Example**:
```python
manager = SnapshotManager(engine)
count = manager.clear_all_snapshots()
print(f"Deleted {count} snapshots")
```

---

#### `get_total_snapshot_memory()`

Get total GPU memory used by snapshots.

```python
def get_total_snapshot_memory() -> int
```

**Returns**:
- `int`: Memory in bytes.

**Example**:
```python
manager = SnapshotManager(engine)
memory_bytes = manager.get_total_snapshot_memory()
memory_gb = memory_bytes / 1e9
print(f"Snapshot memory: {memory_gb:.2f} GB")
```

---

#### `persist_snapshot()`

Save a snapshot to disk.

```python
def persist_snapshot(
    snapshot_id: str,
    path: str,
    format: str = "safetensors",
    compress: bool = True,
    async_mode: bool = True
) -> Optional[Future]
```

**Parameters**:
- `snapshot_id` (str): Snapshot to persist.
- `path` (str): Destination file path.
- `format` (str): Serialization format. Options: "safetensors", "pickle", "custom". Default: "safetensors".
- `compress` (bool): Enable compression. Default: True.
- `async_mode` (bool): Asynchronous I/O. Default: True.

**Returns**:
- `Future` if async, None otherwise.

**Example**:
```python
manager = SnapshotManager(engine)

# Sync persist
manager.persist_snapshot(
    "checkpoint_1",
    "./snapshots/checkpoint_1.bin",
    async_mode=False
)

# Async persist
future = manager.persist_snapshot(
    "checkpoint_2",
    "./snapshots/checkpoint_2.bin",
    async_mode=True
)
# Do other work...
future.result()  # Wait for completion
```

---

#### `load_snapshot()`

Load a snapshot from disk.

```python
def load_snapshot(path: str) -> str
```

**Parameters**:
- `path` (str): Path to snapshot file.

**Returns**:
- `str`: The loaded snapshot ID.

**Example**:
```python
manager = SnapshotManager(engine)
snap_id = manager.load_snapshot("./snapshots/checkpoint_1.bin")
```

---

## Snapshot Class

The `Snapshot` class represents a single snapshot instance.

### Attributes

```python
@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str                    # Unique identifier
    mamba_state_indices: torch.Tensor   # GPU memory indices
    token_ids: List[int]                # Token sequence
    seq_len: int                        # Sequence length
    last_node: TreeNode                 # Radix cache node
    metadata: Dict[str, Any]            # User metadata
    timestamp: float                    # Creation time (Unix timestamp)
    ref_count: int                      # Reference count
```

**Note**: Snapshots are immutable after creation.

### Properties

#### `age`

Get snapshot age in seconds.

```python
@property
def age(self) -> float
```

**Example**:
```python
snapshot = manager.get_snapshot("checkpoint_1")
print(f"Snapshot age: {snapshot.age:.2f} seconds")
```

---

#### `memory_size`

Estimate memory size in bytes.

```python
@property
def memory_size(self) -> int
```

**Example**:
```python
snapshot = manager.get_snapshot("checkpoint_1")
size_mb = snapshot.memory_size / 1e6
print(f"Snapshot size: {size_mb:.2f} MB")
```

---

#### `token_text`

Get decoded token text (if tokenizer available).

```python
@property
def token_text(self) -> str
```

**Example**:
```python
snapshot = manager.get_snapshot("checkpoint_1")
print(f"Text: {snapshot.token_text}")
```

---

## Snapshot Registry

The `SnapshotRegistry` manages the collection of active snapshots.

**Note**: Typically accessed through `SnapshotManager`, not directly.

### Methods

#### `register()`

Register a new snapshot.

```python
def register(snapshot: Snapshot) -> None
```

---

#### `get()`

Retrieve a snapshot by ID.

```python
def get(snapshot_id: str) -> Optional[Snapshot]
```

---

#### `delete()`

Delete a snapshot.

```python
def delete(snapshot_id: str) -> bool
```

---

#### `inc_ref()`

Increment reference count.

```python
def inc_ref(snapshot_id: str) -> None
```

---

#### `dec_ref()`

Decrement reference count. Deletes snapshot if count reaches 0.

```python
def dec_ref(snapshot_id: str) -> None
```

---

## Snapshot Serializer

Handles serialization/deserialization for disk persistence.

### Methods

#### `serialize()`

Serialize a snapshot to disk.

```python
def serialize(
    snapshot: Snapshot,
    path: str,
    format: str = "safetensors",
    compress: bool = True
) -> None
```

**Parameters**:
- `snapshot` (Snapshot): Snapshot to serialize.
- `path` (str): Destination path.
- `format` (str): Format. Options: "safetensors", "pickle", "custom".
- `compress` (bool): Enable compression.

---

#### `deserialize()`

Deserialize a snapshot from disk.

```python
def deserialize(path: str) -> Snapshot
```

**Parameters**:
- `path` (str): Source path.

**Returns**:
- `Snapshot`: Loaded snapshot.

---

#### `get_serialized_size()`

Estimate serialized size.

```python
def get_serialized_size(snapshot: Snapshot) -> int
```

**Returns**:
- `int`: Estimated size in bytes.

---

## Configuration

### SnapshotConfig

Configuration class for snapshot system.

```python
@dataclass
class SnapshotConfig:
    # Enable/disable
    enabled: bool = False

    # Memory management
    max_snapshots: int = 100
    max_snapshot_memory_gb: float = 10.0
    eviction_policy: str = "lru"  # "lru", "fifo", "manual"

    # Copy-on-write
    enable_cow: bool = True

    # Disk persistence
    enable_persistence: bool = False
    storage_path: str = "./snapshots"
    auto_persist: bool = False
    persist_delay_seconds: float = 60.0

    # Serialization
    serialization_format: str = "safetensors"  # "safetensors", "pickle", "custom"
    compression_enabled: bool = True
    compression_level: int = 6  # 0-9, higher = more compression

    # I/O
    async_io: bool = True
    io_threads: int = 4

    # Monitoring
    enable_metrics: bool = True
    metrics_interval_seconds: float = 60.0
```

### Usage

```python
from sglang import Engine
from sglang.snapshot import SnapshotConfig

config = SnapshotConfig(
    enabled=True,
    max_snapshots=50,
    enable_cow=True,
    enable_persistence=True,
    storage_path="/mnt/snapshots"
)

engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    snapshot_config=config
)
```

Or using dict:

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "max_snapshots": 50,
        "enable_cow": True,
        "storage_path": "/mnt/snapshots"
    }
)
```

---

## Exceptions

### SnapshotError

Base exception for all snapshot-related errors.

```python
class SnapshotError(Exception):
    """Base exception for snapshot operations."""
    pass
```

---

### SnapshotDisabledError

Raised when snapshot operations are attempted but snapshots are disabled.

```python
class SnapshotDisabledError(SnapshotError):
    """Snapshots are not enabled."""
    pass
```

**Example**:
```python
try:
    s.save_snapshot()
except SnapshotDisabledError:
    print("Please enable snapshots first")
```

---

### SnapshotNotFoundError

Raised when a snapshot ID is not found.

```python
class SnapshotNotFoundError(SnapshotError):
    """Snapshot ID not found."""
    pass
```

**Example**:
```python
try:
    s.restore_snapshot("nonexistent_id")
except SnapshotNotFoundError as e:
    print(f"Snapshot not found: {e}")
```

---

### SnapshotInvalidError

Raised when a snapshot is corrupted or invalid.

```python
class SnapshotInvalidError(SnapshotError):
    """Snapshot state is invalid or corrupted."""
    pass
```

---

### SnapshotInUseError

Raised when attempting to delete a snapshot that's currently in use.

```python
class SnapshotInUseError(SnapshotError):
    """Snapshot is currently being used and cannot be deleted."""
    pass
```

**Example**:
```python
try:
    manager.delete_snapshot("in_use_snap")
except SnapshotInUseError:
    print("Snapshot is in use by active requests")
```

---

### SnapshotDeserializationError

Raised when loading a snapshot from disk fails.

```python
class SnapshotDeserializationError(SnapshotError):
    """Failed to deserialize snapshot from disk."""
    pass
```

**Example**:
```python
try:
    snap_id = manager.load_snapshot("./corrupt.snapshot")
except SnapshotDeserializationError as e:
    print(f"Failed to load snapshot: {e}")
```

---

### OutOfMemoryError

Raised when insufficient GPU memory for snapshot operation.

```python
class OutOfMemoryError(SnapshotError):
    """Insufficient GPU memory for snapshot operation."""
    pass
```

**Example**:
```python
try:
    snap_id = s.save_snapshot()
except OutOfMemoryError:
    print("Out of memory, cleaning up old snapshots")
    manager.clear_all_snapshots()
```

---

## Utility Functions

### `generate_snapshot_id()`

Generate a unique snapshot ID.

```python
def generate_snapshot_id(prefix: str = "") -> str
```

**Parameters**:
- `prefix` (str, optional): Prefix for the ID.

**Returns**:
- `str`: Unique snapshot ID (UUID-based).

**Example**:
```python
from sglang.snapshot import generate_snapshot_id

snap_id = generate_snapshot_id("checkpoint")
# Example: "checkpoint_550e8400-e29b-41d4-a716-446655440000"
```

---

### `validate_snapshot_id()`

Validate a snapshot ID format.

```python
def validate_snapshot_id(snapshot_id: str) -> bool
```

**Parameters**:
- `snapshot_id` (str): ID to validate.

**Returns**:
- `bool`: True if valid, False otherwise.

**Example**:
```python
from sglang.snapshot import validate_snapshot_id

valid = validate_snapshot_id("checkpoint_1")  # True
valid = validate_snapshot_id("")  # False
```

---

### `get_snapshot_info()`

Get detailed information about a snapshot.

```python
def get_snapshot_info(snapshot: Snapshot) -> Dict[str, Any]
```

**Parameters**:
- `snapshot` (Snapshot): The snapshot to inspect.

**Returns**:
- `dict`: Detailed information.

**Example**:
```python
from sglang.snapshot import get_snapshot_info

snapshot = manager.get_snapshot("checkpoint_1")
info = get_snapshot_info(snapshot)

print(f"ID: {info['snapshot_id']}")
print(f"Size: {info['memory_mb']:.2f} MB")
print(f"Tokens: {info['num_tokens']}")
print(f"Age: {info['age_seconds']:.2f}s")
print(f"Metadata: {info['metadata']}")
```

---

### `compare_snapshots()`

Compare two snapshots.

```python
def compare_snapshots(
    snapshot1: Snapshot,
    snapshot2: Snapshot
) -> Dict[str, Any]
```

**Parameters**:
- `snapshot1` (Snapshot): First snapshot.
- `snapshot2` (Snapshot): Second snapshot.

**Returns**:
- `dict`: Comparison results.

**Example**:
```python
from sglang.snapshot import compare_snapshots

snap1 = manager.get_snapshot("checkpoint_1")
snap2 = manager.get_snapshot("checkpoint_2")

comparison = compare_snapshots(snap1, snap2)

print(f"Common prefix length: {comparison['common_prefix_len']}")
print(f"Divergence point: {comparison['divergence_point']}")
print(f"Length diff: {comparison['length_diff']}")
```

---

## Type Hints

For type checking, import the following types:

```python
from sglang.snapshot import (
    Snapshot,
    SnapshotManager,
    SnapshotConfig,
    SnapshotRegistry,
    SnapshotSerializer,
)
from sglang.snapshot.exceptions import (
    SnapshotError,
    SnapshotNotFoundError,
    SnapshotInvalidError,
    SnapshotDisabledError,
    SnapshotInUseError,
    SnapshotDeserializationError,
    OutOfMemoryError,
)
```

---

## Version Compatibility

- **Minimum SGLang version**: v0.5.0
- **Python version**: 3.8+
- **Model support**: Mamba, Mamba2 architectures only

---

## See Also

- [User Guide](user_guide.md) - Usage examples and patterns
- [Architecture](architecture.md) - Technical architecture details
- [Examples](examples.md) - Real-world examples
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
