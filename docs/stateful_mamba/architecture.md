# Stateful Mamba Architecture

This document provides a technical deep-dive into the snapshot system architecture for Mamba models in SGLang.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [System Architecture](#system-architecture)
- [Data Structures](#data-structures)
- [Memory Management](#memory-management)
- [Concurrency Model](#concurrency-model)
- [Integration Points](#integration-points)
- [Design Decisions](#design-decisions)

## Overview

The snapshot system extends SGLang's Mamba implementation to support saving and restoring SSM (State Space Model) hidden states. The design prioritizes:

1. **Zero overhead** when snapshots are disabled
2. **Minimal copying** during snapshot operations
3. **Seamless integration** with existing radix cache
4. **Thread-safe** snapshot management
5. **Backward compatibility** with existing workflows

## Core Components

### 1. SnapshotManager

**Location**: `python/sglang/srt/snapshot/snapshot_manager.py`

The central coordinator for all snapshot operations.

```python
class SnapshotManager:
    """
    Manages snapshot lifecycle including creation, restoration,
    persistence, and cleanup.
    """

    def __init__(self, mem_pool, radix_cache, config):
        self.registry = SnapshotRegistry()
        self.mem_pool = mem_pool
        self.radix_cache = radix_cache
        self.serializer = SnapshotSerializer(config)
        self.lock = threading.RLock()

    def save_snapshot(self, req: Req, snapshot_id: str) -> Snapshot
    def restore_snapshot(self, snapshot_id: str, req: Req) -> None
    def delete_snapshot(self, snapshot_id: str) -> None
    def persist_to_disk(self, snapshot_id: str, path: str) -> None
    def load_from_disk(self, path: str) -> str
```

**Key Responsibilities**:
- Snapshot lifecycle management
- Coordination with memory pool and radix cache
- Thread-safe access to snapshot registry
- Disk I/O operations (optional)

### 2. SnapshotRegistry

**Location**: `python/sglang/srt/snapshot/snapshot_registry.py`

A thread-safe registry maintaining all active snapshots.

```python
class SnapshotRegistry:
    """
    Thread-safe registry for snapshot metadata and state references.
    Implements reference counting for memory management.
    """

    def __init__(self):
        self.snapshots: Dict[str, Snapshot] = {}
        self.lock = threading.RLock()
        self.ref_counts: Dict[str, int] = {}

    def register(self, snapshot: Snapshot) -> None
    def get(self, snapshot_id: str) -> Optional[Snapshot]
    def delete(self, snapshot_id: str) -> bool
    def inc_ref(self, snapshot_id: str) -> None
    def dec_ref(self, snapshot_id: str) -> None
```

**Key Features**:
- Fast O(1) lookup by snapshot ID
- Reference counting for safe deletion
- Thread-safe with fine-grained locking
- Automatic cleanup when ref count reaches zero

### 3. Snapshot Data Class

**Location**: `python/sglang/srt/snapshot/snapshot.py`

Represents a single snapshot instance.

```python
@dataclass
class Snapshot:
    """
    Immutable snapshot of Mamba SSM state at a specific point.
    """
    snapshot_id: str
    mamba_state_indices: torch.Tensor  # Indices into memory pool
    token_ids: List[int]               # Token sequence up to this point
    seq_len: int                       # Sequence length
    last_node: TreeNode                # Radix cache node reference
    metadata: Dict[str, Any]           # User-provided metadata
    timestamp: float                   # Creation time
    ref_count: int = 0                 # Reference count

    def __post_init__(self):
        # Snapshots are immutable after creation
        object.__setattr__(self, 'frozen', True)
```

**Design Notes**:
- Immutable after creation (enforced via `__setattr__`)
- Stores references, not copies, of SSM states
- Lightweight metadata storage
- Compatible with serialization

### 4. SnapshotSerializer

**Location**: `python/sglang/srt/snapshot/snapshot_serializer.py`

Handles serialization/deserialization for disk persistence.

```python
class SnapshotSerializer:
    """
    Serializes snapshots to disk and deserializes them back.
    Supports multiple formats: safetensors, pickle, custom binary.
    """

    def __init__(self, config: SnapshotConfig):
        self.format = config.serialization_format
        self.compression = config.compression_enabled

    def serialize(self, snapshot: Snapshot, path: str) -> None
    def deserialize(self, path: str) -> Snapshot
    def get_serialized_size(self, snapshot: Snapshot) -> int
```

**Supported Formats**:
- **safetensors** (default): Safe, portable, efficient
- **pickle**: Python native, debugging-friendly
- **custom_binary**: Maximum efficiency, minimal overhead

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend API Layer                        │
│  - save_snapshot(snapshot_id)                               │
│  - restore_snapshot(snapshot_id)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Request
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Snapshot Manager                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Validate request                                   │  │
│  │ 2. Coordinate with radix cache                        │  │
│  │ 3. Update registry                                    │  │
│  │ 4. Manage memory references                           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────┬───────────────────────────┬────────────────────────┘
         │                           │
         │ State Refs                │ Cache Node
         ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Snapshot Registry  │    │   Mamba Radix Cache  │
│                      │    │                      │
│  - Snapshot metadata │    │  - Tree structure    │
│  - Reference counts  │    │  - Lock management   │
│  - Quick lookup      │    │  - LRU eviction      │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           │ Indices                   │ Nodes
           ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Memory Pool (GPU)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Full KV Cache         │  Mamba SSM States           │  │
│  │  [Token-based]         │  [Request-based]            │  │
│  │                        │                             │  │
│  │  - Transformer KV      │  - Conv states              │  │
│  │  - Radix tree refs     │  - SSM states               │  │
│  │                        │  - Snapshot refs (NEW)      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Snapshot Save Flow

```
User Request
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. Frontend: s.save_snapshot(id)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. SnapshotManager.save_snapshot()      │
│    - Generate/validate snapshot_id      │
│    - Extract current request state      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Radix Cache: Lock current node       │
│    - Increment lock refs                │
│    - Prevent eviction                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Memory Pool: Increment ref count     │
│    - Mark SSM states as referenced      │
│    - Prevent deallocation               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. Create Snapshot object               │
│    - Store state indices (reference)    │
│    - Store token sequence               │
│    - Store radix cache node             │
│    - Add user metadata                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 6. Registry: Register snapshot          │
│    - Add to lookup table                │
│    - Initialize ref count = 1           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 7. Optional: Persist to disk            │
│    - Serialize snapshot                 │
│    - Write to storage                   │
│    - Async operation                    │
└──────────────┬──────────────────────────┘
               │
               ▼
        Return snapshot_id
```

### Snapshot Restore Flow

```
User Request
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. Frontend: s.restore_snapshot(id)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. SnapshotManager.restore_snapshot()   │
│    - Validate snapshot exists           │
│    - Check compatibility                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Registry: Retrieve snapshot          │
│    - Lookup by ID                       │
│    - Increment ref count                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Memory Pool: Copy SSM state          │
│    - Copy from snapshot indices         │
│    - Allocate new request state         │
│    - OR reuse if COW enabled            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. Radix Cache: Update request node     │
│    - Point to snapshot's cache node     │
│    - Lock new node                      │
│    - Unlock old node                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 6. Request State: Update                │
│    - Set token sequence                 │
│    - Set sequence length                │
│    - Update state pointers              │
└──────────────┬──────────────────────────┘
               │
               ▼
        Restore complete
```

## Data Structures

### Snapshot Memory Layout

Each snapshot contains references to memory regions:

```
Snapshot Object (CPU)
├── snapshot_id: str (UUID)
├── mamba_state_indices: Tensor (GPU pointers)
│   └── [idx0, idx1, ...] → Points to memory pool
├── token_ids: List[int] (CPU)
├── seq_len: int
├── last_node: TreeNode reference
├── metadata: dict
└── ref_count: int

Memory Pool (GPU)
├── Mamba States
│   ├── Conv States: [batch, d_inner, d_conv]
│   └── SSM States: [batch, d_inner, d_state]
│
└── Reference Counts
    └── Each state has ref_count tracking
```

### Radix Cache Integration

The snapshot system extends the existing radix cache node structure:

```python
# Existing TreeNode (in mamba_radix_cache.py)
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None           # Full KV indices
        self.mamba_value = None     # Mamba state indices
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # NEW: Snapshot references
        self.snapshot_refs = set()  # Set of snapshot IDs referencing this node
```

When a snapshot is created:
1. Current node's `snapshot_refs` is updated
2. `mamba_lock_ref` is incremented
3. Node cannot be evicted while snapshots reference it

When a snapshot is deleted:
1. Node's `snapshot_refs` is updated
2. `mamba_lock_ref` is decremented
3. Node becomes evictable if no other references exist

## Memory Management

### Reference Counting Strategy

The snapshot system uses **dual reference counting**:

1. **Memory Pool Level**: Tracks SSM state tensor references
2. **Registry Level**: Tracks snapshot object references

```python
# Example: Save snapshot increases both counters
snapshot = manager.save_snapshot(req, "snap_1")
# - Memory pool ref count for SSM states: +1
# - Registry ref count for snapshot: +1

# Restore snapshot increases registry count only
manager.restore_snapshot("snap_1", new_req)
# - Memory pool: New allocation OR COW (no change to snapshot's refs)
# - Registry ref count: +1 (another active reference)

# Delete snapshot decreases registry count
manager.delete_snapshot("snap_1")
# - Registry ref count: -1
# - If ref count reaches 0:
#   - Memory pool refs: -1
#   - Remove from registry
```

### Copy-on-Write (COW) Optimization

To minimize memory usage, snapshots support COW:

```python
# Without COW (default)
restore_snapshot("snap_1")
# → Allocates new SSM state memory
# → Copies data from snapshot
# → Modifications don't affect snapshot

# With COW (enabled via config)
restore_snapshot("snap_1")
# → Shares SSM state memory with snapshot
# → Uses reference, not copy
# → First modification triggers copy (lazy)
```

### Memory Pressure Handling

When GPU memory is low:

1. **Eviction Priority**:
   - Unlocked radix cache nodes (lowest priority)
   - Temporary request states
   - **Never evict**: Locked nodes, snapshot-referenced states

2. **Automatic Management**:
   ```python
   # Snapshot manager monitors memory
   if memory_pressure_high():
       # Evict LRU unlocked nodes first
       radix_cache.evict_full(num_tokens)
       # If still insufficient, fail gracefully
       raise OutOfMemoryError("Cannot create snapshot")
   ```

3. **User Control**:
   ```python
   # Manual cleanup
   manager.delete_snapshot("old_snap")

   # Batch cleanup
   manager.cleanup_snapshots(older_than=3600)  # 1 hour
   ```

## Concurrency Model

### Thread Safety

All snapshot operations are thread-safe through:

1. **Manager-level locking**:
   ```python
   class SnapshotManager:
       def __init__(self):
           self.lock = threading.RLock()  # Reentrant lock

       def save_snapshot(self, ...):
           with self.lock:
               # Atomic snapshot creation
               ...
   ```

2. **Registry-level locking**:
   ```python
   class SnapshotRegistry:
       def __init__(self):
           self.lock = threading.RLock()

       def register(self, snapshot):
           with self.lock:
               # Atomic registration
               ...
   ```

3. **Radix cache locking** (existing):
   - `full_lock_ref` and `mamba_lock_ref` are thread-safe counters
   - LRU list operations are protected

### Multi-Request Concurrency

Multiple requests can use snapshots concurrently:

```python
# Request 1: Save snapshot
req1.save_snapshot("checkpoint_1")

# Request 2: Restore from same snapshot (concurrent)
req2.restore_snapshot("checkpoint_1")

# Request 3: Restore from same snapshot (concurrent)
req3.restore_snapshot("checkpoint_1")

# All operations are thread-safe
# Each request gets independent state (with COW optimization)
```

### Async Disk I/O

Disk operations are asynchronous to avoid blocking:

```python
class SnapshotSerializer:
    def __init__(self):
        self.io_executor = ThreadPoolExecutor(max_workers=4)

    def persist_async(self, snapshot, path):
        future = self.io_executor.submit(self._write_to_disk, snapshot, path)
        return future  # Non-blocking
```

## Integration Points

### 1. Frontend API Integration

**Location**: `python/sglang/lang/ir.py`

```python
class SglContext:
    def save_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """Save current state as a snapshot."""
        if snapshot_id is None:
            snapshot_id = str(uuid.uuid4())
        self._snapshot_manager.save_snapshot(self._req, snapshot_id)
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restore state from a snapshot."""
        self._snapshot_manager.restore_snapshot(snapshot_id, self._req)
```

### 2. Request Object Integration

**Location**: `python/sglang/srt/managers/schedule_batch.py`

```python
class Req:
    def __init__(self, ...):
        # Existing fields
        self.mamba_pool_idx = None
        self.last_node = None

        # NEW: Snapshot-related fields
        self.active_snapshot = None     # Currently restored snapshot
        self.derived_from_snapshot = None  # Parent snapshot (for tracking)
```

### 3. Scheduler Integration

**Location**: `python/sglang/srt/managers/scheduler.py`

The scheduler is aware of snapshot constraints:

```python
class Scheduler:
    def can_schedule_request(self, req):
        if req.active_snapshot:
            # Ensure snapshot's cache node is still valid
            if not self._validate_snapshot_state(req.active_snapshot):
                raise SnapshotInvalidError(...)
        return True
```

## Design Decisions

### Why Reference-Based Snapshots?

**Decision**: Store references to memory pool indices, not copies

**Rationale**:
- **Memory Efficiency**: Avoid duplicating large SSM states
- **Fast Save**: O(1) save operation, just increment ref counts
- **Flexible Restore**: Can choose copy or COW on restore
- **Integration**: Works seamlessly with existing memory pool

**Trade-off**: Requires careful reference counting and memory management

### Why Immutable Snapshots?

**Decision**: Snapshots are immutable after creation

**Rationale**:
- **Safety**: Prevents accidental modification
- **Concurrency**: Multiple readers without locks
- **Predictability**: Snapshot always represents same state
- **Simplicity**: Easier to reason about lifecycle

**Trade-off**: Cannot update metadata after creation (must delete and recreate)

### Why Opt-in by Default?

**Decision**: Snapshots disabled unless explicitly enabled

**Rationale**:
- **Backward Compatibility**: Existing code unaffected
- **Zero Overhead**: No performance impact when unused
- **Memory Control**: Users opt into additional memory usage
- **Gradual Adoption**: Can be enabled per-request or globally

### Why Separate from Radix Cache?

**Decision**: Snapshot system is a separate component that uses radix cache

**Rationale**:
- **Separation of Concerns**: Radix cache handles prefix matching, snapshots handle state persistence
- **Modularity**: Can be developed and tested independently
- **Flexibility**: Could potentially support non-Mamba models in future
- **Clarity**: Clear ownership of responsibilities

**Trade-off**: Slightly more complex integration code

## Performance Characteristics

### Time Complexity

- **Save Snapshot**: O(1) - Just reference counting
- **Restore Snapshot**: O(S) where S = SSM state size (for copy) or O(1) (for COW)
- **Delete Snapshot**: O(1) - Decrement refs, possible dealloc
- **Lookup Snapshot**: O(1) - Hash table lookup

### Space Complexity

- **Per Snapshot**: O(S + T) where S = SSM state size, T = token sequence length
- **Registry Overhead**: O(N * M) where N = snapshots, M = metadata size (typically small)
- **Radix Cache**: Unchanged when snapshots disabled, O(N) additional refs when enabled

### Memory Overhead

For a typical Mamba-2.8B model:
- SSM state size: ~50 MB per snapshot
- Metadata: <1 KB per snapshot
- Registry: <10 KB total
- **Total overhead**: ~50 MB per active snapshot

## Future Enhancements

### Planned Improvements

1. **Distributed Snapshots**: Support for multi-GPU/multi-node snapshots
2. **Compression**: Compress snapshots when persisted to disk
3. **Incremental Snapshots**: Store only deltas from parent snapshot
4. **Snapshot Metadata Queries**: Search and filter snapshots by metadata
5. **Automatic Cleanup**: LRU-based automatic snapshot eviction

### Research Directions

1. **Cross-Model Snapshots**: Transfer states between similar models
2. **Snapshot Diffing**: Analyze differences between snapshots
3. **State Interpolation**: Blend multiple snapshots
4. **Snapshot Versioning**: Track lineage and relationships

## Conclusion

The snapshot system is designed to be:
- **Efficient**: Minimal overhead, reference-based
- **Safe**: Thread-safe, immutable, reference-counted
- **Flexible**: COW, disk persistence, metadata
- **Compatible**: Zero impact on existing code
- **Extensible**: Foundation for future enhancements

For implementation details, see the [API Reference](api_reference.md).
