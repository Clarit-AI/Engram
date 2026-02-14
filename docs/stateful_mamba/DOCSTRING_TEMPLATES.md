# Docstring Templates for Snapshot System Implementation

This document provides templates for docstrings that should be added to all snapshot-related code during implementation.

## Module-Level Docstrings

### `snapshot_manager.py`

```python
"""
Snapshot Manager for Mamba State Persistence.

This module provides the SnapshotManager class, which is the central coordinator
for all snapshot operations in SGLang. It handles snapshot lifecycle management,
coordination with memory pools and radix cache, and disk persistence.

The snapshot system is opt-in and maintains full backward compatibility with
existing SGLang workflows.

Example:
    >>> from sglang import Engine
    >>> from sglang.snapshot import SnapshotManager
    >>>
    >>> engine = Engine(
    ...     model_path="state-spaces/mamba-2.8b",
    ...     enable_mamba_snapshots=True
    ... )
    >>> manager = SnapshotManager(engine)
    >>>
    >>> # Create a snapshot
    >>> snapshot_id = manager.save_snapshot(req, "checkpoint_1")
    >>>
    >>> # Restore from snapshot
    >>> manager.restore_snapshot("checkpoint_1", new_req)
    >>>
    >>> # Cleanup
    >>> manager.delete_snapshot("checkpoint_1")

See Also:
    - User Guide: docs/stateful_mamba/user_guide.md
    - API Reference: docs/stateful_mamba/api_reference.md
    - Architecture: docs/stateful_mamba/architecture.md

Note:
    Snapshots are only supported for Mamba/Mamba2 architectures. Transformer
    models are unaffected by this feature.
"""
```

### `snapshot.py`

```python
"""
Snapshot Data Structures.

This module defines the Snapshot dataclass and related data structures for
representing saved Mamba model states.

A Snapshot is an immutable representation of the model's SSM (State Space Model)
state at a specific point in the token sequence. It stores references to GPU
memory rather than copies, making snapshot creation very fast (O(1)).

Example:
    >>> from sglang.snapshot import Snapshot
    >>>
    >>> # Snapshots are typically created by SnapshotManager
    >>> snapshot = Snapshot(
    ...     snapshot_id="checkpoint_1",
    ...     mamba_state_indices=torch.tensor([0, 1, 2]),
    ...     token_ids=[1, 2, 3, 4, 5],
    ...     seq_len=5,
    ...     last_node=radix_cache_node,
    ...     metadata={"turn": 1, "important": True},
    ...     timestamp=time.time()
    ... )
    >>>
    >>> # Access snapshot properties
    >>> print(f"Snapshot age: {snapshot.age:.2f} seconds")
    >>> print(f"Memory size: {snapshot.memory_size / 1e6:.2f} MB")

Note:
    Snapshots are immutable after creation. This ensures thread-safety and
    prevents accidental modifications.
"""
```

### `snapshot_registry.py`

```python
"""
Thread-Safe Snapshot Registry.

This module provides the SnapshotRegistry class, which maintains a thread-safe
registry of all active snapshots with reference counting for memory management.

The registry uses fine-grained locking to allow concurrent access while
preventing race conditions. It implements automatic cleanup when reference
counts reach zero.

Example:
    >>> from sglang.snapshot import SnapshotRegistry, Snapshot
    >>>
    >>> registry = SnapshotRegistry()
    >>>
    >>> # Register a snapshot
    >>> registry.register(snapshot)
    >>>
    >>> # Increment reference (snapshot is being used)
    >>> registry.inc_ref(snapshot.snapshot_id)
    >>>
    >>> # Decrement reference (done using snapshot)
    >>> registry.dec_ref(snapshot.snapshot_id)  # Auto-deletes if ref_count == 0
    >>>
    >>> # Manual deletion
    >>> registry.delete(snapshot.snapshot_id)

Thread Safety:
    All methods are thread-safe and can be called from multiple threads
    concurrently. The registry uses a reentrant lock (RLock) to prevent
    deadlocks in nested calls.
"""
```

### `snapshot_serializer.py`

```python
"""
Snapshot Serialization for Disk Persistence.

This module provides the SnapshotSerializer class for saving snapshots to disk
and loading them back. It supports multiple serialization formats and optional
compression.

Supported formats:
    - safetensors (default): Safe, portable, efficient
    - pickle: Python native, debugging-friendly
    - custom_binary: Maximum efficiency, minimal overhead

Example:
    >>> from sglang.snapshot import SnapshotSerializer, SnapshotConfig
    >>>
    >>> config = SnapshotConfig(
    ...     serialization_format="safetensors",
    ...     compression_enabled=True
    ... )
    >>> serializer = SnapshotSerializer(config)
    >>>
    >>> # Save to disk
    >>> serializer.serialize(snapshot, "./snapshots/checkpoint_1.bin")
    >>>
    >>> # Load from disk
    >>> loaded_snapshot = serializer.deserialize("./snapshots/checkpoint_1.bin")

Performance:
    - safetensors: ~100 MB/s write, ~200 MB/s read
    - pickle: ~50 MB/s write, ~80 MB/s read
    - custom_binary: ~200 MB/s write, ~400 MB/s read

    Compression can reduce file size by 50-70% with minimal performance impact.
"""
```

## Class Docstrings

### `SnapshotManager`

```python
class SnapshotManager:
    """
    Central coordinator for snapshot lifecycle management.

    The SnapshotManager handles all snapshot operations including creation,
    restoration, persistence, and cleanup. It coordinates with the memory pool
    and radix cache to ensure consistent state.

    This class is thread-safe and can be used from multiple threads concurrently.

    Attributes:
        registry (SnapshotRegistry): Thread-safe snapshot registry.
        mem_pool (HybridReqToTokenPool): Memory pool for SSM states.
        radix_cache (MambaRadixCache): Radix cache for prefix matching.
        serializer (SnapshotSerializer): Handles disk I/O operations.

    Example:
        >>> manager = SnapshotManager(engine)
        >>>
        >>> # Save a snapshot
        >>> snapshot_id = manager.save_snapshot(req, "checkpoint_1")
        >>>
        >>> # Restore from snapshot
        >>> manager.restore_snapshot("checkpoint_1", new_req)
        >>>
        >>> # List all snapshots
        >>> snapshots = manager.list_snapshots()
        >>>
        >>> # Cleanup
        >>> manager.delete_snapshot("checkpoint_1")

    Thread Safety:
        All public methods are thread-safe. Internal state is protected by locks.

    See Also:
        - SnapshotRegistry: Thread-safe snapshot storage
        - SnapshotSerializer: Disk persistence
        - Snapshot: Snapshot data structure

    Note:
        The SnapshotManager is typically accessed through the frontend API
        (e.g., s.save_snapshot()) rather than directly.
    """
```

### `Snapshot`

```python
@dataclass(frozen=True)
class Snapshot:
    """
    Immutable snapshot of Mamba SSM state.

    A Snapshot represents the complete state of a Mamba model at a specific
    point in the token sequence. It stores references to GPU memory (not copies),
    making snapshot creation very fast.

    Attributes:
        snapshot_id (str): Unique identifier for this snapshot.
        mamba_state_indices (torch.Tensor): Indices into memory pool for SSM states.
        token_ids (List[int]): Token sequence up to this snapshot point.
        seq_len (int): Total sequence length.
        last_node (TreeNode): Radix cache node reference.
        metadata (Dict[str, Any]): User-provided metadata.
        timestamp (float): Creation time (Unix timestamp).
        ref_count (int): Reference count for memory management.

    Properties:
        age (float): Age of snapshot in seconds.
        memory_size (int): Estimated memory size in bytes.
        token_text (str): Decoded token text (if tokenizer available).

    Example:
        >>> snapshot = manager.get_snapshot("checkpoint_1")
        >>> print(f"ID: {snapshot.snapshot_id}")
        >>> print(f"Tokens: {len(snapshot.token_ids)}")
        >>> print(f"Age: {snapshot.age:.2f}s")
        >>> print(f"Size: {snapshot.memory_size / 1e6:.2f} MB")
        >>> print(f"Metadata: {snapshot.metadata}")

    Note:
        Snapshots are immutable (frozen=True). This ensures thread-safety and
        prevents accidental modifications. The ref_count field is managed by
        the registry and should not be modified directly.

    Invariants:
        - snapshot_id is unique and non-empty
        - mamba_state_indices references valid memory pool indices
        - token_ids matches seq_len
        - ref_count >= 0
        - Snapshot is frozen after creation
    """
```

## Method Docstrings

### `SnapshotManager.save_snapshot()`

```python
def save_snapshot(
    self,
    req: Req,
    snapshot_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Snapshot:
    """
    Save the current request state as a snapshot.

    This method creates a snapshot of the request's current Mamba SSM state.
    The operation is very fast (O(1)) because it stores references to memory
    pool indices rather than copying data.

    Args:
        req (Req): The request to snapshot. Must be a Mamba model request
            with valid SSM state.
        snapshot_id (str, optional): Custom snapshot ID. If None, auto-generates
            a UUID. Must be unique.
        metadata (dict, optional): User-defined metadata to attach to the
            snapshot. Can contain any JSON-serializable data.

    Returns:
        Snapshot: The created snapshot object.

    Raises:
        SnapshotDisabledError: If snapshots are not enabled.
        ValueError: If snapshot_id already exists.
        OutOfMemoryError: If insufficient GPU memory for snapshot.
        TypeError: If req is not a Mamba model request.

    Example:
        >>> # Auto-generated ID
        >>> snapshot = manager.save_snapshot(req)
        >>> print(snapshot.snapshot_id)  # "550e8400-e29b-41d4-a716-446655440000"
        >>>
        >>> # Custom ID with metadata
        >>> snapshot = manager.save_snapshot(
        ...     req,
        ...     snapshot_id="checkpoint_1",
        ...     metadata={"turn": 1, "important": True}
        ... )

    Performance:
        - Time: O(1), typically <1ms
        - Memory: References only, no copying
        - Locks: Brief lock on registry and radix cache

    Thread Safety:
        This method is thread-safe. Multiple threads can create snapshots
        concurrently.

    See Also:
        - restore_snapshot(): Restore from a snapshot
        - delete_snapshot(): Delete a snapshot
        - Frontend API: s.save_snapshot()

    Note:
        The snapshot stores references to SSM states, not copies. The actual
        memory is protected by incrementing reference counts in the memory pool
        and radix cache, preventing eviction until the snapshot is deleted.
    """
```

### `SnapshotManager.restore_snapshot()`

```python
def restore_snapshot(
    self,
    snapshot_id: str,
    req: Req,
    cow: bool = None
) -> None:
    """
    Restore request state from a snapshot.

    This method restores the request's Mamba SSM state from a previously saved
    snapshot. It updates the request's token sequence, SSM states, and radix
    cache node to match the snapshot.

    Args:
        snapshot_id (str): ID of the snapshot to restore.
        req (Req): The request to restore state into. Must be a Mamba model
            request.
        cow (bool, optional): Enable copy-on-write. If None, uses global config.
            With COW enabled, memory is shared until first modification (lazy copy).
            Without COW, memory is copied immediately.

    Returns:
        None: Modifies req in-place.

    Raises:
        SnapshotNotFoundError: If snapshot_id doesn't exist.
        SnapshotInvalidError: If snapshot state is corrupted.
        TypeError: If req is not compatible with snapshot.
        OutOfMemoryError: If insufficient memory for restoration (when COW=False).

    Example:
        >>> # Basic restore
        >>> manager.restore_snapshot("checkpoint_1", req)
        >>>
        >>> # Restore with explicit COW setting
        >>> manager.restore_snapshot("checkpoint_1", req, cow=True)
        >>>
        >>> # After restore, req has snapshot's state
        >>> assert req.seq_len == snapshot.seq_len

    Performance:
        - With COW: O(1), <1ms (shares memory)
        - Without COW: O(S) where S = SSM state size, ~10-50ms (copies memory)

    Thread Safety:
        This method is thread-safe. The same snapshot can be restored by
        multiple threads concurrently.

    Memory Management:
        - Increments snapshot's reference count
        - Decrements req's previous state reference count
        - With COW: Shares memory until modification
        - Without COW: Allocates new memory immediately

    See Also:
        - save_snapshot(): Create a snapshot
        - delete_snapshot(): Delete a snapshot
        - Frontend API: s.restore_snapshot()

    Note:
        Restoring a snapshot does not modify the snapshot itself. The snapshot
        remains valid and can be restored multiple times.
    """
```

### `SnapshotManager.delete_snapshot()`

```python
def delete_snapshot(self, snapshot_id: str) -> bool:
    """
    Delete a snapshot and free its resources.

    This method removes a snapshot from the registry and decrements reference
    counts in the memory pool and radix cache. If no other references exist,
    the associated memory is freed.

    Args:
        snapshot_id (str): ID of the snapshot to delete.

    Returns:
        bool: True if snapshot was deleted, False if not found.

    Raises:
        SnapshotInUseError: If snapshot is currently being used by active
            requests (ref_count > 1).

    Example:
        >>> # Delete a snapshot
        >>> deleted = manager.delete_snapshot("checkpoint_1")
        >>> if deleted:
        ...     print("Snapshot deleted successfully")
        ... else:
        ...     print("Snapshot not found")
        >>>
        >>> # Handle in-use snapshots
        >>> try:
        ...     manager.delete_snapshot("in_use_snapshot")
        ... except SnapshotInUseError:
        ...     print("Snapshot is in use, cannot delete")

    Performance:
        - Time: O(1), <1ms
        - Memory: Frees snapshot memory if no other references

    Thread Safety:
        This method is thread-safe. However, it will fail if the snapshot
        is currently being used by another thread.

    Memory Management:
        - Decrements memory pool reference counts
        - Decrements radix cache node reference counts
        - Frees memory if ref_count reaches 0
        - Registry entry is removed immediately

    See Also:
        - delete_snapshots_before(): Delete old snapshots
        - clear_all_snapshots(): Delete all snapshots
        - Frontend API: s.delete_snapshot()

    Note:
        Deletion is permanent and cannot be undone (unless the snapshot was
        persisted to disk). Be cautious when deleting snapshots that may be
        needed later.
    """
```

### `SnapshotManager.persist_snapshot()`

```python
def persist_snapshot(
    self,
    snapshot_id: str,
    path: str,
    format: str = "safetensors",
    compress: bool = True,
    async_mode: bool = True
) -> Optional[Future]:
    """
    Save a snapshot to disk for long-term storage.

    This method serializes a snapshot to disk, allowing it to be loaded later
    even after the engine is restarted. The snapshot remains in memory after
    persistence unless explicitly deleted.

    Args:
        snapshot_id (str): ID of the snapshot to persist.
        path (str): Destination file path. Parent directory must exist.
        format (str, optional): Serialization format. Options:
            - "safetensors" (default): Safe, portable, efficient
            - "pickle": Python native, debugging-friendly
            - "custom": Maximum efficiency, minimal overhead
        compress (bool, optional): Enable compression. Default: True.
            Reduces file size by 50-70% with minimal performance impact.
        async_mode (bool, optional): Perform I/O asynchronously. Default: True.
            With async=True, returns immediately and I/O happens in background.
            With async=False, blocks until I/O completes.

    Returns:
        Optional[Future]: If async_mode=True, returns a Future that completes
            when I/O finishes. If async_mode=False, returns None.

    Raises:
        SnapshotNotFoundError: If snapshot_id doesn't exist.
        IOError: If disk write fails.
        OSError: If parent directory doesn't exist or insufficient permissions.

    Example:
        >>> # Async persist (non-blocking)
        >>> future = manager.persist_snapshot(
        ...     "checkpoint_1",
        ...     "./snapshots/checkpoint_1.bin",
        ...     async_mode=True
        ... )
        >>> # Do other work...
        >>> future.result()  # Wait for completion
        >>>
        >>> # Sync persist (blocking)
        >>> manager.persist_snapshot(
        ...     "checkpoint_2",
        ...     "./snapshots/checkpoint_2.bin",
        ...     async_mode=False
        ... )
        >>>
        >>> # Custom format and compression
        >>> manager.persist_snapshot(
        ...     "checkpoint_3",
        ...     "./snapshots/checkpoint_3.bin",
        ...     format="custom",
        ...     compress=False
        ... )

    Performance:
        - safetensors: ~100 MB/s write
        - pickle: ~50 MB/s write
        - custom: ~200 MB/s write
        - Compression overhead: ~10-20% slower write, 50-70% smaller files

    File Format:
        The saved file contains:
        - Snapshot metadata (snapshot_id, token_ids, seq_len, metadata)
        - SSM state tensors (conv states, SSM states)
        - Radix cache node information
        - Format version for compatibility checking

    Thread Safety:
        This method is thread-safe. Multiple threads can persist different
        snapshots concurrently.

    See Also:
        - load_snapshot(): Load a snapshot from disk
        - SnapshotSerializer: Serialization implementation

    Note:
        Persisted snapshots can be loaded on any SGLang instance with the same
        model architecture. They are portable across machines but not across
        different model architectures.
    """
```

### `SnapshotManager.load_snapshot()`

```python
def load_snapshot(self, path: str) -> str:
    """
    Load a snapshot from disk.

    This method deserializes a snapshot file and registers it in memory,
    making it available for restoration. The snapshot file format is
    auto-detected.

    Args:
        path (str): Path to the saved snapshot file.

    Returns:
        str: The loaded snapshot's ID.

    Raises:
        FileNotFoundError: If path doesn't exist.
        SnapshotDeserializationError: If file is corrupted or incompatible.
        ValueError: If snapshot_id from file already exists in registry.
        OutOfMemoryError: If insufficient GPU memory for snapshot.

    Example:
        >>> # Load snapshot
        >>> snapshot_id = manager.load_snapshot("./snapshots/checkpoint_1.bin")
        >>> print(f"Loaded snapshot: {snapshot_id}")
        >>>
        >>> # Now can restore from it
        >>> manager.restore_snapshot(snapshot_id, req)
        >>>
        >>> # Check loaded snapshot
        >>> snapshot = manager.get_snapshot(snapshot_id)
        >>> print(f"Tokens: {len(snapshot.token_ids)}")

    Performance:
        - safetensors: ~200 MB/s read
        - pickle: ~80 MB/s read
        - custom: ~400 MB/s read

    Compatibility:
        The loaded snapshot must be compatible with the current model:
        - Same model architecture (Mamba/Mamba2)
        - Same hidden size
        - Same number of layers
        - Compatible SGLang version

    Thread Safety:
        This method is thread-safe. However, if the snapshot_id from the file
        already exists, a ValueError will be raised.

    Memory Management:
        - Allocates GPU memory for SSM states
        - Allocates memory pool indices
        - Registers in radix cache
        - Initial ref_count = 1 (registry reference)

    See Also:
        - persist_snapshot(): Save a snapshot to disk
        - SnapshotSerializer: Deserialization implementation

    Note:
        Loaded snapshots consume GPU memory immediately. Consider deleting
        them when no longer needed.
    """
```

## Property Docstrings

### `Snapshot.age`

```python
@property
def age(self) -> float:
    """
    Get snapshot age in seconds.

    Returns:
        float: Time in seconds since snapshot was created.

    Example:
        >>> snapshot = manager.get_snapshot("checkpoint_1")
        >>> print(f"Snapshot age: {snapshot.age:.2f} seconds")
        >>> if snapshot.age > 3600:
        ...     print("Snapshot is over 1 hour old")
    """
```

### `Snapshot.memory_size`

```python
@property
def memory_size(self) -> int:
    """
    Estimate memory size in bytes.

    Returns the approximate GPU memory consumed by this snapshot, including
    SSM states and metadata. This is an estimate based on tensor dimensions.

    Returns:
        int: Memory size in bytes.

    Example:
        >>> snapshot = manager.get_snapshot("checkpoint_1")
        >>> size_mb = snapshot.memory_size / 1e6
        >>> print(f"Snapshot size: {size_mb:.2f} MB")
        >>>
        >>> # Check total memory usage
        >>> snapshots = manager.list_snapshots()
        >>> total_mb = sum(s.memory_size for s in snapshots) / 1e6
        >>> print(f"Total snapshot memory: {total_mb:.2f} MB")

    Note:
        This is an estimate. Actual memory usage may vary slightly due to
        memory alignment and internal bookkeeping.
    """
```

## Exception Docstrings

### `SnapshotError`

```python
class SnapshotError(Exception):
    """
    Base exception for all snapshot-related errors.

    All snapshot-specific exceptions inherit from this class, making it easy
    to catch any snapshot-related error with a single except clause.

    Example:
        >>> try:
        ...     manager.save_snapshot(req)
        ... except SnapshotError as e:
        ...     print(f"Snapshot operation failed: {e}")
    """
```

### `SnapshotNotFoundError`

```python
class SnapshotNotFoundError(SnapshotError):
    """
    Raised when a snapshot ID is not found in the registry.

    This typically occurs when:
    - Restoring a snapshot that was never created
    - Accessing a snapshot that was already deleted
    - Using an incorrect snapshot ID

    Example:
        >>> try:
        ...     manager.restore_snapshot("nonexistent_id", req)
        ... except SnapshotNotFoundError:
        ...     print("Snapshot not found, cannot restore")
        ...     # Fallback to default behavior
    """
```

## Usage Examples in Docstrings

### Frontend API Integration

```python
# In python/sglang/lang/ir.py

class SglContext:
    def save_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save the current state as a snapshot.

        This method saves the current Mamba SSM state, allowing you to restore
        to this exact point later. It's useful for multi-turn conversations,
        branching scenarios, and checkpoint-based generation.

        Args:
            snapshot_id (str, optional): Custom snapshot ID. If None, auto-generates
                a UUID.
            metadata (dict, optional): User-defined metadata (JSON-serializable).

        Returns:
            str: The snapshot ID.

        Raises:
            SnapshotDisabledError: If snapshots are not enabled.
            OutOfMemoryError: If insufficient GPU memory.

        Example:
            >>> from sglang.lang import function, gen
            >>>
            >>> @function
            >>> def conversation(s):
            >>>     s += "User: Hello\n"
            >>>     s += "Assistant: " + gen("response1", max_tokens=100)
            >>>
            >>>     # Save snapshot after first exchange
            >>>     snapshot_id = s.save_snapshot(
            >>>         metadata={"turn": 1, "timestamp": time.time()}
            >>>     )
            >>>
            >>>     s += "\nUser: Tell me more\n"
            >>>     s += "Assistant: " + gen("response2", max_tokens=100)
            >>>
            >>>     return s, snapshot_id

        See Also:
            - restore_snapshot(): Restore from a snapshot
            - delete_snapshot(): Delete a snapshot
            - User Guide: docs/stateful_mamba/user_guide.md

        Note:
            Snapshots are only supported for Mamba/Mamba2 models. For transformer
            models, this method will raise SnapshotDisabledError even if snapshots
            are enabled globally.
        """
```

## Best Practices for Docstrings

1. **Always include**:
   - Brief one-line summary
   - Args with types and descriptions
   - Returns with type
   - Raises with conditions
   - At least one usage example
   - See Also references

2. **Performance notes**:
   - Time complexity
   - Memory impact
   - Typical execution time

3. **Thread safety**:
   - Whether method is thread-safe
   - Any locking considerations

4. **Examples**:
   - Show typical usage
   - Include edge cases
   - Use realistic scenarios

5. **Cross-references**:
   - Link to related methods
   - Point to documentation
   - Reference similar functionality

6. **Notes and warnings**:
   - Important invariants
   - Common pitfalls
   - Backward compatibility

## Documentation Review Checklist

When implementing code with these docstrings:

- [ ] All public classes have docstrings
- [ ] All public methods have docstrings
- [ ] All parameters are documented
- [ ] Return types are documented
- [ ] Exceptions are documented
- [ ] Examples are provided and tested
- [ ] Cross-references are valid
- [ ] Performance characteristics noted
- [ ] Thread-safety documented
- [ ] Docstrings follow Google style
- [ ] Code examples are runnable
- [ ] Types are accurate

---

**Last Updated**: 2026-02-14
**For**: SGLang Stateful Mamba Phase 2 Implementation
