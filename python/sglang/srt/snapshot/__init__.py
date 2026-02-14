"""
State snapshot and persistence system for Mamba/Hybrid models.

This module provides functionality to serialize and restore Mamba SSM states
to/from disk, enabling stateful inference that persists across server restarts.

**Backward Compatibility**: This module is entirely opt-in. Standard transformer-based
inference is not affected. Snapshot features only activate when explicitly enabled
via `--enable-snapshot-persistence` flag.

Key Components:
    - MambaSnapshotManager: Core snapshot save/load functionality
    - MambaSnapshotMetadata: Metadata tracking for snapshots
    - SnapshotHookManager: Integration hooks for scheduler
    - SnapshotRetentionPolicy: Snapshot lifecycle management
"""

from sglang.srt.snapshot.mamba_snapshot import (
    MambaSnapshotManager,
    MambaSnapshotMetadata,
)

try:
    from sglang.srt.snapshot.snapshot_hooks import SnapshotHookManager
    from sglang.srt.snapshot.snapshot_policy import SnapshotRetentionPolicy
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    "MambaSnapshotManager",
    "MambaSnapshotMetadata",
    "SnapshotHookManager",
    "SnapshotRetentionPolicy",
]
