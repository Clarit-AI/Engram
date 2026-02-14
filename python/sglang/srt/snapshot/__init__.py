"""
State snapshot and persistence system for Mamba/Hybrid models.

This module provides functionality to serialize and restore Mamba SSM states
to/from disk, enabling stateful inference that persists across server restarts.

**Phase 2**: Core snapshot serialization
**Phase 2.5**: 3-tier memory hierarchy (VRAM→Host RAM→Disk)

**Backward Compatibility**: This module is entirely opt-in. Standard transformer-based
inference is not affected. Snapshot features only activate when explicitly enabled
via `--enable-snapshot-persistence` flag.

Key Components:
    - MambaSnapshotManager: Core snapshot save/load functionality
    - MambaSnapshotMetadata: Metadata tracking for snapshots
    - SnapshotHookManager: Integration hooks for scheduler
    - SnapshotRetentionPolicy: Snapshot lifecycle management
    - MambaHostPool: Host memory tier (Phase 2.5)
    - ConversationTracker: Conversation state and tier tracking (Phase 2.5)
    - TierManager: 3-tier memory orchestration (Phase 2.5)
"""

from sglang.srt.snapshot.mamba_snapshot import (
    MambaSnapshotManager,
    MambaSnapshotMetadata,
)

try:
    from sglang.srt.snapshot.snapshot_hooks import SnapshotHookManager
    from sglang.srt.snapshot.snapshot_policy import SnapshotRetentionPolicy
    from sglang.srt.snapshot.mamba_host_pool import MambaHostPool, HostPoolEntry
    from sglang.srt.snapshot.conversation_tracker import (
        ConversationTracker,
        ConversationTier,
        ConversationState,
    )
    from sglang.srt.snapshot.tier_manager import TierManager
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    # Phase 2: Core snapshots
    "MambaSnapshotManager",
    "MambaSnapshotMetadata",
    "SnapshotHookManager",
    "SnapshotRetentionPolicy",
    # Phase 2.5: Memory tiers
    "MambaHostPool",
    "HostPoolEntry",
    "ConversationTracker",
    "ConversationTier",
    "ConversationState",
    "TierManager",
]
