"""
Snapshot retention and lifecycle management policies.

This module provides configurable policies for:
- When to take snapshots
- How many to retain
- When to prune old snapshots
- Branch creation and management
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SnapshotTriggerPolicy(Enum):
    """
    Policy for when to trigger snapshots.

    EVERY_TURN: Snapshot after every turn (most storage, best continuity)
    EVERY_N_TURNS: Snapshot every N turns (configurable)
    ON_TOOL_CALL: Snapshot only after tool executions
    MANUAL_ONLY: Only snapshot on explicit API calls
    """

    EVERY_TURN = "every_turn"
    EVERY_N_TURNS = "every_n_turns"
    ON_TOOL_CALL = "on_tool_call"
    MANUAL_ONLY = "manual_only"


@dataclass
class SnapshotRetentionConfig:
    """
    Configuration for snapshot retention policy.

    Attributes:
        max_snapshots_per_conversation: Maximum snapshots to keep per conversation
        snapshot_trigger_policy: When to trigger snapshots
        snapshot_every_n_turns: If EVERY_N_TURNS, snapshot every N turns
        enable_auto_pruning: Automatically prune old snapshots
        min_snapshot_interval_seconds: Minimum time between snapshots (prevents spam)
        keep_named_branches: Always keep named branches (don't prune)
    """

    max_snapshots_per_conversation: int = 10
    snapshot_trigger_policy: SnapshotTriggerPolicy = SnapshotTriggerPolicy.EVERY_TURN
    snapshot_every_n_turns: int = 5
    enable_auto_pruning: bool = True
    min_snapshot_interval_seconds: float = 1.0  # Prevent snapshot spam
    keep_named_branches: bool = True


class SnapshotRetentionPolicy:
    """
    Manages snapshot lifecycle and retention.

    This class decides:
    - Whether to take a snapshot (based on trigger policy)
    - When to prune old snapshots
    - How to manage named branches

    Thread Safety:
        This class is NOT thread-safe. Caller must ensure exclusive access.

    Usage:
        config = SnapshotRetentionConfig(max_snapshots_per_conversation=5)
        policy = SnapshotRetentionPolicy(snapshot_manager, config)

        # In post-forward hook
        if policy.should_snapshot(req, turn_number):
            # Take snapshot
            policy.mark_snapshot_taken(conversation_id)
            policy.prune_old_snapshots(conversation_id)
    """

    def __init__(
        self,
        snapshot_manager,  # MambaSnapshotManager instance
        config: SnapshotRetentionConfig,
    ):
        """
        Initialize retention policy.

        Args:
            snapshot_manager: MambaSnapshotManager instance
            config: Retention configuration
        """
        self.snapshot_manager = snapshot_manager
        self.config = config
        self._last_snapshot_time = {}  # conversation_id -> timestamp

        logger.info(
            f"SnapshotRetentionPolicy initialized: "
            f"max={config.max_snapshots_per_conversation}, "
            f"policy={config.snapshot_trigger_policy.value}"
        )

    def should_snapshot(
        self,
        req: Any,
        turn_number: int,
        conversation_id: Optional[str] = None,
        additional_context: Optional[dict] = None,
    ) -> bool:
        """
        Determine if a snapshot should be taken.

        Args:
            req: Request object
            turn_number: Current turn number
            conversation_id: Conversation ID (or derived from req)
            additional_context: Additional context (e.g., tool_call_made)

        Returns:
            True if snapshot should be taken, False otherwise
        """
        if conversation_id is None:
            # Derive from req if available
            conversation_id = getattr(req, "conversation_id", None)
            if conversation_id is None:
                # Fallback to rid
                conversation_id = getattr(req, "rid", "default")

        # Check minimum interval
        last_time = self._last_snapshot_time.get(conversation_id, 0)
        current_time = time.time()

        if (current_time - last_time) < self.config.min_snapshot_interval_seconds:
            logger.debug(
                f"Skipping snapshot for {conversation_id}: "
                f"too soon (last={last_time:.2f}s ago)"
            )
            return False

        # Apply trigger policy
        policy = self.config.snapshot_trigger_policy

        if policy == SnapshotTriggerPolicy.MANUAL_ONLY:
            return False

        elif policy == SnapshotTriggerPolicy.EVERY_TURN:
            return True

        elif policy == SnapshotTriggerPolicy.EVERY_N_TURNS:
            n = self.config.snapshot_every_n_turns
            return (turn_number % n) == 0

        elif policy == SnapshotTriggerPolicy.ON_TOOL_CALL:
            # Check if tool call was made
            if additional_context and additional_context.get("tool_call_made", False):
                return True
            return False

        else:
            logger.warning(f"Unknown snapshot policy: {policy}")
            return False

    def mark_snapshot_taken(self, conversation_id: str) -> None:
        """
        Mark that a snapshot was just taken for a conversation.

        This updates the last snapshot time to enforce minimum interval.

        Args:
            conversation_id: Conversation ID
        """
        self._last_snapshot_time[conversation_id] = time.time()

    def prune_old_snapshots(self, conversation_id: str) -> int:
        """
        Prune old snapshots beyond retention limit.

        Keeps the most recent N snapshots according to max_snapshots_per_conversation.
        Named branches are optionally kept if keep_named_branches is True.

        Args:
            conversation_id: Conversation ID

        Returns:
            Number of snapshots deleted
        """
        if not self.config.enable_auto_pruning:
            return 0

        # Get all snapshots for conversation
        snapshots = self.snapshot_manager.list_snapshots(conversation_id)

        if len(snapshots) <= self.config.max_snapshots_per_conversation:
            return 0  # Under limit

        # Determine how many to delete
        to_delete = len(snapshots) - self.config.max_snapshots_per_conversation

        # Delete oldest snapshots
        deleted_count = 0
        for turn_number in sorted(snapshots)[:to_delete]:
            try:
                self.snapshot_manager.delete_snapshot(conversation_id, turn_number)
                deleted_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to delete snapshot {conversation_id}/{turn_number}: {e}"
                )

        if deleted_count > 0:
            logger.info(
                f"Pruned {deleted_count} old snapshots for {conversation_id} "
                f"(kept {len(snapshots) - deleted_count})"
            )

        return deleted_count

    def create_branch(
        self,
        conversation_id: str,
        branch_name: str,
        source_turn: Optional[int] = None,
    ) -> bool:
        """
        Create a named branch from a snapshot.

        This allows saving checkpoints that won't be pruned by retention policy.

        Args:
            conversation_id: Conversation ID
            branch_name: Name for the branch (e.g., "exploration_1")
            source_turn: Turn number to branch from (None = latest)

        Returns:
            True if branch created successfully, False otherwise
        """
        # Get source snapshot
        if source_turn is None:
            latest = self.snapshot_manager.get_latest_snapshot(conversation_id)
            if latest is None:
                logger.error(f"No snapshots found for {conversation_id}")
                return False
            source_turn, _ = latest

        # Load source snapshot
        try:
            conv_states, temporal_states, source_metadata = (
                self.snapshot_manager.load_snapshot(conversation_id, source_turn)
            )
        except Exception as e:
            logger.error(f"Failed to load source snapshot: {e}")
            return False

        # Create branch metadata
        from sglang.srt.snapshot.mamba_snapshot import MambaSnapshotMetadata

        branch_metadata = MambaSnapshotMetadata(
            conversation_id=conversation_id,
            turn_number=source_metadata.turn_number,
            timestamp=time.time(),
            token_count=source_metadata.token_count,
            model_name=source_metadata.model_name,
            mamba_pool_idx=source_metadata.mamba_pool_idx,
            req_pool_idx=source_metadata.req_pool_idx,
            layer_config=source_metadata.layer_config,
            branch_name=branch_name,
            parent_snapshot=f"turn_{source_turn}",
        )

        # Save branch
        try:
            self.snapshot_manager.save_snapshot(
                conv_states, temporal_states, branch_metadata
            )
            logger.info(
                f"Created branch '{branch_name}' from turn {source_turn} "
                f"for conversation {conversation_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save branch: {e}")
            return False

    def delete_branch(self, conversation_id: str, branch_name: str) -> bool:
        """
        Delete a named branch.

        Args:
            conversation_id: Conversation ID
            branch_name: Branch name to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.snapshot_manager.delete_snapshot(
                conversation_id, branch_name=branch_name
            )
            logger.info(f"Deleted branch '{branch_name}' for {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete branch: {e}")
            return False

    def list_branches(self, conversation_id: str) -> list:
        """
        List all branches for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of branch names
        """
        return self.snapshot_manager.list_branches(conversation_id)

    def get_total_snapshot_size(self, conversation_id: str) -> int:
        """
        Get total disk usage for all snapshots of a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Total size in bytes
        """
        total_size = 0

        # Add main snapshots
        for turn_number in self.snapshot_manager.list_snapshots(conversation_id):
            total_size += self.snapshot_manager.get_snapshot_size(
                conversation_id, turn_number
            )

        # Add branches
        for branch_name in self.snapshot_manager.list_branches(conversation_id):
            total_size += self.snapshot_manager.get_snapshot_size(
                conversation_id, branch_name=branch_name
            )

        return total_size

    def cleanup_all_snapshots(self, conversation_id: str) -> int:
        """
        Delete ALL snapshots for a conversation (including branches).

        WARNING: This is destructive and irreversible.

        Args:
            conversation_id: Conversation ID

        Returns:
            Number of items deleted
        """
        deleted = 0

        # Delete main snapshots
        for turn_number in self.snapshot_manager.list_snapshots(conversation_id):
            try:
                self.snapshot_manager.delete_snapshot(conversation_id, turn_number)
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete snapshot {turn_number}: {e}")

        # Delete branches
        for branch_name in self.snapshot_manager.list_branches(conversation_id):
            try:
                self.snapshot_manager.delete_snapshot(
                    conversation_id, branch_name=branch_name
                )
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete branch {branch_name}: {e}")

        logger.warning(f"Cleaned up {deleted} snapshots for {conversation_id}")
        return deleted
