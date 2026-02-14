"""
Tier manager for Mamba state memory hierarchy.

Orchestrates state transitions between VRAM, host RAM, and disk storage,
implementing a 3-tier memory system for optimal performance and capacity.
"""

import logging
import threading
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.snapshot.conversation_tracker import (
    ConversationState,
    ConversationTier,
    ConversationTracker,
)
from sglang.srt.snapshot.mamba_host_pool import MambaHostPool
from sglang.srt.snapshot.mamba_snapshot import (
    MambaSnapshotManager,
    MambaSnapshotMetadata,
)

logger = logging.getLogger(__name__)


class TierManager:
    """
    Manages 3-tier memory hierarchy for Mamba states.

    **Memory Tiers:**
    - Tier 1 (ACTIVE): VRAM (MambaPool) - Active inference, instant access
    - Tier 2 (WARM): Host RAM (MambaHostPool) - Fast restore (~10-50ms)
    - Tier 3 (COLD): Disk (Snapshots) - Long-term archive (~100-500ms)

    **Responsibilities:**
    - Automatic tier transitions based on access patterns
    - State promotion (COLD→WARM→ACTIVE) on access
    - State demotion (ACTIVE→WARM→COLD) on timeout
    - Cross-session reference loading
    - Background cleanup thread

    Thread Safety:
        All public methods are thread-safe.
    """

    def __init__(
        self,
        conversation_tracker: ConversationTracker,
        host_pool: MambaHostPool,
        snapshot_manager: MambaSnapshotManager,
        enable_background_cleanup: bool = True,
        cleanup_interval: float = 60.0,  # Check every 60 seconds
    ):
        """
        Initialize tier manager.

        Args:
            conversation_tracker: Conversation state tracker
            host_pool: Host memory pool
            snapshot_manager: Snapshot manager
            enable_background_cleanup: Run background cleanup thread
            cleanup_interval: Seconds between cleanup checks
        """
        self.conversation_tracker = conversation_tracker
        self.host_pool = host_pool
        self.snapshot_manager = snapshot_manager
        self.cleanup_interval = cleanup_interval

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_running = False

        if enable_background_cleanup:
            self.start_background_cleanup()

        logger.info("TierManager initialized")

    def save_to_warm_tier(
        self,
        conversation_id: str,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save state to Tier 2 (host RAM).

        This is typically called when state is evicted from VRAM.

        Args:
            conversation_id: Conversation identifier
            conv_states: Convolution state tensors
            temporal_states: Temporal state tensor
            metadata: Optional metadata

        Returns:
            True if saved successfully
        """
        with self._lock:
            # Save to host pool
            success = self.host_pool.save_state(
                conversation_id, conv_states, temporal_states, metadata
            )

            if success:
                # Update tracker
                self.conversation_tracker.transition_tier(
                    conversation_id, ConversationTier.WARM
                )

                logger.info(
                    f"Saved to WARM tier: {conversation_id}, "
                    f"host_pool_size={len(self.host_pool)}"
                )

            return success

    def save_to_cold_tier(
        self,
        conversation_id: str,
        conv_states: List[torch.Tensor],
        temporal_states: torch.Tensor,
        metadata: MambaSnapshotMetadata,
    ) -> bool:
        """
        Save state to Tier 3 (disk).

        This is called when state is evicted from host RAM or when
        explicitly archiving a conversation.

        Args:
            conversation_id: Conversation identifier
            conv_states: Convolution state tensors
            temporal_states: Temporal state tensor
            metadata: Snapshot metadata

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                # Save snapshot to disk
                self.snapshot_manager.save_snapshot(
                    conv_states, temporal_states, metadata
                )

                # Update tracker
                self.conversation_tracker.transition_tier(
                    conversation_id, ConversationTier.COLD
                )

                logger.info(f"Saved to COLD tier: {conversation_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to save to COLD tier: {e}", exc_info=True)
                return False

    def restore_from_warm_tier(
        self, conversation_id: str
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, dict]]:
        """
        Restore state from Tier 2 (host RAM).

        Args:
            conversation_id: Conversation identifier

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None
        """
        with self._lock:
            result = self.host_pool.get_state(conversation_id)

            if result:
                conv_states, temporal_states, metadata = result

                # Mark as accessed
                self.conversation_tracker.mark_accessed(conversation_id)

                logger.info(
                    f"Restored from WARM tier: {conversation_id}, "
                    f"hit_rate={self.host_pool.get_hit_rate():.2%}"
                )

                return conv_states, temporal_states, metadata

            return None

    def restore_from_cold_tier(
        self, conversation_id: str, turn_number: Optional[int] = None
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, MambaSnapshotMetadata]]:
        """
        Restore state from Tier 3 (disk).

        Args:
            conversation_id: Conversation identifier
            turn_number: Specific turn to restore (None = latest)

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None
        """
        with self._lock:
            try:
                # Get latest snapshot if turn not specified
                if turn_number is None:
                    latest = self.snapshot_manager.get_latest_snapshot(conversation_id)
                    if not latest:
                        return None
                    turn_number, _ = latest

                # Load snapshot
                conv_states, temporal_states, metadata = (
                    self.snapshot_manager.load_snapshot(conversation_id, turn_number)
                )

                # Mark as accessed
                self.conversation_tracker.mark_accessed(conversation_id)

                logger.info(
                    f"Restored from COLD tier: {conversation_id}, turn={turn_number}"
                )

                return conv_states, temporal_states, metadata

            except Exception as e:
                logger.error(f"Failed to restore from COLD tier: {e}", exc_info=True)
                return None

    def restore_conversation(
        self, conversation_id: str, turn_number: Optional[int] = None
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, dict]]:
        """
        Restore conversation state from optimal tier.

        This method automatically determines which tier the conversation
        is in and restores from the fastest available source.

        **Restoration Priority:**
        1. Host RAM (WARM) - Fastest
        2. Disk (COLD) - Slower but available

        Args:
            conversation_id: Conversation identifier
            turn_number: Specific turn to restore (None = latest)

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None
        """
        with self._lock:
            tier = self.conversation_tracker.get_tier(conversation_id)

            # Try WARM tier first (fastest)
            if tier == ConversationTier.WARM or self.host_pool.has_state(
                conversation_id
            ):
                result = self.restore_from_warm_tier(conversation_id)
                if result:
                    return result

            # Fall back to COLD tier
            if tier == ConversationTier.COLD or turn_number is not None:
                result = self.restore_from_cold_tier(conversation_id, turn_number)
                if result:
                    conv_states, temporal_states, metadata = result
                    # Promote to WARM tier for future fast access
                    self.save_to_warm_tier(
                        conversation_id,
                        conv_states,
                        temporal_states,
                        metadata.to_dict(),
                    )
                    return conv_states, temporal_states, metadata.to_dict()

            logger.warning(f"Failed to restore conversation: {conversation_id}")
            return None

    def load_cross_session_reference(
        self, current_conversation_id: str, reference_conversation_id: str
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, dict]]:
        """
        Load state from a different conversation (cross-session reference).

        This allows an agent to pull context from a previous conversation
        when the user references it.

        Example:
            User in conv_B: "Remember what we discussed about Python in our
                             last conversation?"
            Agent loads context from conv_A to answer.

        Args:
            current_conversation_id: Current conversation ID
            reference_conversation_id: Referenced conversation ID to load

        Returns:
            Tuple of (conv_states, temporal_states, metadata) or None
        """
        with self._lock:
            logger.info(
                f"Cross-session reference: {current_conversation_id} "
                f"loading from {reference_conversation_id}"
            )

            # Try host pool first (supports cross-session refs)
            result = self.host_pool.get_state_for_reference(
                current_conversation_id, reference_conversation_id
            )

            if result:
                return result

            # Fall back to restoring from cold tier
            return self.restore_conversation(reference_conversation_id)

    def demote_to_cold_tier(self, conversation_id: str) -> bool:
        """
        Explicitly demote conversation from WARM to COLD tier.

        This is called by the background cleanup thread when conversations
        in WARM tier exceed the timeout threshold.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if successfully demoted
        """
        with self._lock:
            # Get state from host pool
            result = self.host_pool.get_state(conversation_id)

            if not result:
                logger.warning(f"Cannot demote {conversation_id}: not in WARM tier")
                return False

            conv_states, temporal_states, metadata = result

            # Create snapshot metadata
            snapshot_metadata = MambaSnapshotMetadata(
                conversation_id=conversation_id,
                turn_number=metadata.get("turn_number", 0),
                timestamp=time.time(),
                token_count=metadata.get("token_count", 0),
                model_name=metadata.get("model_name", "unknown"),
                mamba_pool_idx=metadata.get("mamba_pool_idx", 0),
                req_pool_idx=metadata.get("req_pool_idx", 0),
                layer_config=metadata.get("layer_config", {}),
            )

            # Save to cold tier
            success = self.save_to_cold_tier(
                conversation_id, conv_states, temporal_states, snapshot_metadata
            )

            if success:
                # Remove from host pool
                self.host_pool.remove_state(conversation_id)

            return success

    def run_cleanup_cycle(self) -> dict:
        """
        Run one cleanup cycle to handle tier transitions.

        This method:
        - Checks for conversations that need demotion
        - WARM→COLD: Based on warm_timeout
        - COLD→ARCHIVED: Based on cold_retention

        Returns:
            Dict with counts of transitions performed
        """
        with self._lock:
            transitions_needed = self.conversation_tracker.check_transitions_needed()

            results = {
                "warm_to_cold": 0,
                "cold_to_archived": 0,
            }

            # WARM → COLD transitions
            for conv_id in transitions_needed["warm_to_cold"]:
                if self.demote_to_cold_tier(conv_id):
                    results["warm_to_cold"] += 1

            # COLD → ARCHIVED transitions (delete snapshots)
            for conv_id in transitions_needed["cold_to_archived"]:
                try:
                    # Delete all snapshots for this conversation
                    snapshots = self.snapshot_manager.list_snapshots(conv_id)
                    for turn in snapshots:
                        self.snapshot_manager.delete_snapshot(conv_id, turn)

                    # Update tracker
                    self.conversation_tracker.transition_tier(
                        conv_id, ConversationTier.ARCHIVED
                    )

                    results["cold_to_archived"] += 1

                except Exception as e:
                    logger.error(
                        f"Failed to archive conversation {conv_id}: {e}",
                        exc_info=True,
                    )

            if any(results.values()):
                logger.info(f"Cleanup cycle completed: {results}")

            return results

    def start_background_cleanup(self):
        """Start background cleanup thread."""
        if self._cleanup_running:
            logger.warning("Background cleanup already running")
            return

        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup_loop, daemon=True, name="TierManagerCleanup"
        )
        self._cleanup_thread.start()

        logger.info("Background cleanup thread started")

    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        if not self._cleanup_running:
            return

        self._cleanup_running = False

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        logger.info("Background cleanup thread stopped")

    def _background_cleanup_loop(self):
        """Background cleanup loop (runs in separate thread)."""
        while self._cleanup_running:
            try:
                time.sleep(self.cleanup_interval)

                if self._cleanup_running:
                    self.run_cleanup_cycle()

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get comprehensive tier manager statistics."""
        with self._lock:
            return {
                "conversation_tracker": self.conversation_tracker.get_stats(),
                "host_pool": self.host_pool.get_stats(),
                "cleanup_interval": self.cleanup_interval,
                "cleanup_running": self._cleanup_running,
            }

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_background_cleanup()
