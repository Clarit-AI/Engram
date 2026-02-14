"""
Conversation state tracker and tier management.

Tracks which tier each conversation is in and manages automatic
transitions between tiers based on access patterns and timeouts.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConversationTier(Enum):
    """
    Tier where conversation state currently resides.

    ACTIVE (Tier 1): In VRAM (MambaPool), active inference
    WARM (Tier 2): In host RAM (MambaHostPool), fast restore
    COLD (Tier 3): On disk (snapshots), slower restore
    ARCHIVED: Marked for deletion, no longer accessible
    """

    ACTIVE = "active"  # VRAM
    WARM = "warm"  # Host RAM
    COLD = "cold"  # Disk
    ARCHIVED = "archived"  # Deleted


@dataclass
class ConversationState:
    """
    State information for a conversation.

    Attributes:
        conversation_id: Unique identifier
        tier: Current tier (ACTIVE/WARM/COLD/ARCHIVED)
        last_access_time: Last time conversation was accessed
        last_transition_time: Last time tier changed
        access_count: Total access count
        mamba_pool_idx: Index in MambaPool (if ACTIVE), None otherwise
        metadata: Optional user metadata (user_id, session_info, etc.)
    """

    conversation_id: str
    tier: ConversationTier
    last_access_time: float
    last_transition_time: float
    access_count: int = 0
    mamba_pool_idx: Optional[int] = None
    metadata: Optional[dict] = None


class ConversationTracker:
    """
    Tracks conversation states and manages tier transitions.

    This class:
    - Maintains a registry of all conversations and their current tier
    - Automatically transitions conversations between tiers based on timeouts
    - Provides APIs for querying and manually managing conversation tiers
    - Handles cross-session reference bookkeeping

    **Tier Transition Rules:**
    - ACTIVE → WARM: After `active_timeout` seconds of inactivity
    - WARM → COLD: After `warm_timeout` seconds of inactivity
    - COLD → ARCHIVED: After `cold_retention` seconds
    - Any tier → ACTIVE: On access/resume

    Thread Safety:
        All methods are thread-safe.
    """

    def __init__(
        self,
        active_timeout: float = 300.0,  # 5 minutes
        warm_timeout: float = 1800.0,  # 30 minutes
        cold_retention: float = 604800.0,  # 7 days
    ):
        """
        Initialize conversation tracker.

        Args:
            active_timeout: Seconds before ACTIVE→WARM transition
            warm_timeout: Seconds before WARM→COLD transition
            cold_retention: Seconds before COLD→ARCHIVED transition
        """
        self.active_timeout = active_timeout
        self.warm_timeout = warm_timeout
        self.cold_retention = cold_retention

        # Conversation registry
        self._conversations: Dict[str, ConversationState] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self._tier_transitions = {
            "active_to_warm": 0,
            "warm_to_cold": 0,
            "cold_to_archived": 0,
            "promotions": 0,  # Any tier → ACTIVE
        }

        logger.info(
            f"ConversationTracker initialized: "
            f"active_timeout={active_timeout}s, "
            f"warm_timeout={warm_timeout}s, "
            f"cold_retention={cold_retention}s"
        )

    def register_conversation(
        self,
        conversation_id: str,
        tier: ConversationTier = ConversationTier.ACTIVE,
        mamba_pool_idx: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Register a new conversation or update existing.

        Args:
            conversation_id: Conversation identifier
            tier: Initial tier
            mamba_pool_idx: MambaPool index if ACTIVE
            metadata: Optional metadata dict
        """
        with self._lock:
            now = time.time()

            if conversation_id in self._conversations:
                # Update existing
                state = self._conversations[conversation_id]
                old_tier = state.tier

                state.tier = tier
                state.last_access_time = now
                state.mamba_pool_idx = mamba_pool_idx
                state.metadata = metadata or state.metadata

                if old_tier != tier:
                    state.last_transition_time = now
                    logger.debug(
                        f"Conversation tier updated: {conversation_id} "
                        f"{old_tier.value}→{tier.value}"
                    )
            else:
                # Create new
                state = ConversationState(
                    conversation_id=conversation_id,
                    tier=tier,
                    last_access_time=now,
                    last_transition_time=now,
                    mamba_pool_idx=mamba_pool_idx,
                    metadata=metadata,
                )
                self._conversations[conversation_id] = state

                logger.info(
                    f"Conversation registered: {conversation_id}, tier={tier.value}"
                )

    def mark_accessed(self, conversation_id: str) -> None:
        """
        Mark conversation as accessed (updates timestamp).

        Args:
            conversation_id: Conversation identifier
        """
        with self._lock:
            if conversation_id in self._conversations:
                state = self._conversations[conversation_id]
                state.last_access_time = time.time()
                state.access_count += 1

    def transition_tier(
        self,
        conversation_id: str,
        new_tier: ConversationTier,
        mamba_pool_idx: Optional[int] = None,
    ) -> bool:
        """
        Manually transition conversation to a new tier.

        Args:
            conversation_id: Conversation identifier
            new_tier: Target tier
            mamba_pool_idx: MambaPool index if transitioning to ACTIVE

        Returns:
            True if transitioned, False if conversation not found
        """
        with self._lock:
            if conversation_id not in self._conversations:
                return False

            state = self._conversations[conversation_id]
            old_tier = state.tier

            if old_tier == new_tier:
                return True  # Already in target tier

            state.tier = new_tier
            state.last_transition_time = time.time()
            state.mamba_pool_idx = mamba_pool_idx

            # Update metrics
            transition_key = f"{old_tier.value}_to_{new_tier.value}"
            if transition_key in self._tier_transitions:
                self._tier_transitions[transition_key] += 1
            elif new_tier == ConversationTier.ACTIVE:
                self._tier_transitions["promotions"] += 1

            logger.info(
                f"Conversation tier transition: {conversation_id} "
                f"{old_tier.value}→{new_tier.value}"
            )

            return True

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get conversation state.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationState or None if not found
        """
        with self._lock:
            return self._conversations.get(conversation_id)

    def get_tier(self, conversation_id: str) -> Optional[ConversationTier]:
        """Get current tier for conversation."""
        with self._lock:
            state = self._conversations.get(conversation_id)
            return state.tier if state else None

    def check_transitions_needed(self) -> Dict[str, list]:
        """
        Check which conversations need tier transitions based on timeouts.

        Returns:
            Dict with lists of conversation_ids for each transition:
            {
                "active_to_warm": [...],
                "warm_to_cold": [...],
                "cold_to_archived": [...]
            }
        """
        with self._lock:
            now = time.time()
            transitions = {
                "active_to_warm": [],
                "warm_to_cold": [],
                "cold_to_archived": [],
            }

            for conv_id, state in self._conversations.items():
                time_since_access = now - state.last_access_time

                if state.tier == ConversationTier.ACTIVE:
                    if time_since_access > self.active_timeout:
                        transitions["active_to_warm"].append(conv_id)

                elif state.tier == ConversationTier.WARM:
                    if time_since_access > self.warm_timeout:
                        transitions["warm_to_cold"].append(conv_id)

                elif state.tier == ConversationTier.COLD:
                    if time_since_access > self.cold_retention:
                        transitions["cold_to_archived"].append(conv_id)

            return transitions

    def list_conversations_by_tier(
        self, tier: ConversationTier
    ) -> list[ConversationState]:
        """
        List all conversations in a specific tier.

        Args:
            tier: Tier to filter by

        Returns:
            List of ConversationState objects
        """
        with self._lock:
            return [
                state
                for state in self._conversations.values()
                if state.tier == tier
            ]

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        with self._lock:
            tier_counts = {
                "active": 0,
                "warm": 0,
                "cold": 0,
                "archived": 0,
            }

            for state in self._conversations.values():
                tier_counts[state.tier.value] += 1

            return {
                "total_conversations": len(self._conversations),
                "tier_counts": tier_counts,
                "transitions": self._tier_transitions.copy(),
                "timeouts": {
                    "active_timeout": self.active_timeout,
                    "warm_timeout": self.warm_timeout,
                    "cold_retention": self.cold_retention,
                },
            }

    def remove_conversation(self, conversation_id: str) -> bool:
        """
        Remove conversation from tracker.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                logger.info(f"Conversation removed from tracker: {conversation_id}")
                return True
            return False

    def clear(self):
        """Clear all conversations from tracker."""
        with self._lock:
            self._conversations.clear()
            logger.info("Conversation tracker cleared")
