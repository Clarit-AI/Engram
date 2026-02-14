"""
Unit tests for ConversationTracker.

Tests conversation state tracking and tier transition logic.
"""

import time

import pytest

from sglang.srt.snapshot.conversation_tracker import (
    ConversationState,
    ConversationTier,
    ConversationTracker,
)


class TestConversationTracker:
    """Test ConversationTracker functionality."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ConversationTracker(
            active_timeout=100.0,
            warm_timeout=500.0,
            cold_retention=1000.0,
        )

        assert tracker.active_timeout == 100.0
        assert tracker.warm_timeout == 500.0
        assert tracker.cold_retention == 1000.0

    def test_register_conversation(self):
        """Test registering a new conversation."""
        tracker = ConversationTracker()

        tracker.register_conversation(
            "conv_1",
            tier=ConversationTier.ACTIVE,
            mamba_pool_idx=1,
            metadata={"user": "alice"},
        )

        state = tracker.get_state("conv_1")

        assert state is not None
        assert state.conversation_id == "conv_1"
        assert state.tier == ConversationTier.ACTIVE
        assert state.mamba_pool_idx == 1
        assert state.metadata["user"] == "alice"

    def test_mark_accessed(self):
        """Test marking conversation as accessed."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)

        state_before = tracker.get_state("conv_1")
        initial_time = state_before.last_access_time
        initial_count = state_before.access_count

        time.sleep(0.01)

        tracker.mark_accessed("conv_1")

        state_after = tracker.get_state("conv_1")

        assert state_after.last_access_time > initial_time
        assert state_after.access_count == initial_count + 1

    def test_transition_tier(self):
        """Test manual tier transition."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)

        # Transition to WARM
        success = tracker.transition_tier("conv_1", ConversationTier.WARM)

        assert success is True

        state = tracker.get_state("conv_1")
        assert state.tier == ConversationTier.WARM

    def test_get_tier(self):
        """Test getting tier for conversation."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.WARM)

        tier = tracker.get_tier("conv_1")

        assert tier == ConversationTier.WARM

    def test_check_transitions_needed_active_to_warm(self):
        """Test detecting ACTIVE→WARM transitions."""
        tracker = ConversationTracker(active_timeout=0.1)  # 0.1 seconds

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)

        # Immediately check - no transition needed
        transitions = tracker.check_transitions_needed()
        assert len(transitions["active_to_warm"]) == 0

        # Wait for timeout
        time.sleep(0.15)

        transitions = tracker.check_transitions_needed()
        assert "conv_1" in transitions["active_to_warm"]

    def test_check_transitions_needed_warm_to_cold(self):
        """Test detecting WARM→COLD transitions."""
        tracker = ConversationTracker(warm_timeout=0.1)

        tracker.register_conversation("conv_1", tier=ConversationTier.WARM)

        # Wait for timeout
        time.sleep(0.15)

        transitions = tracker.check_transitions_needed()
        assert "conv_1" in transitions["warm_to_cold"]

    def test_check_transitions_needed_cold_to_archived(self):
        """Test detecting COLD→ARCHIVED transitions."""
        tracker = ConversationTracker(cold_retention=0.1)

        tracker.register_conversation("conv_1", tier=ConversationTier.COLD)

        # Wait for timeout
        time.sleep(0.15)

        transitions = tracker.check_transitions_needed()
        assert "conv_1" in transitions["cold_to_archived"]

    def test_list_conversations_by_tier(self):
        """Test listing conversations by tier."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)
        tracker.register_conversation("conv_2", tier=ConversationTier.WARM)
        tracker.register_conversation("conv_3", tier=ConversationTier.ACTIVE)
        tracker.register_conversation("conv_4", tier=ConversationTier.COLD)

        active_convs = tracker.list_conversations_by_tier(ConversationTier.ACTIVE)
        warm_convs = tracker.list_conversations_by_tier(ConversationTier.WARM)
        cold_convs = tracker.list_conversations_by_tier(ConversationTier.COLD)

        assert len(active_convs) == 2
        assert len(warm_convs) == 1
        assert len(cold_convs) == 1

    def test_get_stats(self):
        """Test getting tracker statistics."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)
        tracker.register_conversation("conv_2", tier=ConversationTier.WARM)
        tracker.register_conversation("conv_3", tier=ConversationTier.COLD)

        stats = tracker.get_stats()

        assert stats["total_conversations"] == 3
        assert stats["tier_counts"]["active"] == 1
        assert stats["tier_counts"]["warm"] == 1
        assert stats["tier_counts"]["cold"] == 1

    def test_remove_conversation(self):
        """Test removing conversation from tracker."""
        tracker = ConversationTracker()

        tracker.register_conversation("conv_1", tier=ConversationTier.ACTIVE)

        assert tracker.get_state("conv_1") is not None

        success = tracker.remove_conversation("conv_1")

        assert success is True
        assert tracker.get_state("conv_1") is None

    def test_clear(self):
        """Test clearing all conversations."""
        tracker = ConversationTracker()

        for i in range(5):
            tracker.register_conversation(f"conv_{i}", tier=ConversationTier.ACTIVE)

        tracker.clear()

        stats = tracker.get_stats()
        assert stats["total_conversations"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
