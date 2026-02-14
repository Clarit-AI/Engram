"""
Unit tests for MambaHostPool (Tier 2: Warm Archive).

Tests the host memory pool functionality including LRU eviction,
cross-session references, and memory management.
"""

import tempfile
import time
from pathlib import Path

import pytest
import torch

from sglang.srt.snapshot.mamba_host_pool import MambaHostPool, HostPoolEntry


class TestMambaHostPool:
    """Test MambaHostPool functionality."""

    def create_dummy_state(self, num_layers=4, device="cpu"):
        """Create dummy Mamba state tensors for testing."""
        # Conv states
        conv_states = [
            torch.randn(num_layers, 64, 3, device=device) for _ in range(2)
        ]

        # Temporal state
        temporal_states = torch.randn(num_layers, 16, 128, 64, device=device)

        return conv_states, temporal_states

    def test_pool_initialization(self):
        """Test host pool initialization."""
        pool = MambaHostPool(
            max_conversations=50,
            max_memory_gb=5.0,
            enable_cross_session_refs=True,
        )

        assert pool.max_conversations == 50
        assert pool.max_memory_bytes == int(5.0 * 1024 * 1024 * 1024)
        assert pool.enable_cross_session_refs is True
        assert len(pool) == 0

    def test_save_and_get_state(self):
        """Test saving and retrieving state."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()

        # Save state
        success = pool.save_state("conv_1", conv_states, temporal_states)
        assert success is True
        assert len(pool) == 1

        # Get state
        result = pool.get_state("conv_1")
        assert result is not None

        loaded_conv, loaded_temporal, metadata = result

        # Verify shapes match
        assert len(loaded_conv) == len(conv_states)
        for i in range(len(conv_states)):
            assert loaded_conv[i].shape == conv_states[i].shape

        assert loaded_temporal.shape == temporal_states.shape

    def test_state_cloning(self):
        """Test that retrieved state is a clone, not original."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()

        # Save state
        pool.save_state("conv_1", conv_states, temporal_states)

        # Get state twice
        result1 = pool.get_state("conv_1")
        result2 = pool.get_state("conv_1")

        # Both should succeed
        assert result1 is not None
        assert result2 is not None

        conv1, temporal1, _ = result1
        conv2, temporal2, _ = result2

        # Data should match
        assert torch.allclose(conv1[0], conv2[0])
        assert torch.allclose(temporal1, temporal2)

        # But tensors should be different objects (clones)
        assert conv1[0].data_ptr() != conv2[0].data_ptr()
        assert temporal1.data_ptr() != temporal2.data_ptr()

    def test_lru_eviction(self):
        """Test LRU eviction when pool is full."""
        pool = MambaHostPool(max_conversations=3)

        # Add 3 conversations (fill pool)
        for i in range(3):
            conv_states, temporal_states = self.create_dummy_state()
            pool.save_state(f"conv_{i}", conv_states, temporal_states)

        assert len(pool) == 3

        # Access conv_0 to make it more recent
        pool.get_state("conv_0")

        # Add 4th conversation - should evict conv_1 (least recently used)
        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_3", conv_states, temporal_states)

        assert len(pool) == 3

        # conv_1 should be evicted, others should remain
        assert pool.has_state("conv_0")  # Accessed recently
        assert not pool.has_state("conv_1")  # Evicted (LRU)
        assert pool.has_state("conv_2")  # Still in pool
        assert pool.has_state("conv_3")  # Just added

    def test_memory_limit_eviction(self):
        """Test eviction based on memory limit."""
        # Set very small memory limit
        pool = MambaHostPool(max_conversations=100, max_memory_gb=0.001)  # 1MB

        # Add states until memory limit reached
        states_added = 0
        for i in range(10):
            conv_states, temporal_states = self.create_dummy_state()
            success = pool.save_state(f"conv_{i}", conv_states, temporal_states)
            if success:
                states_added += 1

        # Should have added some but not all (limited by memory)
        assert states_added > 0
        assert states_added < 10

        stats = pool.get_stats()
        assert stats["current_memory_bytes"] <= stats["max_memory_bytes"]

    def test_cross_session_reference(self):
        """Test cross-session reference functionality."""
        pool = MambaHostPool(enable_cross_session_refs=True)

        # Save state for conv_A
        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_A", conv_states, temporal_states, metadata={"user": "alice"})

        # Load from conv_B (cross-session ref)
        result = pool.get_state_for_reference("conv_B", "conv_A")

        assert result is not None
        loaded_conv, loaded_temporal, metadata = result
        assert metadata.get("user") == "alice"

    def test_cross_session_reference_disabled(self):
        """Test that cross-session refs can be disabled."""
        pool = MambaHostPool(enable_cross_session_refs=False)

        # Save state for conv_A
        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_A", conv_states, temporal_states)

        # Try cross-session ref (should fail)
        result = pool.get_state_for_reference("conv_B", "conv_A")

        assert result is None

    def test_remove_state(self):
        """Test removing state from pool."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_1", conv_states, temporal_states)

        assert pool.has_state("conv_1")

        # Remove state
        success = pool.remove_state("conv_1")

        assert success is True
        assert not pool.has_state("conv_1")
        assert len(pool) == 0

    def test_hit_miss_metrics(self):
        """Test hit/miss tracking."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_1", conv_states, temporal_states)

        # Hit
        pool.get_state("conv_1")

        # Miss
        pool.get_state("conv_2")

        stats = pool.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_access_count(self):
        """Test access count tracking."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()
        pool.save_state("conv_1", conv_states, temporal_states)

        # Access multiple times
        for _ in range(5):
            pool.get_state("conv_1")

        # Get metadata
        metadata = pool.get_conversation_metadata("conv_1")

        assert metadata is not None
        assert metadata["access_count"] == 5

    def test_list_conversations(self):
        """Test listing conversations in pool."""
        pool = MambaHostPool()

        # Add multiple conversations
        for i in range(5):
            conv_states, temporal_states = self.create_dummy_state()
            pool.save_state(f"conv_{i}", conv_states, temporal_states)
            time.sleep(0.01)  # Ensure different access times

        conversations = pool.list_conversations()

        assert len(conversations) == 5
        # Should be in LRU order (oldest first)
        assert conversations[0] == "conv_0"
        assert conversations[-1] == "conv_4"

    def test_clear_pool(self):
        """Test clearing the pool."""
        pool = MambaHostPool()

        # Add states
        for i in range(3):
            conv_states, temporal_states = self.create_dummy_state()
            pool.save_state(f"conv_{i}", conv_states, temporal_states)

        assert len(pool) == 3

        # Clear
        pool.clear()

        assert len(pool) == 0
        assert pool.get_stats()["current_memory_bytes"] == 0

    def test_metadata_persistence(self):
        """Test that metadata is preserved."""
        pool = MambaHostPool()

        conv_states, temporal_states = self.create_dummy_state()
        metadata = {
            "user_id": "user_123",
            "session_id": "session_456",
            "custom_field": "test_value",
        }

        pool.save_state("conv_1", conv_states, temporal_states, metadata=metadata)

        # Retrieve and check metadata
        result = pool.get_state("conv_1")
        assert result is not None

        _, _, loaded_metadata = result

        assert loaded_metadata["user_id"] == "user_123"
        assert loaded_metadata["session_id"] == "session_456"
        assert loaded_metadata["custom_field"] == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
