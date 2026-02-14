"""
Unit tests for Mamba snapshot serialization and management.

Tests the core snapshot save/load functionality, metadata handling,
and state extraction/injection.
"""

import tempfile
import time
from pathlib import Path

import pytest
import torch

from sglang.srt.snapshot import MambaSnapshotManager, MambaSnapshotMetadata


class TestMambaSnapshotMetadata:
    """Test MambaSnapshotMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating snapshot metadata."""
        metadata = MambaSnapshotMetadata(
            conversation_id="test_conv_1",
            turn_number=5,
            timestamp=time.time(),
            token_count=100,
            model_name="ibm-granite/granite-4.0-h-small",
            mamba_pool_idx=1,
            req_pool_idx=1,
            layer_config={"num_layers": 32, "state_size": 16},
        )

        assert metadata.conversation_id == "test_conv_1"
        assert metadata.turn_number == 5
        assert metadata.token_count == 100

    def test_metadata_to_from_dict(self):
        """Test metadata serialization to/from dict."""
        metadata = MambaSnapshotMetadata(
            conversation_id="test_conv_2",
            turn_number=10,
            timestamp=time.time(),
            token_count=200,
            model_name="test-model",
            mamba_pool_idx=2,
            req_pool_idx=2,
            layer_config={"num_layers": 16},
        )

        # Convert to dict
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["conversation_id"] == "test_conv_2"

        # Convert back from dict
        restored = MambaSnapshotMetadata.from_dict(metadata_dict)
        assert restored.conversation_id == metadata.conversation_id
        assert restored.turn_number == metadata.turn_number
        assert restored.token_count == metadata.token_count

    def test_metadata_json_serialization(self):
        """Test metadata save/load to/from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = MambaSnapshotMetadata(
                conversation_id="test_conv_3",
                turn_number=15,
                timestamp=time.time(),
                token_count=300,
                model_name="test-model",
                mamba_pool_idx=3,
                req_pool_idx=3,
                layer_config={"num_layers": 8},
            )

            # Save to JSON
            json_path = Path(tmpdir) / "test_metadata.json"
            metadata.to_json(json_path)

            # Verify file exists
            assert json_path.exists()

            # Load from JSON
            restored = MambaSnapshotMetadata.from_json(json_path)
            assert restored.conversation_id == metadata.conversation_id
            assert restored.turn_number == metadata.turn_number


class TestMambaSnapshotManager:
    """Test MambaSnapshotManager functionality."""

    def create_dummy_state(self, num_layers=4, device="cpu"):
        """Create dummy Mamba state tensors for testing."""
        # Conv states (list of tensors, one per layer)
        conv_states = [
            torch.randn(num_layers, 64, 3, device=device) for _ in range(2)
        ]

        # Temporal state
        temporal_states = torch.randn(num_layers, 16, 128, 64, device=device)

        return conv_states, temporal_states

    def test_manager_initialization(self):
        """Test snapshot manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            assert manager.base_dir == Path(tmpdir)
            assert manager.base_dir.exists()

    def test_save_and_load_snapshot(self):
        """Test saving and loading a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            # Create dummy state
            conv_states, temporal_states = self.create_dummy_state(num_layers=4)

            # Create metadata
            metadata = MambaSnapshotMetadata(
                conversation_id="test_save_load",
                turn_number=1,
                timestamp=time.time(),
                token_count=50,
                model_name="test-model",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
            )

            # Save snapshot
            metadata_path, state_path = manager.save_snapshot(
                conv_states, temporal_states, metadata
            )

            assert metadata_path.exists()
            assert state_path.exists()

            # Load snapshot
            loaded_conv, loaded_temporal, loaded_metadata = manager.load_snapshot(
                "test_save_load", turn_number=1
            )

            # Verify shapes match
            assert len(loaded_conv) == len(conv_states)
            for i in range(len(conv_states)):
                assert loaded_conv[i].shape == conv_states[i].shape

            assert loaded_temporal.shape == temporal_states.shape

            # Verify metadata
            assert loaded_metadata.conversation_id == "test_save_load"
            assert loaded_metadata.turn_number == 1

    def test_list_conversations(self):
        """Test listing conversations with snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            # Create snapshots for multiple conversations
            for conv_id in ["conv_1", "conv_2", "conv_3"]:
                conv_states, temporal_states = self.create_dummy_state()
                metadata = MambaSnapshotMetadata(
                    conversation_id=conv_id,
                    turn_number=1,
                    timestamp=time.time(),
                    token_count=10,
                    model_name="test",
                    mamba_pool_idx=1,
                    req_pool_idx=1,
                    layer_config={"num_layers": 4},
                )
                manager.save_snapshot(conv_states, temporal_states, metadata)

            conversations = manager.list_conversations()
            assert len(conversations) == 3
            assert "conv_1" in conversations
            assert "conv_2" in conversations
            assert "conv_3" in conversations

    def test_list_snapshots(self):
        """Test listing snapshots for a conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            # Create multiple turns
            conv_id = "test_list_snapshots"
            for turn in [0, 1, 2, 5, 10]:
                conv_states, temporal_states = self.create_dummy_state()
                metadata = MambaSnapshotMetadata(
                    conversation_id=conv_id,
                    turn_number=turn,
                    timestamp=time.time(),
                    token_count=turn * 10,
                    model_name="test",
                    mamba_pool_idx=1,
                    req_pool_idx=1,
                    layer_config={"num_layers": 4},
                )
                manager.save_snapshot(conv_states, temporal_states, metadata)

            snapshots = manager.list_snapshots(conv_id)
            assert len(snapshots) == 5
            assert snapshots == [0, 1, 2, 5, 10]

    def test_get_latest_snapshot(self):
        """Test getting the latest snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            conv_id = "test_latest"

            # Create snapshots
            for turn in [0, 1, 2]:
                conv_states, temporal_states = self.create_dummy_state()
                metadata = MambaSnapshotMetadata(
                    conversation_id=conv_id,
                    turn_number=turn,
                    timestamp=time.time(),
                    token_count=turn * 10,
                    model_name="test",
                    mamba_pool_idx=1,
                    req_pool_idx=1,
                    layer_config={"num_layers": 4},
                )
                manager.save_snapshot(conv_states, temporal_states, metadata)

            # Get latest
            latest_turn, latest_metadata = manager.get_latest_snapshot(conv_id)

            assert latest_turn == 2
            assert latest_metadata.turn_number == 2

    def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            conv_id = "test_delete"

            # Create snapshot
            conv_states, temporal_states = self.create_dummy_state()
            metadata = MambaSnapshotMetadata(
                conversation_id=conv_id,
                turn_number=1,
                timestamp=time.time(),
                token_count=10,
                model_name="test",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
            )
            manager.save_snapshot(conv_states, temporal_states, metadata)

            # Verify it exists
            snapshots = manager.list_snapshots(conv_id)
            assert 1 in snapshots

            # Delete it
            manager.delete_snapshot(conv_id, turn_number=1)

            # Verify it's gone
            snapshots = manager.list_snapshots(conv_id)
            assert 1 not in snapshots

    def test_branch_creation(self):
        """Test creating named branches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            conv_id = "test_branch"

            # Create main snapshot
            conv_states, temporal_states = self.create_dummy_state()
            main_metadata = MambaSnapshotMetadata(
                conversation_id=conv_id,
                turn_number=5,
                timestamp=time.time(),
                token_count=50,
                model_name="test",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
            )
            manager.save_snapshot(conv_states, temporal_states, main_metadata)

            # Create branch
            branch_metadata = MambaSnapshotMetadata(
                conversation_id=conv_id,
                turn_number=5,
                timestamp=time.time(),
                token_count=50,
                model_name="test",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
                branch_name="exploration_1",
                parent_snapshot="turn_5",
            )
            manager.save_snapshot(conv_states, temporal_states, branch_metadata)

            # Verify branch exists
            branches = manager.list_branches(conv_id)
            assert "exploration_1" in branches

            # Load branch
            loaded_conv, loaded_temporal, loaded_metadata = manager.load_snapshot(
                conv_id, branch_name="exploration_1"
            )

            assert loaded_metadata.branch_name == "exploration_1"
            assert loaded_metadata.parent_snapshot == "turn_5"

    def test_get_snapshot_size(self):
        """Test getting snapshot disk size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MambaSnapshotManager(Path(tmpdir))

            conv_id = "test_size"

            # Create snapshot
            conv_states, temporal_states = self.create_dummy_state()
            metadata = MambaSnapshotMetadata(
                conversation_id=conv_id,
                turn_number=1,
                timestamp=time.time(),
                token_count=10,
                model_name="test",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
            )
            manager.save_snapshot(conv_states, temporal_states, metadata)

            # Get size
            size = manager.get_snapshot_size(conv_id, turn_number=1)

            assert size > 0  # Should have some disk usage


class TestSnapshotStateExtraction:
    """Test state extraction and injection helpers."""

    def test_extract_and_inject_roundtrip(self):
        """Test extracting and re-injecting state (integration test)."""
        # This test requires a mock MambaPool
        # For now, we'll skip this and implement it when we have
        # full integration tests with the actual pool
        pytest.skip("Requires MambaPool integration - deferred to integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
