"""
Unit tests for snapshot retention policy and lifecycle management.

Tests the policy decision logic, retention management, and branch handling.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.snapshot import MambaSnapshotManager
from sglang.srt.snapshot.snapshot_policy import (
    SnapshotRetentionConfig,
    SnapshotRetentionPolicy,
    SnapshotTriggerPolicy,
)


class TestSnapshotRetentionPolicy:
    """Test snapshot retention policy."""

    def create_test_manager(self, tmpdir):
        """Create a test snapshot manager."""
        return MambaSnapshotManager(Path(tmpdir))

    def create_mock_req(self, conversation_id="test_conv", rid="req_001"):
        """Create a mock request object."""
        req = Mock()
        req.conversation_id = conversation_id
        req.rid = rid
        return req

    def test_policy_every_turn(self):
        """Test EVERY_TURN policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                snapshot_trigger_policy=SnapshotTriggerPolicy.EVERY_TURN,
                min_snapshot_interval_seconds=0.0,  # Disable interval check for test
            )

            policy = SnapshotRetentionPolicy(manager, config)

            req = self.create_mock_req()

            # Should snapshot every turn
            assert policy.should_snapshot(req, turn_number=0)
            policy.mark_snapshot_taken("test_conv")

            assert policy.should_snapshot(req, turn_number=1)
            policy.mark_snapshot_taken("test_conv")

            assert policy.should_snapshot(req, turn_number=2)

    def test_policy_every_n_turns(self):
        """Test EVERY_N_TURNS policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                snapshot_trigger_policy=SnapshotTriggerPolicy.EVERY_N_TURNS,
                snapshot_every_n_turns=5,
                min_snapshot_interval_seconds=0.0,
            )

            policy = SnapshotRetentionPolicy(manager, config)

            req = self.create_mock_req()

            # Turn 0: should snapshot (0 % 5 == 0)
            assert policy.should_snapshot(req, turn_number=0)

            # Turn 1-4: should NOT snapshot
            assert not policy.should_snapshot(req, turn_number=1)
            assert not policy.should_snapshot(req, turn_number=2)
            assert not policy.should_snapshot(req, turn_number=3)
            assert not policy.should_snapshot(req, turn_number=4)

            # Turn 5: should snapshot (5 % 5 == 0)
            assert policy.should_snapshot(req, turn_number=5)

            # Turn 10: should snapshot (10 % 5 == 0)
            assert policy.should_snapshot(req, turn_number=10)

    def test_policy_on_tool_call(self):
        """Test ON_TOOL_CALL policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                snapshot_trigger_policy=SnapshotTriggerPolicy.ON_TOOL_CALL,
                min_snapshot_interval_seconds=0.0,
            )

            policy = SnapshotRetentionPolicy(manager, config)

            req = self.create_mock_req()

            # No tool call: should NOT snapshot
            assert not policy.should_snapshot(
                req, turn_number=1, additional_context={}
            )

            # Tool call made: should snapshot
            assert policy.should_snapshot(
                req,
                turn_number=1,
                additional_context={"tool_call_made": True},
            )

    def test_policy_manual_only(self):
        """Test MANUAL_ONLY policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                snapshot_trigger_policy=SnapshotTriggerPolicy.MANUAL_ONLY,
            )

            policy = SnapshotRetentionPolicy(manager, config)

            req = self.create_mock_req()

            # Should never snapshot automatically
            assert not policy.should_snapshot(req, turn_number=0)
            assert not policy.should_snapshot(req, turn_number=1)
            assert not policy.should_snapshot(req, turn_number=10)

    def test_minimum_interval_enforcement(self):
        """Test that minimum interval is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                snapshot_trigger_policy=SnapshotTriggerPolicy.EVERY_TURN,
                min_snapshot_interval_seconds=1.0,  # 1 second minimum
            )

            policy = SnapshotRetentionPolicy(manager, config)

            req = self.create_mock_req()

            # First snapshot: should work
            assert policy.should_snapshot(req, turn_number=0)
            policy.mark_snapshot_taken("test_conv")

            # Immediate next snapshot: should be blocked by interval
            assert not policy.should_snapshot(req, turn_number=1)

            # Wait for interval
            time.sleep(1.1)

            # Now should work
            assert policy.should_snapshot(req, turn_number=2)

    def test_prune_old_snapshots(self):
        """Test automatic pruning of old snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig(
                max_snapshots_per_conversation=3,
                enable_auto_pruning=True,
            )

            policy = SnapshotRetentionPolicy(manager, config)

            conv_id = "test_prune"

            # Create 5 snapshots
            from sglang.srt.snapshot import MambaSnapshotMetadata

            for turn in range(5):
                conv_states = [torch.randn(4, 64, 3) for _ in range(2)]
                temporal_states = torch.randn(4, 16, 128, 64)

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

            # Should have 5 snapshots
            assert len(manager.list_snapshots(conv_id)) == 5

            # Prune (should keep only last 3)
            deleted = policy.prune_old_snapshots(conv_id)

            assert deleted == 2
            assert len(manager.list_snapshots(conv_id)) == 3

            # Verify it kept the most recent ones
            snapshots = manager.list_snapshots(conv_id)
            assert snapshots == [2, 3, 4]

    def test_create_and_list_branches(self):
        """Test branch creation and listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig()
            policy = SnapshotRetentionPolicy(manager, config)

            conv_id = "test_branch"

            # Create a main snapshot first
            from sglang.srt.snapshot import MambaSnapshotMetadata

            conv_states = [torch.randn(4, 64, 3) for _ in range(2)]
            temporal_states = torch.randn(4, 16, 128, 64)

            metadata = MambaSnapshotMetadata(
                conversation_id=conv_id,
                turn_number=5,
                timestamp=time.time(),
                token_count=50,
                model_name="test",
                mamba_pool_idx=1,
                req_pool_idx=1,
                layer_config={"num_layers": 4},
            )

            manager.save_snapshot(conv_states, temporal_states, metadata)

            # Create branch from turn 5
            success = policy.create_branch(conv_id, "exploration_1", source_turn=5)

            assert success

            # List branches
            branches = policy.list_branches(conv_id)
            assert "exploration_1" in branches

    def test_delete_branch(self):
        """Test branch deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig()
            policy = SnapshotRetentionPolicy(manager, config)

            conv_id = "test_delete_branch"

            # Create snapshot and branch
            from sglang.srt.snapshot import MambaSnapshotMetadata

            conv_states = [torch.randn(4, 64, 3) for _ in range(2)]
            temporal_states = torch.randn(4, 16, 128, 64)

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

            # Create branch
            policy.create_branch(conv_id, "test_branch", source_turn=1)

            assert "test_branch" in policy.list_branches(conv_id)

            # Delete branch
            success = policy.delete_branch(conv_id, "test_branch")

            assert success
            assert "test_branch" not in policy.list_branches(conv_id)

    def test_get_total_snapshot_size(self):
        """Test calculating total snapshot size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig()
            policy = SnapshotRetentionPolicy(manager, config)

            conv_id = "test_size"

            # Create a few snapshots
            from sglang.srt.snapshot import MambaSnapshotMetadata

            for turn in range(3):
                conv_states = [torch.randn(4, 64, 3) for _ in range(2)]
                temporal_states = torch.randn(4, 16, 128, 64)

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

            # Get total size
            total_size = policy.get_total_snapshot_size(conv_id)

            assert total_size > 0  # Should have some disk usage

    def test_cleanup_all_snapshots(self):
        """Test cleaning up all snapshots for a conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self.create_test_manager(tmpdir)

            config = SnapshotRetentionConfig()
            policy = SnapshotRetentionPolicy(manager, config)

            conv_id = "test_cleanup"

            # Create snapshots and branches
            from sglang.srt.snapshot import MambaSnapshotMetadata

            for turn in range(3):
                conv_states = [torch.randn(4, 64, 3) for _ in range(2)]
                temporal_states = torch.randn(4, 16, 128, 64)

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

            # Create a branch
            policy.create_branch(conv_id, "test_branch", source_turn=1)

            # Should have 3 snapshots + 1 branch = 4 total
            assert len(manager.list_snapshots(conv_id)) == 3
            assert len(manager.list_branches(conv_id)) == 1

            # Cleanup all
            deleted = policy.cleanup_all_snapshots(conv_id)

            assert deleted == 4
            assert len(manager.list_snapshots(conv_id)) == 0
            assert len(manager.list_branches(conv_id)) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
