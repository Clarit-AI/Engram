import logging

import torch

from sglang.srt.snapshot.conversation_tracker import ConversationTier, ConversationTracker
from sglang.srt.snapshot.mamba_host_pool import MambaHostPool
from sglang.srt.snapshot.mamba_snapshot import (
    MambaSnapshotManager,
    MambaSnapshotMetadata,
)
from sglang.srt.snapshot.tier_manager import (
    TierManager,
    restore_latest_snapshots_to_warm_tier,
)


def _save_snapshot(
    snapshot_manager: MambaSnapshotManager,
    conversation_id: str,
    turn_number: int,
    token_count: int,
):
    conv_states = [torch.full((1, 2), float(turn_number + 1))]
    temporal_states = torch.full((1, 2), float(turn_number + 10))
    metadata = MambaSnapshotMetadata(
        conversation_id=conversation_id,
        turn_number=turn_number,
        timestamp=turn_number + 0.5,
        token_count=token_count,
        model_name="test-mamba",
        mamba_pool_idx=1,
        req_pool_idx=1,
        layer_config={"num_layers": 1},
        fill_ids=[1, 2, 3],
    )
    snapshot_manager.save_snapshot(conv_states, temporal_states, metadata)


def test_restore_snapshots_on_startup_preloads_latest_snapshots_to_warm_tier(tmp_path):
    snapshot_manager = MambaSnapshotManager(tmp_path / "snapshots")
    conversation_tracker = ConversationTracker()
    host_pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)
    tier_manager = TierManager(
        conversation_tracker=conversation_tracker,
        host_pool=host_pool,
        snapshot_manager=snapshot_manager,
        enable_background_cleanup=False,
    )

    _save_snapshot(snapshot_manager, "conv-a", 0, 11)
    _save_snapshot(snapshot_manager, "conv-a", 2, 29)
    _save_snapshot(snapshot_manager, "conv-b", 1, 17)

    restore_latest_snapshots_to_warm_tier(snapshot_manager, tier_manager)

    assert host_pool.has_state("conv-a")
    assert host_pool.has_state("conv-b")
    assert conversation_tracker.get_tier("conv-a") == ConversationTier.WARM
    assert conversation_tracker.get_tier("conv-b") == ConversationTier.WARM

    _, _, conv_a_metadata = host_pool.get_state("conv-a")
    assert conv_a_metadata["turn_number"] == 2
    assert conv_a_metadata["token_count"] == 29


def test_restore_snapshots_on_startup_continues_after_failure(tmp_path, caplog):
    snapshot_manager = MambaSnapshotManager(tmp_path / "snapshots")
    conversation_tracker = ConversationTracker()
    host_pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)
    tier_manager = TierManager(
        conversation_tracker=conversation_tracker,
        host_pool=host_pool,
        snapshot_manager=snapshot_manager,
        enable_background_cleanup=False,
    )

    _save_snapshot(snapshot_manager, "healthy", 0, 7)

    broken_dir = tmp_path / "snapshots" / "conversation_broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "turn_0_metadata.json").write_text("{not-json")

    caplog.set_level(logging.INFO)
    restore_latest_snapshots_to_warm_tier(snapshot_manager, tier_manager)

    assert host_pool.has_state("healthy")
    assert conversation_tracker.get_tier("healthy") == ConversationTier.WARM
    assert "Failed to restore conversation broken on startup" in caplog.text
    assert "Startup restore complete: 1/2 conversation(s) pre-loaded to WARM tier" in caplog.text


def test_save_to_warm_tier_registers_untracked_conversation(tmp_path):
    snapshot_manager = MambaSnapshotManager(tmp_path / "snapshots")
    conversation_tracker = ConversationTracker()
    host_pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)
    tier_manager = TierManager(
        conversation_tracker=conversation_tracker,
        host_pool=host_pool,
        snapshot_manager=snapshot_manager,
        enable_background_cleanup=False,
    )

    conv_states = [torch.ones((1, 2))]
    temporal_states = torch.ones((1, 2))
    metadata = {"conversation_id": "conv-a", "turn_number": 3}

    assert tier_manager.save_to_warm_tier(
        "conv-a", conv_states, temporal_states, metadata
    )
    assert host_pool.has_state("conv-a")
    assert conversation_tracker.get_tier("conv-a") == ConversationTier.WARM


def test_restore_snapshots_on_startup_no_snapshots_logs_cleanly(tmp_path, caplog):
    snapshot_manager = MambaSnapshotManager(tmp_path / "snapshots")
    conversation_tracker = ConversationTracker()
    host_pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)
    tier_manager = TierManager(
        conversation_tracker=conversation_tracker,
        host_pool=host_pool,
        snapshot_manager=snapshot_manager,
        enable_background_cleanup=False,
    )

    caplog.set_level(logging.INFO)
    restore_latest_snapshots_to_warm_tier(snapshot_manager, tier_manager)

    assert "No previous snapshots found" in caplog.text
    assert len(host_pool) == 0
