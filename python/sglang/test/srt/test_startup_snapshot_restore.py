import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from sglang.srt.managers.io_struct import RestoreSnapshotReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_batch import DisaggregationMode, Req
from sglang.srt.sampling.sampling_params import SamplingParams
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


def _build_scheduler(snapshot_manager, tier_manager=None, conversation_tracker=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.snapshot_manager = snapshot_manager
    scheduler.tier_manager = tier_manager
    scheduler.conversation_tracker = conversation_tracker
    return scheduler


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


def test_scheduler_restore_snapshots_on_startup_guard_and_delegate(tmp_path):
    snapshot_manager = MambaSnapshotManager(tmp_path / "snapshots")
    conversation_tracker = ConversationTracker()
    host_pool = MambaHostPool(max_conversations=10, max_memory_gb=1.0)
    tier_manager = TierManager(
        conversation_tracker=conversation_tracker,
        host_pool=host_pool,
        snapshot_manager=snapshot_manager,
        enable_background_cleanup=False,
    )

    with patch(
        "sglang.srt.snapshot.tier_manager.restore_latest_snapshots_to_warm_tier"
    ) as restore_mock:
        _build_scheduler(snapshot_manager, None, conversation_tracker).restore_snapshots_on_startup()
        restore_mock.assert_not_called()

    with patch(
        "sglang.srt.snapshot.tier_manager.restore_latest_snapshots_to_warm_tier"
    ) as restore_mock:
        _build_scheduler(snapshot_manager, tier_manager, None).restore_snapshots_on_startup()
        restore_mock.assert_not_called()

    with patch(
        "sglang.srt.snapshot.tier_manager.restore_latest_snapshots_to_warm_tier"
    ) as restore_mock:
        _build_scheduler(
            snapshot_manager, tier_manager, conversation_tracker
        ).restore_snapshots_on_startup()
        restore_mock.assert_called_once_with(
            snapshot_manager=snapshot_manager,
            tier_manager=tier_manager,
            restore_logger=logging.getLogger("sglang.srt.managers.scheduler"),
        )


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


def test_save_to_warm_tier_does_not_register_ghost_conversation(tmp_path):
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

    with patch.object(host_pool, "save_state", return_value=False):
        assert not tier_manager.save_to_warm_tier(
            "ghost-conv", conv_states, temporal_states, {}
        )

    assert not host_pool.has_state("ghost-conv")
    assert conversation_tracker.get_state("ghost-conv") is None


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


def test_handle_restore_snapshot_treats_empty_continuation_as_stateful():
    scheduler = Scheduler.__new__(Scheduler)
    snapshot_manager = Mock()
    snapshot_manager.load_snapshot.return_value = (
        [torch.ones((1, 2))],
        torch.ones((1, 2)),
        SimpleNamespace(fill_ids=[11, 12, 13], token_count=3),
    )
    snapshot_manager.inject_state_to_pool = Mock()
    scheduler.snapshot_manager = snapshot_manager

    mamba_pool = Mock()
    mamba_pool.alloc.return_value = torch.tensor([4], dtype=torch.int64)
    scheduler.req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
    scheduler.max_req_input_len = 128
    scheduler.model_config = SimpleNamespace(vocab_size=32000)
    scheduler.device = torch.device("cpu")
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.init_req_max_new_tokens = lambda req: None

    admitted_req = {}

    def _add_request_to_queue(req):
        admitted_req["req"] = req
        req.time_stats.wait_queue_entry_time = 1.0

    scheduler._add_request_to_queue = _add_request_to_queue

    recv_req = RestoreSnapshotReqInput(
        conversation_id="conv-a",
        turn_number=2,
        create_new_request=True,
        continuation_ids=[],
        max_new_tokens=16,
    )

    result = scheduler.handle_restore_snapshot(recv_req)

    assert result is None
    queued_req = admitted_req["req"]
    assert queued_req._stateful_generate is True
    assert queued_req.origin_input_ids == [11, 12, 13]
    assert queued_req.fill_ids.tolist() == [11, 12, 13]


def _build_waiting_req(rid: str, priority: int, mamba_pool_idx: int) -> Req:
    sampling_params = SamplingParams()
    sampling_params.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=sampling_params,
        vocab_size=32000,
        priority=priority,
    )
    req.mamba_pool_idx = torch.tensor(mamba_pool_idx, dtype=torch.int64)
    return req


def _build_queue_cleanup_scheduler() -> tuple[Scheduler, Mock]:
    scheduler = Scheduler.__new__(Scheduler)
    mamba_pool = Mock()
    scheduler.tree_cache = SimpleNamespace(
        supports_mamba=lambda: True,
        req_to_token_pool=SimpleNamespace(mamba_pool=mamba_pool),
    )
    scheduler.send_to_tokenizer = SimpleNamespace(send_output=Mock())
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_priority_scheduling = True
    scheduler.schedule_low_priority_values_first = True
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    scheduler.req_to_metadata_buffer_idx_allocator = None
    return scheduler, mamba_pool


def test_abort_on_queued_limit_frees_mamba_slot_for_preempted_waiting_request():
    scheduler, mamba_pool = _build_queue_cleanup_scheduler()
    existing_req = _build_waiting_req("queued-existing", priority=9, mamba_pool_idx=3)
    existing_req.time_stats.wait_queue_entry_time = time.perf_counter() - 1.0
    incoming_req = _build_waiting_req("queued-new", priority=1, mamba_pool_idx=5)

    scheduler.max_queued_requests = 1
    scheduler.waiting_queue = [existing_req]

    incoming_aborted = scheduler._abort_on_queued_limit(incoming_req)

    assert incoming_aborted is False
    assert scheduler.waiting_queue == []
    mamba_pool.free.assert_called_once()
    assert existing_req.mamba_pool_idx is None


def test_abort_on_waiting_timeout_frees_mamba_slot_for_timed_out_request():
    scheduler, mamba_pool = _build_queue_cleanup_scheduler()
    timed_out_req = _build_waiting_req("queued-timeout", priority=1, mamba_pool_idx=7)
    timed_out_req.time_stats.wait_queue_entry_time = time.perf_counter() - 1.0

    scheduler.waiting_queue = [timed_out_req]

    with patch("sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get", return_value=0.1):
        scheduler._abort_on_waiting_timeout()

    assert scheduler.waiting_queue == []
    mamba_pool.free.assert_called_once()
    assert timed_out_req.mamba_pool_idx is None


def test_trigger_snapshot_hooks_skips_finished_decode_requests():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.snapshot_hook_manager = Mock()
    scheduler.req_to_token_pool = SimpleNamespace(mamba_pool=Mock())

    req = Mock()
    req.mamba_pool_idx = torch.tensor(1, dtype=torch.int64)
    req.finished.return_value = True
    req.output_ids = [1, 2, 3]

    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )

    scheduler._trigger_snapshot_hooks(batch)

    scheduler.snapshot_hook_manager.trigger_post_forward.assert_not_called()


def test_trigger_snapshot_hooks_keeps_finished_non_decode_requests():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.snapshot_hook_manager = Mock()
    scheduler.req_to_token_pool = SimpleNamespace(mamba_pool=Mock())

    req = Mock()
    req.mamba_pool_idx = torch.tensor(2, dtype=torch.int64)
    req.finished.return_value = True
    req.output_ids = [4, 5]

    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(is_decode=lambda: False),
    )

    scheduler._trigger_snapshot_hooks(batch)

    scheduler.snapshot_hook_manager.trigger_post_forward.assert_called_once()
