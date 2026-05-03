"""Unit tests for the pending-restore registry on the Scheduler.

The registry stages loaded snapshot state for a future incoming request when
/restore_snapshot is called after the originating Req has finished. Tests
exercise the producer (_stage_pending_restore), capacity bound + LRU eviction,
key aliasing (rid + conversation_id), and the consumer hydration semantics
(_maybe_hydrate_from_pending_restore) including pool allocation and Req-state
mutation.
"""

import os
from collections import OrderedDict
from types import SimpleNamespace

import torch

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from sglang.srt.managers.scheduler import (
    PENDING_RESTORE_REGISTRY_MAX,
    PendingRestoreEntry,
    Scheduler,
)


class _FakeMambaPool:
    """Minimal mamba_pool stub: hands out monotonic 0-dim tensor indices."""

    def __init__(self, capacity: int = 8):
        self._next = 0
        self._capacity = capacity
        self._freed = []

    def alloc(self, n: int):
        if self._next >= self._capacity:
            return None
        out = torch.tensor([self._next], dtype=torch.int32)
        self._next += n
        return out

    def free(self, idx):
        self._freed.append(idx)


class _FakeSnapshotManager:
    """Records inject_state_to_pool calls so tests can assert on them."""

    def __init__(self):
        self.injected = []

    def inject_state_to_pool(self, conv_states, temporal_states, mamba_pool, idx):
        self.injected.append(
            {
                "conv_states": conv_states,
                "temporal_states": temporal_states,
                "mamba_pool": mamba_pool,
                "idx": idx,
            }
        )


def _build_scheduler():
    """Build a bare Scheduler with only the fields the registry methods touch."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pending_restore_registry = OrderedDict()
    scheduler.snapshot_manager = _FakeSnapshotManager()
    scheduler.state_health_monitor = None
    scheduler._health_check_counter = {}
    mamba_pool = _FakeMambaPool()
    scheduler.req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
    return scheduler, mamba_pool


def _make_state(value: float = 1.0):
    return [torch.full((1, 2), value)], torch.full((1, 2), value + 1)


def _make_req(rid: str, conversation_id=None, mamba_pool_idx=None):
    return SimpleNamespace(
        rid=rid,
        conversation_id=conversation_id,
        mamba_pool_idx=mamba_pool_idx,
        origin_input_ids=[10, 20, 30],
    )


def test_stage_indexes_by_rid_and_conversation_id():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1, 2, 3],
    )

    assert "rid-A" in scheduler.pending_restore_registry
    assert "conv-1" in scheduler.pending_restore_registry
    assert (
        scheduler.pending_restore_registry["rid-A"]
        is scheduler.pending_restore_registry["conv-1"]
    )
    assert scheduler.pending_restore_registry["rid-A"].fill_ids == [1, 2, 3]


def test_stage_when_rid_equals_conversation_id_uses_single_key():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()

    scheduler._stage_pending_restore(
        rid="same-id",
        conversation_id="same-id",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=None,
    )

    assert list(scheduler.pending_restore_registry.keys()) == ["same-id"]


def test_stage_evicts_oldest_when_over_capacity():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()

    for i in range(PENDING_RESTORE_REGISTRY_MAX + 5):
        scheduler._stage_pending_restore(
            rid=f"rid-{i}",
            conversation_id=f"rid-{i}",
            conv_states=conv_states,
            temporal_states=temporal_states,
            fill_ids=[i],
        )

    assert len(scheduler.pending_restore_registry) == PENDING_RESTORE_REGISTRY_MAX
    # Oldest five should have been evicted
    for i in range(5):
        assert f"rid-{i}" not in scheduler.pending_restore_registry
    # Newest entry should be present
    assert (
        f"rid-{PENDING_RESTORE_REGISTRY_MAX + 4}" in scheduler.pending_restore_registry
    )


def test_stage_replaces_existing_entry_for_same_rid():
    scheduler, _ = _build_scheduler()
    conv_states_old, temporal_states_old = _make_state(1.0)
    conv_states_new, temporal_states_new = _make_state(2.0)

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=conv_states_old,
        temporal_states=temporal_states_old,
        fill_ids=[1],
    )
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=conv_states_new,
        temporal_states=temporal_states_new,
        fill_ids=[2],
    )

    entry = scheduler.pending_restore_registry["rid-A"]
    assert entry.fill_ids == [2]
    assert torch.equal(entry.conv_states[0], conv_states_new[0])


def test_hydrate_misses_when_registry_empty():
    scheduler, _ = _build_scheduler()
    req = _make_req("rid-missing")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is False
    assert req.mamba_pool_idx is None
    assert scheduler.snapshot_manager.injected == []


def test_hydrate_hits_by_rid_and_attaches_pool_slot():
    scheduler, mamba_pool = _build_scheduler()
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1, 2, 3],
    )
    req = _make_req("rid-A", conversation_id="conv-1")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is True
    assert req.mamba_pool_idx is not None
    assert req.mamba_pool_idx.item() == 0
    assert len(scheduler.snapshot_manager.injected) == 1
    assert scheduler.snapshot_manager.injected[0]["idx"] == 0
    # Origin input ids are NOT modified — SSM state carries history.
    assert req.origin_input_ids == [10, 20, 30]


def test_hydrate_falls_back_to_conversation_id_when_rid_misses():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-original",
        conversation_id="conv-1",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1],
    )
    # Different rid but same conversation_id
    req = _make_req("rid-different", conversation_id="conv-1")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is True
    assert req.mamba_pool_idx is not None


def test_hydrate_skips_when_req_already_has_pool_slot():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="rid-A",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1],
    )
    existing_idx = torch.tensor(99, dtype=torch.int32)
    req = _make_req("rid-A", mamba_pool_idx=existing_idx)

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is False
    assert req.mamba_pool_idx is existing_idx
    assert scheduler.snapshot_manager.injected == []


def test_hydrate_returns_false_when_pool_exhausted():
    scheduler, mamba_pool = _build_scheduler()
    mamba_pool._next = mamba_pool._capacity  # full
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="rid-A",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1],
    )
    req = _make_req("rid-A")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is False
    assert req.mamba_pool_idx is None


def test_hydrate_consumption_keeps_entry_for_back_to_back_requests():
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="rid-A",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1],
    )

    req1 = _make_req("rid-A")
    assert scheduler._maybe_hydrate_from_pending_restore(req1) is True
    assert "rid-A" in scheduler.pending_restore_registry

    req2 = _make_req("rid-A")
    assert scheduler._maybe_hydrate_from_pending_restore(req2) is True
    assert req1.mamba_pool_idx.item() != req2.mamba_pool_idx.item()
    assert len(scheduler.snapshot_manager.injected) == 2


def test_pending_restore_entry_dataclass_fields():
    """Sanity check that the dataclass exposes the expected attributes."""
    entry = PendingRestoreEntry(
        conv_states=[torch.zeros(1)],
        temporal_states=torch.zeros(1),
        fill_ids=[1, 2],
        conversation_id="conv-x",
        timestamp=123.0,
    )
    assert entry.fill_ids == [1, 2]
    assert entry.conversation_id == "conv-x"
    assert entry.timestamp == 123.0
