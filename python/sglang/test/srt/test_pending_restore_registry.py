"""Unit tests for the pending-restore registry on the Scheduler.

The registry stages loaded snapshot state for a future incoming request when
/restore_snapshot is called after the originating Req has finished. Tests
exercise the producer (_stage_pending_restore), capacity bound + LRU eviction,
key aliasing (rid + conversation_id), and the consumer hydration semantics
(_maybe_hydrate_from_pending_restore) including pool allocation and Req-state
mutation.

The registry uses a canonical OrderedDict (rid → entry) plus a separate alias
map (conv_id → canonical rid). Tests validate that replacement, eviction, and
lookup are atomic across all keys mapping to a given logical entry.
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


class _FailingSnapshotManager:
    """Raises RuntimeError on every inject_state_to_pool call."""

    def inject_state_to_pool(self, conv_states, temporal_states, mamba_pool, idx):
        raise RuntimeError("inject_state_to_pool failure")


def _build_scheduler():
    """Build a bare Scheduler with only the fields the registry methods touch."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pending_restore_registry = OrderedDict()
    scheduler.pending_restore_aliases = {}
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


# ---------------------------------------------------------------------------
# Producer tests — _stage_pending_restore
# ---------------------------------------------------------------------------


def test_stage_indexes_by_rid_and_conversation_id():
    """Canonical entry keyed by rid; conversation_id stored as alias."""
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1, 2, 3],
    )

    # Canonical entry is in the registry, not under the alias key
    assert "rid-A" in scheduler.pending_restore_registry
    assert "conv-1" not in scheduler.pending_restore_registry
    # Alias maps conv_id → canonical rid
    assert scheduler.pending_restore_aliases["conv-1"] == "rid-A"
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
    assert scheduler.pending_restore_aliases == {}


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


# ---------------------------------------------------------------------------
# Finding 1 tests — multi-key replacement, cross-rid replacement, eviction
# ---------------------------------------------------------------------------


def test_multi_key_replacement_same_rid_different_conv():
    """Stage rid-A/conv-1 then rid-A/conv-2: old alias is gone, new entry
    reachable via both rid-A and conv-2."""
    scheduler, _ = _build_scheduler()
    cs1, ts1 = _make_state(1.0)
    cs2, ts2 = _make_state(2.0)

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs1,
        temporal_states=ts1,
        fill_ids=[1],
    )
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-2",
        conv_states=cs2,
        temporal_states=ts2,
        fill_ids=[2],
    )

    # Old alias conv-1 must be gone
    assert "conv-1" not in scheduler.pending_restore_aliases
    assert "conv-1" not in scheduler.pending_restore_registry
    # New alias conv-2 points to canonical rid-A
    assert scheduler.pending_restore_aliases["conv-2"] == "rid-A"
    # Canonical entry holds the new data
    assert scheduler.pending_restore_registry["rid-A"].fill_ids == [2]
    assert torch.equal(
        scheduler.pending_restore_registry["rid-A"].conv_states[0], cs2[0]
    )


def test_cross_rid_same_conv_replacement():
    """Stage rid-A/conv-1 then rid-B/conv-1: rid-A removed, rid-B and conv-1
    both resolve to entry-2."""
    scheduler, _ = _build_scheduler()
    cs1, ts1 = _make_state(1.0)
    cs2, ts2 = _make_state(2.0)

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs1,
        temporal_states=ts1,
        fill_ids=[1],
    )
    scheduler._stage_pending_restore(
        rid="rid-B",
        conversation_id="conv-1",
        conv_states=cs2,
        temporal_states=ts2,
        fill_ids=[2],
    )

    # rid-A should be fully evicted (conv-1 collided as canonical with another entry)
    assert "rid-A" not in scheduler.pending_restore_registry
    # rid-B is the new canonical
    assert "rid-B" in scheduler.pending_restore_registry
    assert scheduler.pending_restore_registry["rid-B"].fill_ids == [2]
    # conv-1 alias points to rid-B
    assert scheduler.pending_restore_aliases.get("conv-1") == "rid-B"


def test_multi_key_eviction_with_distinct_keys():
    """Stage MAX+5 pairs all with conv != rid.
    Assert canonical size == MAX and alias map contains only surviving aliases."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()
    n = PENDING_RESTORE_REGISTRY_MAX + 5

    for i in range(n):
        scheduler._stage_pending_restore(
            rid=f"rid-{i}",
            conversation_id=f"conv-{i}",
            conv_states=cs,
            temporal_states=ts,
            fill_ids=[i],
        )

    # Canonical size bounded to MAX
    assert len(scheduler.pending_restore_registry) == PENDING_RESTORE_REGISTRY_MAX

    # Oldest 5 canonicals evicted
    for i in range(5):
        assert f"rid-{i}" not in scheduler.pending_restore_registry
        # Their aliases should also be gone
        assert f"conv-{i}" not in scheduler.pending_restore_aliases

    # Every alias in the map points to a surviving canonical
    for alias, canonical in scheduler.pending_restore_aliases.items():
        assert canonical in scheduler.pending_restore_registry

    # Latest entry is present in both structures
    assert f"rid-{n - 1}" in scheduler.pending_restore_registry
    assert scheduler.pending_restore_aliases[f"conv-{n - 1}"] == f"rid-{n - 1}"


# ---------------------------------------------------------------------------
# Finding 3 + Case B tests — alias remap evicts old canonical
# ---------------------------------------------------------------------------


def test_stage_evicts_old_canonical_when_alias_remapped():
    """Finding 3: staging rid-B/conv-1 after rid-A/conv-1 evicts rid-A
    and all its aliases. The old canonical must not remain reachable."""
    scheduler, _ = _build_scheduler()
    cs1, ts1 = _make_state(1.0)
    cs2, ts2 = _make_state(2.0)

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs1,
        temporal_states=ts1,
        fill_ids=[1],
    )
    scheduler._stage_pending_restore(
        rid="rid-B",
        conversation_id="conv-1",
        conv_states=cs2,
        temporal_states=ts2,
        fill_ids=[2],
    )

    # Old canonical rid-A must be gone — alias was remapped
    assert "rid-A" not in scheduler.pending_restore_registry
    # No leftover alias for rid-A
    assert "rid-A" not in scheduler.pending_restore_aliases
    # conv-1 alias now points at rid-B
    assert scheduler.pending_restore_aliases.get("conv-1") == "rid-B"
    # Only one canonical entry remains
    assert len(scheduler.pending_restore_registry) == 1
    assert scheduler.pending_restore_registry["rid-B"].fill_ids == [2]


def test_stage_evicts_old_canonical_when_rid_was_previously_alias():
    """Case B: staging rid-R/conv-X after rid-P/conv-R evicts rid-P.
    rid-R was previously an alias for rid-P; after staging as canonical
    there must be no orphan alias rid-R→rid-P remaining."""
    scheduler, _ = _build_scheduler()
    cs1, ts1 = _make_state(1.0)
    cs2, ts2 = _make_state(2.0)

    # First: rid-P is canonical, conv-R alias → rid-P (so rid-R is an alias)
    scheduler._stage_pending_restore(
        rid="rid-P",
        conversation_id="rid-R",
        conv_states=cs1,
        temporal_states=ts1,
        fill_ids=[1],
    )
    # Second: rid-R is now the new canonical with its own alias conv-X
    scheduler._stage_pending_restore(
        rid="rid-R",
        conversation_id="conv-X",
        conv_states=cs2,
        temporal_states=ts2,
        fill_ids=[2],
    )

    # Old canonical rid-P must be gone (rid-R was its alias, now promoted)
    assert "rid-P" not in scheduler.pending_restore_registry
    # No orphan alias rid-R→rid-P remains
    assert "rid-R" not in scheduler.pending_restore_aliases
    # Only rid-R canonical exists
    assert len(scheduler.pending_restore_registry) == 1
    assert scheduler.pending_restore_registry["rid-R"].fill_ids == [2]
    # conv-X alias points to rid-R
    assert scheduler.pending_restore_aliases.get("conv-X") == "rid-R"


def test_orphan_alias_does_not_resurrect_on_eviction():
    """Evict an entry whose alias was previously remapped.
    Assert no stale alias resurrects the old canonical on subsequent hydration."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()

    # Fill registry with MAX entries, each with distinct rid and conv_id
    for i in range(PENDING_RESTORE_REGISTRY_MAX):
        scheduler._stage_pending_restore(
            rid=f"rid-{i}",
            conversation_id=f"conv-{i}",
            conv_states=cs,
            temporal_states=ts,
            fill_ids=[i],
        )

    # Remap an existing alias → new canonical (Finding 3 scenario)
    # rid-0/conv-0 exists; stage rid-NEW/conv-0 to force remap
    scheduler._stage_pending_restore(
        rid="rid-NEW",
        conversation_id="conv-0",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[999],
    )

    # Old canonical rid-0 must be gone
    assert "rid-0" not in scheduler.pending_restore_registry
    # conv-0 alias points at rid-NEW
    assert scheduler.pending_restore_aliases.get("conv-0") == "rid-NEW"

    # Hydration via old rid-0 must be a true miss (None), not resurrect stale state
    req = _make_req("rid-0")
    result = scheduler._maybe_hydrate_from_pending_restore(req)
    assert result is None

    # Hydration via alias conv-0 hits the new canonical rid-NEW
    req2 = _make_req("rid-different", conversation_id="conv-0")
    result2 = scheduler._maybe_hydrate_from_pending_restore(req2)
    assert result2 is True
    assert req2.mamba_pool_idx is not None


# ---------------------------------------------------------------------------
# Consumer tests — _maybe_hydrate_from_pending_restore
# ---------------------------------------------------------------------------


def test_hydrate_misses_when_registry_empty():
    """Tri-state: true miss returns None."""
    scheduler, _ = _build_scheduler()
    req = _make_req("rid-missing")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is None
    assert req.mamba_pool_idx is None
    assert scheduler.snapshot_manager.injected == []


def test_hydrate_hits_by_rid_and_attaches_pool_slot():
    """Tri-state: successful hydration returns True."""
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
    """Tri-state: alias resolution finds the canonical entry, returns True."""
    scheduler, _ = _build_scheduler()
    conv_states, temporal_states = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-original",
        conversation_id="conv-1",
        conv_states=conv_states,
        temporal_states=temporal_states,
        fill_ids=[1],
    )
    # Different rid but same conversation_id — resolves via alias
    req = _make_req("rid-different", conversation_id="conv-1")

    hit = scheduler._maybe_hydrate_from_pending_restore(req)

    assert hit is True
    assert req.mamba_pool_idx is not None


def test_hydrate_skips_when_req_already_has_pool_slot():
    """Tri-state: already-hydrated request returns None (true miss)."""
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

    assert hit is None
    assert req.mamba_pool_idx is existing_idx
    assert scheduler.snapshot_manager.injected == []


def test_hydrate_returns_false_when_pool_exhausted():
    """Tri-state: staged entry, pool full, returns False, no pool slot leaked."""
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
    """Tri-state: back-to-back hydration returns True each time, fresh slots."""
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


# ---------------------------------------------------------------------------
# Finding 2 tests — tri-state hydration (true miss, success, exhausted, error)
# ---------------------------------------------------------------------------


def test_tri_state_true_miss_empty_registry():
    """Empty registry returns None (true miss)."""
    scheduler, _ = _build_scheduler()
    req = _make_req("rid-absent")

    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is None


def test_tri_state_success_returns_true():
    """Staged entry with available pool returns True and allocates slot."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )
    req = _make_req("rid-A")

    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is True
    assert req.mamba_pool_idx is not None
    assert len(scheduler.snapshot_manager.injected) == 1


def test_tri_state_pool_exhausted_returns_false_no_leak():
    """Pool full: returns False, no slot was allocated (so nothing to leak)."""
    scheduler, mamba_pool = _build_scheduler()
    mamba_pool._next = mamba_pool._capacity  # pool already full
    cs, ts = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="rid-A",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )
    req = _make_req("rid-A")

    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is False
    assert req.mamba_pool_idx is None
    # alloc() returned None, so no inject was attempted
    assert scheduler.snapshot_manager.injected == []


def test_tri_state_inject_error_returns_false_and_frees_slot():
    """inject_state_to_pool raises: returns False, allocated pool slot freed."""
    scheduler, mamba_pool = _build_scheduler()
    scheduler.snapshot_manager = _FailingSnapshotManager()
    cs, ts = _make_state()
    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="rid-A",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )
    req = _make_req("rid-A")

    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is False
    assert req.mamba_pool_idx is None
    # The allocated slot (idx 0) should have been freed during cleanup
    assert len(mamba_pool._freed) == 1


# ---------------------------------------------------------------------------
# Finding 4 tests — _clear_pending_restore
# ---------------------------------------------------------------------------


def test_clear_pending_restore_removes_canonical_and_aliases():
    """Stage rid-A/conv-1, clear via rid, assert registry and aliases are empty."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )

    scheduler._clear_pending_restore(rid="rid-A", conversation_id="conv-1")

    assert "rid-A" not in scheduler.pending_restore_registry
    assert "conv-1" not in scheduler.pending_restore_aliases
    assert len(scheduler.pending_restore_registry) == 0
    assert len(scheduler.pending_restore_aliases) == 0


def test_clear_pending_restore_resolves_via_alias():
    """Stage rid-A/conv-1, clear via unmatched rid + alias conv-1; entry gone."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )

    scheduler._clear_pending_restore(rid="rid-other", conversation_id="conv-1")

    assert "rid-A" not in scheduler.pending_restore_registry
    assert "conv-1" not in scheduler.pending_restore_aliases
    assert len(scheduler.pending_restore_registry) == 0


def test_clear_pending_restore_no_op_on_miss():
    """Empty registry: call raises no exception and leaves state unchanged."""
    scheduler, _ = _build_scheduler()

    scheduler._clear_pending_restore("rid-absent", "conv-absent")

    assert len(scheduler.pending_restore_registry) == 0
    assert len(scheduler.pending_restore_aliases) == 0


# ---------------------------------------------------------------------------
# Finding 5 tests — conv_id conflict resolution during hydration
# ---------------------------------------------------------------------------


def test_hydrate_warns_and_overwrites_on_conv_mismatch(caplog):
    """stage rid-A/conv-X, hydrate with req.conversation_id=conv-Y.
    Warning logged, req.conversation_id overwritten to conv-X, hydration ok."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-X",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )

    req = _make_req("rid-A", conversation_id="conv-Y")
    import logging

    caplog.set_level(logging.WARNING)

    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is True
    assert req.conversation_id == "conv-X"
    # Warning was emitted for the conflict
    assert any(
        "differs from" in record.message and "conv-Y" in record.message
        for record in caplog.records
    )


def test_hydrate_alias_path_does_not_warn_on_conv_match():
    """stage rid-A/conv-1, hydrate with different rid + matching conv_id=conv-1.
    No warning needed — alias path by construction matches the correct entry."""
    scheduler, _ = _build_scheduler()
    cs, ts = _make_state()

    scheduler._stage_pending_restore(
        rid="rid-A",
        conversation_id="conv-1",
        conv_states=cs,
        temporal_states=ts,
        fill_ids=[1],
    )

    req = _make_req("rid-different", conversation_id="conv-1")
    result = scheduler._maybe_hydrate_from_pending_restore(req)

    assert result is True
    assert req.conversation_id == "conv-1"
    assert req.mamba_pool_idx is not None


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


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
