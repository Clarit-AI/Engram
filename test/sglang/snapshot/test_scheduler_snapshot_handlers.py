from types import SimpleNamespace

from sglang.srt.managers.scheduler_snapshot_handlers import _find_request_by_rid


class FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs

    def is_empty(self):
        return len(self.reqs) == 0


def req(rid):
    return SimpleNamespace(rid=rid)


def scheduler_with(**kwargs):
    defaults = {
        "running_batch": FakeBatch([]),
        "cur_batch": None,
        "last_batch": None,
        "waiting_queue": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_find_request_in_running_batch():
    target = req("target")
    scheduler = scheduler_with(running_batch=FakeBatch([target]))

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_in_waiting_queue():
    target = req("target")
    scheduler = scheduler_with(waiting_queue=[target])

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_in_cur_batch_only():
    target = req("target")
    scheduler = scheduler_with(cur_batch=FakeBatch([target]))

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_in_last_batch_only():
    target = req("target")
    scheduler = scheduler_with(last_batch=FakeBatch([target]))

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_deduplicates_batch_aliases():
    target = req("target")
    shared = FakeBatch([target])
    scheduler = scheduler_with(
        running_batch=shared, cur_batch=shared, last_batch=shared
    )

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_returns_none_when_absent():
    scheduler = scheduler_with(
        running_batch=FakeBatch([req("running")]),
        cur_batch=FakeBatch([req("cur")]),
        last_batch=FakeBatch([req("last")]),
        waiting_queue=[req("waiting")],
    )

    assert _find_request_by_rid(scheduler, "target") is None


# --- Additional edge-case and regression tests ---


def test_find_request_scheduler_without_cur_batch_attr():
    """Scheduler that never set cur_batch/last_batch — getattr must not raise."""
    target = req("target")
    # SimpleNamespace with only running_batch and waiting_queue, no cur_batch/last_batch
    scheduler = SimpleNamespace(
        running_batch=FakeBatch([target]),
        waiting_queue=[],
    )

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_scheduler_without_any_batch_attr():
    """Scheduler missing all optional batch attrs — falls through to waiting_queue."""
    target = req("target")
    scheduler = SimpleNamespace(
        running_batch=None,
        waiting_queue=[target],
    )

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_running_batch_takes_priority_over_cur_batch():
    """When same rid appears in running_batch and cur_batch, running_batch wins."""
    r_running = req("target")
    r_cur = req("target")
    scheduler = scheduler_with(
        running_batch=FakeBatch([r_running]),
        cur_batch=FakeBatch([r_cur]),
    )

    result = _find_request_by_rid(scheduler, "target")
    assert result is r_running


def test_find_request_cur_batch_takes_priority_over_last_batch():
    """When same rid appears in cur_batch and last_batch, cur_batch wins."""
    r_cur = req("target")
    r_last = req("target")
    scheduler = scheduler_with(
        cur_batch=FakeBatch([r_cur]),
        last_batch=FakeBatch([r_last]),
    )

    result = _find_request_by_rid(scheduler, "target")
    assert result is r_cur


def test_find_request_batches_before_waiting_queue():
    """Batch entries are returned before scanning waiting_queue."""
    r_batch = req("target")
    r_queue = req("target")
    scheduler = scheduler_with(
        cur_batch=FakeBatch([r_batch]),
        waiting_queue=[r_queue],
    )

    result = _find_request_by_rid(scheduler, "target")
    assert result is r_batch


def test_find_request_none_running_batch_falls_through():
    """None running_batch does not crash; cur_batch is still checked."""
    target = req("target")
    scheduler = scheduler_with(
        running_batch=None,
        cur_batch=FakeBatch([target]),
    )

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_multiple_reqs_in_batch():
    """Correct request is returned when a batch holds several entries."""
    r1, r2, r3 = req("a"), req("target"), req("b")
    scheduler = scheduler_with(running_batch=FakeBatch([r1, r2, r3]))

    assert _find_request_by_rid(scheduler, "target") is r2


def test_find_request_target_at_end_of_waiting_queue():
    """Search scans the entire waiting_queue, not just the first element."""
    others = [req(f"other-{i}") for i in range(5)]
    target = req("target")
    scheduler = scheduler_with(waiting_queue=others + [target])

    assert _find_request_by_rid(scheduler, "target") is target


def test_find_request_all_batches_empty_not_in_queue():
    """Empty batches and empty queue → None."""
    scheduler = scheduler_with(
        running_batch=FakeBatch([]),
        cur_batch=FakeBatch([]),
        last_batch=FakeBatch([]),
        waiting_queue=[],
    )

    assert _find_request_by_rid(scheduler, "target") is None


def test_find_request_deduplication_does_not_skip_distinct_batches():
    """Deduplication only skips the same object; distinct batches are both searched."""
    r_cur = req("in-cur")
    r_last = req("in-last")
    scheduler = scheduler_with(
        cur_batch=FakeBatch([r_cur]),
        last_batch=FakeBatch([r_last]),
    )

    assert _find_request_by_rid(scheduler, "in-cur") is r_cur
    assert _find_request_by_rid(scheduler, "in-last") is r_last
