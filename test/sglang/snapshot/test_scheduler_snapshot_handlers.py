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
