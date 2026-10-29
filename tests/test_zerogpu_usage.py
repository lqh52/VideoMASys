"""Local-only allowance accounting: no HTTP requests or hosted state changes."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from videomasys.zerogpu import usage
from videomasys.zerogpu.usage import UsageGuard, UsageLimitError


@pytest.mark.parametrize("field", ["max_runs", "max_requests", "max_requests_per_run"])
@pytest.mark.parametrize("value", [0, -1, 10001, True, 1.5, "2"])
def test_allowances_require_bounded_positive_integers(tmp_path, field, value):
    with pytest.raises(ValueError, match="integers"):
        UsageGuard(tmp_path / "usage.sqlite3", **{field: value})
    assert not (tmp_path / "usage.sqlite3").exists()


def test_reopen_preserves_counts_and_failures_are_not_refunded(tmp_path):
    path = tmp_path / "usage.sqlite3"
    guard = UsageGuard(path, max_runs=2, max_requests=3)
    allowance = guard.start_run()
    allowance.before_request()
    # Simulate a failure after reservation; no HTTP is issued by this fixture.
    with pytest.raises(RuntimeError, match="fixture failure"):
        raise RuntimeError("fixture failure after request reservation")
    allowance.close()
    reopened = UsageGuard(path, max_runs=2, max_requests=3)
    assert reopened.snapshot()["runs_used"] == 1
    assert reopened.snapshot()["request_attempts_used"] == 1
    second = reopened.start_run()
    second.before_request()
    second.close()
    with pytest.raises(UsageLimitError):
        reopened.start_run()
    assert guard.snapshot()["runs_used"] == 2
    assert guard.snapshot()["request_attempts_used"] == 2


def test_exhausted_request_budget_does_not_admit_or_charge_another_run(tmp_path):
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=5, max_requests=1)
    first = guard.start_run()
    first.before_request()
    with pytest.raises(UsageLimitError):
        guard.start_run()
    assert guard.snapshot()["runs_used"] == 1
    assert guard.snapshot()["request_attempts_used"] == 1


def test_per_run_limit_does_not_charge_rejected_requests(tmp_path):
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_requests_per_run=2)
    first = guard.start_run()
    first.before_request()
    first.before_request()
    with pytest.raises(UsageLimitError, match="Analysis request limit"):
        first.before_request()
    assert first.requests == 2
    second = guard.start_run()
    second.before_request()
    assert guard.snapshot()["request_attempts_used"] == 3


def test_closed_or_expired_allowance_never_reserves(tmp_path, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(usage.time, "monotonic", lambda: clock["now"])
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    closed = guard.start_run()
    closed.close()
    closed.close()
    with pytest.raises(UsageLimitError, match="ended"):
        closed.before_request()
    expired = guard.start_run()
    clock["now"] = 701.0
    with pytest.raises(UsageLimitError, match="time limit"):
        expired.before_request()
    assert guard.snapshot()["request_attempts_used"] == 0


def test_cross_instance_concurrent_admission_never_exceeds_shared_limit(tmp_path):
    path = tmp_path / "usage.sqlite3"
    guards = [UsageGuard(path, max_runs=7) for _ in range(4)]

    def start(index):
        try:
            guards[index % len(guards)].start_run().close()
            return True
        except UsageLimitError:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(start, range(24)))
    assert sum(results) == 7
    assert guards[0].snapshot()["runs_used"] == 7


def test_concurrent_requests_share_database_ceiling_across_runs(tmp_path):
    path = tmp_path / "usage.sqlite3"
    guards = [UsageGuard(path, max_requests=9) for _ in range(4)]
    runs = [guard.start_run() for guard in guards]

    def reserve(index):
        try:
            runs[index % len(runs)].before_request()
            return True
        except UsageLimitError:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(24)))
    assert sum(results) == sum(run.requests for run in runs) == 9
    assert guards[0].snapshot()["request_attempts_used"] == 9


def test_concurrent_requests_share_the_same_per_run_lock(tmp_path):
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_requests_per_run=7)
    run = guard.start_run()

    def reserve(_index):
        try:
            run.before_request()
            return True
        except UsageLimitError:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(24)))
    assert sum(results) == run.requests == 7
    assert guard.snapshot()["request_attempts_used"] == 7


def test_sqlite_connections_close_on_success_and_rejected_reservation(tmp_path, monkeypatch):
    connections = []
    original_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def close(self):
            self.was_closed = True
            super().close()

    def connect(*args, **kwargs):
        connection = original_connect(*args, factory=TrackedConnection, **kwargs)
        connection.was_closed = False
        connections.append(connection)
        return connection

    monkeypatch.setattr(usage.sqlite3, "connect", connect)
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1, max_requests=1)
    run = guard.start_run()
    run.before_request()
    with pytest.raises(UsageLimitError):
        guard.start_run()
    with pytest.raises(UsageLimitError):
        run.before_request()
    guard.snapshot()
    assert len(connections) >= 6
    assert all(connection.was_closed for connection in connections)


def test_failed_reservation_is_fail_closed_and_does_not_increment_run_count(tmp_path, monkeypatch):
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    run = guard.start_run()

    def unavailable(*_args):
        raise sqlite3.OperationalError("isolated unavailable database")

    monkeypatch.setattr(guard, "_reserve", unavailable)
    with pytest.raises(sqlite3.OperationalError):
        run.before_request()
    assert run.requests == 0
    assert guard.snapshot()["request_attempts_used"] == 0


def test_counter_name_is_not_arbitrary_sql(tmp_path):
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    with pytest.raises(ValueError, match="Unknown allowance"):
        guard._reserve("requests; DROP TABLE usage", 10)
    assert guard.snapshot()["request_attempts_used"] == 0
