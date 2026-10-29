"""Transactional request allowances, not a provider-side dollar spending cap.

The ledger survives process restarts only while its filesystem survives. A free
Space's ephemeral disk may be replaced; do not present this as an account cap.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class UsageLimitError(RuntimeError):
    pass


class UsageGuard:
    def __init__(self, path, *, max_runs=5, max_requests=120, max_requests_per_run=32):
        for value in (max_runs, max_requests, max_requests_per_run):
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError("Demo allowances must be integers between 1 and 10000.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_runs = max_runs
        self.max_requests = max_requests
        self.max_requests_per_run = max_requests_per_run
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY, "
                "runs INTEGER NOT NULL, requests INTEGER NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO usage VALUES (1, 0, 0)")

    @contextmanager
    def _connect(self):
        # sqlite3's own context manager commits/rolls back; it does not close.
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _reserve(self, column, limit):
        # column is an internal constant, never a UI or environment input.
        if column not in {"runs", "requests"}:
            raise ValueError("Unknown allowance counter.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(f"SELECT {column} FROM usage WHERE id=1").fetchone()[0]
            if count >= limit:
                raise UsageLimitError(
                    "The demo usage allowance has been reached. "
                    "Ask the owner to review usage and extend the allowance."
                )
            db.execute(f"UPDATE usage SET {column}={column}+1 WHERE id=1")

    def start_run(self):
        # Admit atomically against both counters; an exhausted request budget
        # must not consume another run that cannot submit even one request.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            runs, requests = db.execute("SELECT runs, requests FROM usage WHERE id=1").fetchone()
            if runs >= self.max_runs or requests >= self.max_requests:
                raise UsageLimitError(
                    "The demo usage allowance has been reached. "
                    "Ask the owner to review usage and extend the allowance."
                )
            db.execute("UPDATE usage SET runs=runs+1 WHERE id=1")
        return RunAllowance(self)

    def snapshot(self):
        with self._connect() as db:
            runs, requests = db.execute("SELECT runs, requests FROM usage WHERE id=1").fetchone()
        return {
            "runs_used": runs,
            "request_attempts_used": requests,
            "max_runs": self.max_runs,
            "max_request_attempts": self.max_requests,
            "scope": "current ledger; not an account billing cap",
        }


class RunAllowance:
    def __init__(self, guard):
        self.guard = guard
        self.requests = 0
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.closed = False

    def before_request(self):
        with self.lock:
            if self.closed:
                raise UsageLimitError("This analysis has ended. Start a new analysis.")
            if time.monotonic() - self.started > 600:
                raise UsageLimitError("Analysis time limit reached. Try a narrower question.")
            if self.requests >= self.guard.max_requests_per_run:
                raise UsageLimitError("Analysis request limit reached. Try a narrower question.")
            # Reserve before HTTP submission. Failed/ambiguous calls are not refunded.
            self.guard._reserve("requests", self.guard.max_requests)
            self.requests += 1

    def close(self):
        with self.lock:
            self.closed = True
