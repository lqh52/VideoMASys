"""Durable, conservative local accounting for the pinned benchmark model.

This guards this ledger's requests, not unrelated account spending or an invoice.
Prices checked 2026-09-08: https://developers.openai.com/api/docs/models/gpt-4.1-mini
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from evaluate_prompt_tuning import ProvenanceVLM

from videomasys.zerogpu.remote import MAX_CONTEXT_BOUND, MAX_OUTPUT_TOKENS

# Integer units of $0.0000001 make all three per-token prices exact.
UNITS_PER_USD = 10_000_000
RESERVATION_UNITS = 160_000
RESERVATION_USD = RESERVATION_UNITS / UNITS_PER_USD
DEFAULT_MAX_USD = 4.0


class BudgetExceeded(RuntimeError):
    """No more requests may be admitted under this benchmark ledger."""


def _count(value):
    return type(value) is int and 0 <= value <= 1_000_000_000


def _cost(usage):
    """Return cost and completeness; malformed/missing usage never earns a refund."""
    if not isinstance(usage, dict):
        return RESERVATION_UNITS, False
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
    complete = (
        _count(prompt) and _count(output) and _count(cached) and cached <= prompt
        and ("total_tokens" not in usage or (
            _count(usage["total_tokens"]) and usage["total_tokens"] == prompt + output
        ))
    )
    if complete:
        return 4 * (prompt - cached) + cached + 16 * output, True
    # Preserve any known larger amount even when the rest of usage is malformed.
    known = (4 * prompt if _count(prompt) else 0) + (16 * output if _count(output) else 0)
    return max(RESERVATION_UNITS, known), False


class Budget:
    """SQLite admission shared safely by threads/processes; reopening never resets it."""

    def __init__(self, path, max_usd=DEFAULT_MAX_USD):
        try:
            amount = Decimal(str(max_usd)) * UNITS_PER_USD
            if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
                raise ValueError
            units = int(amount)
            if units > 1_000_000 * UNITS_PER_USD:
                raise ValueError
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Budget must be a positive USD amount with at most seven decimals.") \
                from None
        if 4 * MAX_CONTEXT_BOUND + 16 * MAX_OUTPUT_TOKENS > RESERVATION_UNITS:
            raise ValueError("The adapter limits exceed the fixed request reservation.")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("The budget ledger must not be a symbolic link.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, "
                       "max_units INTEGER NOT NULL, halted_reason TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (ticket TEXT PRIMARY KEY, "
                       "case_id TEXT NOT NULL, charged_units INTEGER NOT NULL, "
                       "state TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?, NULL)", (units,))
            # The existing cap, including a smaller one, is always authoritative.
            self.max_usd = db.execute("SELECT max_units FROM budget WHERE id=1").fetchone()[0] \
                / UNITS_PER_USD

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, case_id):
        if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 200:
            raise ValueError("A nonempty bounded case ID is required.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            maximum, halted = db.execute(
                "SELECT max_units, halted_reason FROM budget WHERE id=1"
            ).fetchone()
            charged = db.execute(
                "SELECT COALESCE(SUM(charged_units), 0) FROM requests"
            ).fetchone()[0]
            if halted or charged + RESERVATION_UNITS > maximum:
                raise BudgetExceeded("The benchmark dollar allowance is exhausted or halted.")
            ticket = uuid4().hex
            db.execute("INSERT INTO requests VALUES (?, ?, ?, 'reserved')",
                       (ticket, case_id, RESERVATION_UNITS))
        return ticket

    def settle(self, ticket, usage):
        charged, complete = _cost(usage)
        state = "reported" if complete else "ambiguous"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM requests WHERE ticket=?", (ticket,)).fetchone()
            if row is None or row[0] != "reserved":
                raise ValueError("Settlement requires an outstanding reservation ticket.")
            db.execute("UPDATE requests SET charged_units=?, state=? WHERE ticket=?",
                       (charged, state, ticket))
            if charged > RESERVATION_UNITS:
                db.execute("UPDATE budget SET halted_reason='request_exceeded_reservation' "
                           "WHERE id=1")
        return {"charged_usd": charged / UNITS_PER_USD, "state": state,
                "reservation_exceeded": charged > RESERVATION_UNITS}

    def snapshot(self):
        with self._connect() as db:
            db.execute("BEGIN")
            maximum, halted = db.execute(
                "SELECT max_units, halted_reason FROM budget WHERE id=1"
            ).fetchone()
            rows = db.execute("SELECT case_id, state, COUNT(*), SUM(charged_units) "
                              "FROM requests GROUP BY case_id, state").fetchall()
        def empty():
            return {"requests_reserved": 0, "charged_usd": 0, "reported_usd": 0,
                    "reserved_usd": 0, "ambiguous_usd": 0}
        totals, by_case = empty(), {}
        for case_id, state, count, charged in rows:
            for target in (totals, by_case.setdefault(case_id, empty())):
                target["requests_reserved"] += count
                target["charged_usd"] += charged
                target[f"{state}_usd"] += charged
        charged = totals["charged_usd"]
        for target in (totals, *by_case.values()):
            for key in ("charged_usd", "reported_usd", "reserved_usd", "ambiguous_usd"):
                target[key] /= UNITS_PER_USD
        return {**totals, "max_usd": maximum / UNITS_PER_USD,
                "remaining_usd": max(0, maximum - charged) / UNITS_PER_USD,
                "budget_exhausted": bool(halted or charged + RESERVATION_UNITS > maximum),
                "halted_reason": halted, "by_case": by_case,
                "scope": "This persistent benchmark ledger; not an account billing cap."}


class BenchmarkVLM(ProvenanceVLM):
    """Add accounting to existing admission, transport, retry and provenance hooks."""

    def __init__(self, *args, budget, case_id, **kwargs):
        self.budget = budget
        self.case_id = case_id
        self.budget_exhausted = False
        super().__init__(*args, **kwargs)

    def _perform(self, payload, deadline, record):
        original_admission = self._before_request
        pending = []

        def settle_finished():
            while pending:
                ticket, attempt = pending[0]
                result = self.budget.settle(ticket, attempt.get("usage"))
                pending.pop(0)
                attempt["budget_settlement"] = result
                if result["reservation_exceeded"]:
                    self.budget_exhausted = record["budget_exhausted"] = True

        def admit():
            # A previous attempt has fully exited before this retry callback.
            # Its reported overrun must block the retry as well as later calls.
            settle_finished()
            if original_admission is not None:
                original_admission()
            try:
                ticket = self.budget.reserve(self.case_id)
            except BudgetExceeded:
                self.budget_exhausted = record["budget_exhausted"] = True
                raise
            # HostedVLM appends this exact attempt before calling admission.
            attempt = record["attempts"][-1]
            attempt.update(budget_ticket=ticket, reserved_usd=RESERVATION_USD)
            pending.append((ticket, attempt))

        self._before_request = admit
        try:
            return super()._perform(payload, deadline, record)
        finally:
            # This finally runs in the transport worker, not a timed-out caller.
            # Unknown billing retains the reservation, including ambiguous failures.
            try:
                settle_finished()
            finally:
                self._before_request = original_admission
