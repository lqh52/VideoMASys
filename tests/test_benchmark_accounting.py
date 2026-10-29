"""Offline dollar guard checks; mocked transport never reads owner credentials."""

import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_accounting.py"
spec = importlib.util.spec_from_file_location("benchmark_accounting_fixture", SCRIPT)
accounting = importlib.util.module_from_spec(spec)
# Reuse the same scripts import path convention as the benchmark runners.
sys.path.insert(0, str(SCRIPT.parent))
spec.loader.exec_module(accounting)


def test_reservations_enforce_cap_and_reopening_cannot_raise_or_reset_it(tmp_path):
    path = tmp_path / "budget.sqlite3"
    budget = accounting.Budget(path, 0.032)
    budget.reserve("a")
    reopened = accounting.Budget(path, 4)
    reopened.reserve("b")
    with pytest.raises(accounting.BudgetExceeded):
        budget.reserve("c")
    result = reopened.snapshot()
    assert result["max_usd"] == result["charged_usd"] == 0.032
    assert result["remaining_usd"] == 0 and result["budget_exhausted"]
    assert result["by_case"]["a"]["reserved_usd"] == 0.016


def test_missing_and_invalid_usage_retain_reservations(tmp_path):
    budget = accounting.Budget(tmp_path / "budget.sqlite3")
    for usage in (None, {}, {"prompt_tokens": True, "completion_tokens": 1},
                  {"prompt_tokens": 20, "completion_tokens": 1,
                   "prompt_tokens_details": {"cached_tokens": 21}}):
        assert budget.settle(budget.reserve("a"), usage)["state"] == "ambiguous"
    assert budget.snapshot()["ambiguous_usd"] == 0.064


def test_complete_usage_uses_cached_pricing_and_never_underreports_an_overrun(tmp_path):
    budget = accounting.Budget(tmp_path / "budget.sqlite3")
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
             "prompt_tokens_details": {"cached_tokens": 800}}
    result = budget.settle(budget.reserve("a"), usage)
    assert result == {"charged_usd": 0.00032, "state": "reported",
                      "reservation_exceeded": False}
    budget.settle(budget.reserve("b"), {"prompt_tokens": 100_000, "completion_tokens": 0})
    assert budget.snapshot()["reported_usd"] == 0.04032
    assert budget.snapshot()["halted_reason"] == "request_exceeded_reservation"
    with pytest.raises(accounting.BudgetExceeded):
        accounting.Budget(budget.path).reserve("c")


def test_concurrent_reservations_are_atomic(tmp_path):
    budget = accounting.Budget(tmp_path / "budget.sqlite3", 0.16)
    def reserve(index):
        try:
            return budget.reserve(str(index))
        except accounting.BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets = list(pool.map(reserve, range(30)))
    assert len({ticket for ticket in tickets if ticket}) == 10
    assert budget.snapshot()["charged_usd"] == 0.16


def test_adapter_accounts_each_real_retry_and_preserves_provenance(tmp_path):
    budget = accounting.Budget(tmp_path / "budget.sqlite3")
    requests, admissions = [], []
    def respond(request):
        requests.append(request)
        assert budget.snapshot()["requests_reserved"] == len(requests)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"code": "server_error"}})
        return httpx.Response(200, json={
            "model": "gpt-4.1-mini-2025-04-14",
            "choices": [{"message": {"role": "assistant", "content": "Fixture answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22,
                      "prompt_tokens_details": {"cached_tokens": 10}},
        })
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        def callback():
            admissions.append(1)
        adapter = accounting.BenchmarkVLM(
            "sk-fixture-not-a-key", client=client, retry_backoff_seconds=0,
            budget=budget, case_id="a", before_request=callback,
        )
        try:
            assert adapter("Fixture question", [], 100) == "Fixture answer"
            assert adapter._before_request is callback
            assert len(admissions) == len(requests) == 2
            record = adapter.calls[0]
            assert len(record["request_sha256"]) == 64
            assert all(item["request_attempted"] for item in record["attempts"])
            assert [item["budget_settlement"]["state"] for item in record["attempts"]] == [
                "ambiguous", "reported",
            ]
            assert "sk-fixture-not-a-key" not in json.dumps(record)
        finally:
            adapter.close()
    result = budget.snapshot()
    assert result["requests_reserved"] == 2
    assert result["charged_usd"] == pytest.approx(0.0160082)
    assert result["reserved_usd"] == 0


def test_adapter_denied_retry_is_not_an_http_attempt_or_an_extra_charge(tmp_path):
    from videomasys.zerogpu.remote import HostedVLMError

    budget = accounting.Budget(tmp_path / "budget.sqlite3", 0.016)
    requests = []
    def unavailable(request):
        requests.append(request)
        return httpx.Response(503, json={})
    with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
        adapter = accounting.BenchmarkVLM(
            "sk-fixture-not-a-key", client=client, retry_backoff_seconds=0,
            budget=budget, case_id="a",
        )
        try:
            with pytest.raises(HostedVLMError) as failure:
                adapter("Fixture question", [], 100)
            assert failure.value.code == "admission_denied"
            assert adapter.budget_exhausted and len(requests) == 1
            assert [item["request_attempted"] for item in adapter.calls[0]["attempts"]] == [
                True, False,
            ]
            assert budget.snapshot()["requests_reserved"] == 1
            assert budget.snapshot()["charged_usd"] == 0.016
        finally:
            adapter.close()


def test_caller_timeout_does_not_settle_or_restore_a_running_worker(tmp_path):
    from videomasys.zerogpu.remote import HostedVLMError

    budget = accounting.Budget(tmp_path / "budget.sqlite3")
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    class NotifyingBudget(accounting.Budget):
        def settle(self, ticket, usage):
            result = super().settle(ticket, usage)
            settled.set()
            return result
    budget = NotifyingBudget(budget.path)
    def block(request):
        entered.set()
        assert release.wait(5)
        return httpx.Response(503, json={})
    with httpx.Client(transport=httpx.MockTransport(block)) as client:
        def callback():
            pass
        adapter = accounting.BenchmarkVLM(
            "sk-fixture-not-a-key", client=client, timeout_seconds=0.2,
            budget=budget, case_id="a", before_request=callback,
        )
        try:
            with pytest.raises(HostedVLMError) as failure:
                adapter("Fixture question", [], 100)
            assert failure.value.code == "timeout" and entered.is_set()
            assert budget.snapshot()["reserved_usd"] == 0.016
            assert adapter._before_request is not callback
        finally:
            release.set()
            assert settled.wait(5)
            # Taking the existing slot proves the complete worker cleanup exited.
            assert adapter._worker_slot.acquire(timeout=5)
            adapter._worker_slot.release()
            assert adapter._before_request is callback
            assert budget.snapshot()["ambiguous_usd"] == 0.016
            adapter.close()
