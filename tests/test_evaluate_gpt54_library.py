"""Offline full-model runner admission/accounting and explicit final-pair checks."""

import base64
import hashlib
import importlib.util
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from PIL import Image

from videomasys.zerogpu.core import Caption
from videomasys.zerogpu.openai import GPT54_MODEL, OPENAI_URL, strict_json_schema

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "gpt54_library_fixture", REPO / "scripts/evaluate_gpt54_library.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_single_cached_token_is_exact_and_new_budget_cannot_modify_old_defaults(tmp_path):
    import benchmark_accounting as old

    budget = runner.Budget(tmp_path / "budget.sqlite3")
    result = budget.settle(budget.reserve("cached"), {
        "prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 1},
    })
    assert result == {"charged_usd": 0.00000025, "state": "reported",
                      "reservation_exceeded": False}
    with sqlite3.connect(budget.path) as db:
        assert db.execute("SELECT charged_units FROM requests").fetchone()[0] == 5
    for index in range(10):
        budget.reserve(f"pending-{index}")
    reopened = runner.Budget(budget.path, max_usd=4)
    with pytest.raises(runner.accounting.BudgetExceeded):
        reopened.reserve("over-cap")
    assert reopened.snapshot()["max_usd"] == 2
    assert reopened.snapshot()["charged_usd"] == 1.90000025
    assert old.RESERVATION_USD == 0.016 and old.UNITS_PER_USD == 10_000_000
    assert old._cost({"prompt_tokens": 1, "completion_tokens": 0}) == (4, True)
    assert runner.base.GPT41_MINI_MODEL == GPT54_MODEL
    assert runner.base.LIMITS == {"max_runs": 4, "max_requests": 100, "max_requests_per_run": 25}


@pytest.mark.parametrize("usage", [
    None, {"prompt_tokens": 1, "completion_tokens": 0,
           "prompt_tokens_details": {"cached_tokens": 2}},
])
def test_missing_or_bad_usage_keeps_full_reservation_across_restart(tmp_path, usage):
    budget = runner.Budget(tmp_path / "budget.sqlite3", max_usd=0.19)
    assert budget.settle(budget.reserve("ambiguous"), usage) == {
        "charged_usd": 0.19, "state": "ambiguous", "reservation_exceeded": False,
    }
    reopened = runner.Budget(budget.path)
    assert reopened.snapshot()["max_usd"] == reopened.snapshot()["charged_usd"] == 0.19
    assert reopened.snapshot()["budget_exhausted"]
    with pytest.raises(runner.accounting.BudgetExceeded):
        reopened.reserve("retry")


def test_logged_mro_sends_full_model_none_schema_and_accounts_actual_retries(tmp_path):
    token = "sk-fixture-never-a-real-key"
    frame = tmp_path / "frame.png"
    Image.new("RGB", (20, 20), "blue").save(frame)
    budget = runner.Budget(tmp_path / "budget.sqlite3")
    requests, admissions = [], []

    def respond(request):
        requests.append(request)
        assert budget.snapshot()["requests_reserved"] == len(requests)
        if len(requests) == 1:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={
            "model": GPT54_MODEL,
            "choices": [{"message": {"role": "assistant", "content": '{"caption":"Fixture"}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                      "prompt_tokens_details": {"cached_tokens": 1},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = runner.LoggedVLM(
            token, client=client, output=tmp_path, budget=budget, case_id="fixture",
            before_request=lambda: admissions.append(1), retry_backoff_seconds=0,
            image_bounds=(512, 384),
        )
        try:
            result = adapter.generate_structured("Fixture", [str(frame)], 100,
                                                 Caption.model_json_schema())
            assert result == '{"caption":"Fixture"}'
            call = adapter.calls[0]
            assert call["context_token_limit"] == 65_536
            assert call["request_sha256"] == hashlib.sha256(requests[-1].content).hexdigest()
            assert [item["budget_settlement"]["state"] for item in call["attempts"]] == [
                "ambiguous", "reported",
            ]
            assert all(item["request_attempted"] for item in call["attempts"])
            for request in requests:
                body = json.loads(request.content)
                assert str(request.url) == OPENAI_URL
                assert body["model"] == GPT54_MODEL
                assert body["reasoning_effort"] == "none" and body["temperature"] == 0
                assert body["response_format"]["json_schema"]["schema"] == strict_json_schema(
                    Caption.model_json_schema(),
                )
                image = body["messages"][1]["content"][0]["image_url"]
                assert image["detail"] == "high"
                digest = hashlib.sha256(base64.b64decode(image["url"].split(",", 1)[1])).hexdigest()
                assert call["wire_image_sha256"] == [digest]
            logged = json.loads((tmp_path / "call-001.json").read_text())
            assert logged["transport"] == call and token not in json.dumps(logged)
        finally:
            adapter.close()
    assert len(requests) == len(admissions) == 2
    assert budget.snapshot()["charged_usd"] == 0.19002025
    assert budget.snapshot()["reserved_usd"] == 0


def test_prepare_and_default_preflight_never_read_keys_or_admit_and_manifest_is_exclusive(
    tmp_path, monkeypatch, capsys,
):
    # This is the historical v6 runner's unit fixture, not a current-source run.
    monkeypatch.setattr(runner.base, "PROMPT_VERSION", runner.VERSION)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Preparation/default preflight must not access credentials or inference")

    monkeypatch.setattr(runner.base, "ROOT", tmp_path / "new-run")
    monkeypatch.setattr(runner, "_prepare_manifest", lambda: {
        "cases": [{"id": name} for name in runner.library.EXAMPLE_IDS],
        "historical_files_sha256": {},
    })
    monkeypatch.setattr(runner.shared, "controls_unchanged", lambda _manifest: True)
    monkeypatch.setattr(runner.base, "sources", lambda: {"fixture": "source-hash"})
    for name in ("read_owner_key", "CPUEmbedder", "UsageGuard", "Budget", "LoggedVLM", "run"):
        monkeypatch.setattr(runner.base, name, forbidden)
    assert runner.main(["--prepare"]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["prepared"] and prepared["model_called"] is False
    path = runner.base.ROOT / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    assert [case["id"] for case in manifest["cases"]] == list(runner.library.EXAMPLE_IDS)
    assert manifest["model"] == GPT54_MODEL and manifest["reasoning_effort"] == "none"
    assert manifest["intended_prompt_version"] == "structured-evidence-v6"
    assert manifest["max_usd"] == 2 and manifest["reservation_usd"] == 0.19
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and not result["credential_read"] and not result["model_called"]
    with pytest.raises(runner.base.EvaluationError, match="no overwrite"):
        runner.main(["--prepare"])
    assert path.read_bytes() == original
    assert set(item.name for item in path.parent.iterdir()) == {"manifest.json"}


def test_final_pair_accepts_only_matching_reviewed_restore_not_an_incomplete_draft():
    raw = {"fixture": "actual structured review"}
    answer, reasoning = {"selected_choice": "Fixture"}, {"claims": ["fixture"]}
    verdict = {"accepted": True, "explanation_accepted": False, "revision_scope": "explanation",
               "needs_revision": True, "advisories": [], "direct_assessments": raw}
    bundle = {
        "answer": deepcopy(answer), "verification": deepcopy(verdict),
        "evidence_graph": {"reasoning": deepcopy(reasoning)}, "status": "unverified",
        "reviewed_pair": {"answer_trace_index": 0, "review_trace_index": 1},
        "final_answer_complete": True, "review_complete": True,
        "trace": [{"stage": "Answer", "status": "complete", "revision": 0,
                   "output": answer, "structured_output": reasoning},
                  {"stage": "Verifier", "status": "complete", "revision": 0,
                   "output": verdict, "structured_output": raw}],
    }
    assert runner.shared.final_state is runner.final_state
    assert runner.final_state(bundle)
    bundle["trace"].append({"stage": "Answer", "status": "failed"})
    assert not runner.final_state(bundle)
    bundle.update(retained_reviewed_answer=True, explanation_repair_failed=True)
    bundle["verification"].update(
        needs_revision=False, explanation_repair_failed=True,
        advisories=["Explanation refinement did not complete; "
                    "the previously reviewed answer is unchanged."],
    )
    assert runner.final_state(bundle)
    for change in ({"retained_previous_draft": True}, {"review_complete": False},
                   {"final_answer_complete": False}, {"status": "verified"},
                   {"reviewed_pair": {}}, {"explanation_repair_failed": False},
                   {"evidence_graph": {"reasoning": {"claims": ["different"]}}}):
        assert not runner.final_state({**bundle, **change})
    changed = deepcopy(bundle)
    changed["verification"]["accepted"] = False
    assert not runner.final_state(changed)
