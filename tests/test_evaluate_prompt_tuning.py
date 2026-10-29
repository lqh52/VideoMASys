"""Offline prompt replay checks; never reads owner credentials or calls paid inference."""

import importlib.util
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_prompt_tuning.py"
spec = importlib.util.spec_from_file_location("evaluate_prompt_tuning_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_requires_explicit_run_and_rejects_expansive_variants(monkeypatch):
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    with pytest.raises(SystemExit):
        runner.main(["--mode", "replay", "--profile", "clean", "--case", "sofa_mc"])
    with pytest.raises(runner.EvaluationError):
        runner.evaluate("full", "clean", "ball_mc")
    with pytest.raises(runner.EvaluationError):
        runner.evaluate("replay", "tuned", "ball_mc")


@pytest.mark.parametrize("case,suspect", [("sofa_mc", 3), ("kitchen_open", 6)])
def test_actual_baseline_reproduces_exact_requests_schemas_and_images(case, suspect):
    if not runner.BASELINE.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    data, answer, inspection = runner.replay_inputs(case)
    assert data["scene"]["index"] == suspect
    assert len(data["images"]) == 18 and len(data["scene"]["frames"]) == 6
    assert data["provenance"]["exact_clean_request_and_schema_match"] is True
    assert len(data["provenance"]["frame_files"]) == 18
    assert answer.model_json_schema()["title"] in ("GroundedAnswer", "OpenEndedAnswer")
    assert inspection.model_json_schema()["properties"]["scene"]["const"] == suspect


@pytest.fixture
def replay_data():
    if not runner.BASELINE.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    return runner.replay_inputs("kitchen_open")


def test_tuned_changes_instructions_not_fixed_arguments(replay_data):
    data, _, _ = replay_data
    original = deepcopy(data)
    clean = runner.replay_prompts(runner.prompt_module("clean"), data)
    tuned = runner.replay_prompts(runner.prompt_module("tuned"), data)
    assert data == original
    assert clean["answer"] != tuned["answer"]
    assert runner.review_payload(data["original_draft"], []) in tuned["original_review"]
    assert data["evidence"] in clean["original_review"]
    # Deliberate tuned treatment: independently review images without inspector priming.
    assert data["evidence"] not in tuned["original_review"]


def test_controlled_replay_keeps_bad_observations_and_original_draft(replay_data):
    from videomasys.zerogpu.core import LivePipeline

    data, answer_schema, inspect_schema = replay_data
    requests = []
    responses = [
        {"scene": 6, "observation": "A blue sponge touches the board.", "relevant": True,
         "uncertainty": "", "refine_at": None},
        {"answer": "The person washes kitchen items.", "supporting_scenes": [6],
         "limitations": [], "inference": "Washing actions support a cleanup objective."},
        {"accepted": False, "issues": ["The draft names an unsupported tool."],
         "needs_revision": True},
        {"accepted": True, "issues": [], "needs_revision": False},
    ]

    def generated(prompt, paths, tokens, schema):
        requests.append((prompt, list(paths), tokens, schema))
        return json.dumps(responses[len(requests) - 1])

    active = LivePipeline(None, None, runner.GPT41_MINI_MODEL, "fixture",
                          generate_structured=generated, prompt_profile="clean")
    bundle = {"trace": []}
    runner.run_replay(active, bundle, data, answer_schema, inspect_schema, runner.clean_prompts)
    assert len(requests) == 4
    assert "A blue sponge touches the board." not in requests[1][0]
    assert data["evidence"] in requests[1][0]
    assert "with a brush" in requests[2][0]
    assert "The person washes kitchen items." in requests[3][0]
    assert requests[1][1] == requests[2][1] == requests[3][1] == data["images"]
    assert bundle["original_draft_verification"]["accepted"] is False
    assert bundle["verification"]["accepted"] is True


def test_shared_budget_close_before_artifacts_no_overwrite(tmp_path, monkeypatch, replay_data):
    from videomasys.zerogpu.core import LivePipeline

    key = "sk-fixture-only-no-network"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    monkeypatch.setattr(runner, "CPUEmbedder", lambda: None)
    monkeypatch.setattr(runner, "replay_inputs", lambda case: deepcopy(replay_data))
    calls = []

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["provider"] == "openai" and kwargs["token"] == key
            assert kwargs["vlm_factory"] is runner.ProvenanceVLM
            assert os.environ["VLM_MODEL"] == runner.GPT41_MINI_MODEL
            self.guard = kwargs["guard"]
            self.profile = kwargs["prompt_profile"]
            calls.append(self)

        @contextmanager
        def analysis_session(self, bundle):
            allowance = self.guard.start_run()
            bundle["inference"] = {"calls": []}

            def generated(prompt, paths, tokens, schema):
                allowance.before_request()
                bundle["inference"]["calls"].append({
                    "attempts": [{"request_attempted": True}],
                })
                if "ObservationScene" in schema["title"]:
                    result = {"scene": 6, "observation": "Cleaning kitchen items.",
                              "relevant": True, "uncertainty": "", "refine_at": None}
                elif schema["title"] == "Verdict":
                    result = {"accepted": True, "issues": [], "needs_revision": False}
                else:
                    result = {"answer": "Cleaning kitchen items.", "supporting_scenes": [6],
                              "limitations": [], "inference": "Scrubbing implies cleaning."}
                return json.dumps(result)

            active = LivePipeline(None, None, runner.GPT41_MINI_MODEL, "fixture",
                                  generate_structured=generated, prompt_profile=self.profile)
            try:
                yield active
            finally:
                allowance.close()
                bundle["inference"]["closed_before_publication"] = True

    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    for profile, version in (("clean", "clean-roles-v2"), ("tuned", "grounded-roles-v1"),
                             ("tuned", "grounded-roles-v2")):
        monkeypatch.setattr(runner.prompt_module(profile), "PROMPT_VERSION", version)
        assert runner.evaluate("replay", profile, "kitchen_open") == 0
        output = runner.evaluation_directory(profile, version, "replay", "kitchen_open")
        report = json.loads((output / "report.json").read_text())
        trace = json.loads((output / "raw-trace.json").read_text())
        assert report["qa_completed"] is True and report["inference_invoked"] is True
        assert trace[-1]["closed_before_publication"] is True
        assert all(key not in path.read_text() for path in output.glob("*.json"))
    snapshot = calls[-1].guard.snapshot()
    assert snapshot["runs_used"] == 3 and snapshot["request_attempts_used"] == 12
    assert snapshot["max_request_attempts"] == 60
    assert snapshot["max_runs"] == 8
    assert (tmp_path / "tuned__replay__kitchen_open/report.json").is_file()
    assert (tmp_path / "tuned__grounded-roles-v2__replay__kitchen_open/report.json").is_file()
    with pytest.raises(runner.EvaluationError, match="no rerun"):
        runner.evaluate("replay", "clean", "kitchen_open")
    with pytest.raises(runner.EvaluationError, match="no rerun"):
        runner.evaluate("replay", "tuned", "kitchen_open")
    assert len(calls) == 3


def test_unauthorized_version_rejected_before_credentials_or_new_ledger(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(runner, "ROOT", tmp_path / "uncreated")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner, "prompt_module", lambda profile: SimpleNamespace(
        PROMPT_VERSION="grounded-roles-v3"
    ))
    with pytest.raises(runner.EvaluationError, match="not authorized"):
        runner.evaluate("replay", "tuned", "sofa_mc")
    assert not runner.ROOT.exists()


def test_true_holdout_inputs_do_not_include_labels():
    case = runner.full_inputs("ball_mc")
    assert case["video"].name == "2535384528.mp4"
    assert len(case["options"]) == 5
    assert not {"answer", "label", "correct_choice"}.intersection(case)
    assert runner.full_inputs("painting_open")["options"] == []


@pytest.mark.parametrize("schema", [None, {"type": "object", "properties": {
    "caption": {"type": "string"}}, "required": ["caption"]}])
def test_wire_hashes_match_actual_payload_without_header_secrets(tmp_path, monkeypatch, schema):
    import base64

    import httpx
    from PIL import Image

    path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), color="blue").save(path)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("No HTTP request permitted in provenance test")
    ))
    adapter = runner.ProvenanceVLM("sk-fixture-provenance", client=client)
    monkeypatch.setattr(runner.OpenAI41MiniVLM, "_perform", lambda *args: "fixture result")
    try:
        payload, _ = adapter._payload("Fixture prompt", [str(path)], 100, schema=schema)
        record = {}
        assert adapter._perform(payload, 1, record) == "fixture result"
        body = json.loads(payload)
        url = body["messages"][1]["content"][0]["image_url"]["url"]
        assert record["request_sha256"] == runner.sha(payload)
        assert record["wire_image_sha256"] == [
            runner.sha(base64.b64decode(url.split(",", 1)[1]))
        ]
        assert "sk-fixture-provenance" not in payload.decode()
        assert set(record) == {"request_sha256", "wire_image_sha256", "request_schema_sha256"}
    finally:
        adapter.close()
        client.close()


def test_new_ledger_refuses_request_sixty_one(tmp_path):
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    first = guard.start_run()
    for _ in range(32):
        first.before_request()
    first.close()
    second = guard.start_run()
    for _ in range(28):
        second.before_request()
    from videomasys.zerogpu.usage import UsageLimitError

    with pytest.raises(UsageLimitError):
        second.before_request()
    second.close()
    assert guard.snapshot()["request_attempts_used"] == 60
