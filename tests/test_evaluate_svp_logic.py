"""Offline checks for the fixed-cache SVP evaluation; never uses owner credentials."""

import importlib.util
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_svp_logic.py"
spec = importlib.util.spec_from_file_location("evaluate_svp_logic_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("case,count", [("sofa_mc", 5), ("kitchen_open", 9)])
def test_historical_seed_is_read_only_and_excludes_old_answers(case, count):
    if not runner.BASELINE.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    seed = runner.cache_seed(case)
    assert len(seed["scenes"]) == count
    assert len(seed["vectors"]) == count and len(seed["vectors"][0]) == 384
    assert {"answer", "verification", "trace", "inference", "selected"}.isdisjoint(seed)
    assert set(seed["scenes"][0]) == {
        "index", "start", "end", "frames", "frame_times", "caption"
    }
    database = seed["provenance"]["baseline_database"]
    assert runner.file_sha(Path(database["path"])) == database["sha256"]
    assert seed["provenance"]["ground_truth_in_model_input"] is False
    assert len(seed["provenance"]["index_frame_sha256"]) == count * 4
    assert runner.cache_seed(case) == seed


@pytest.mark.parametrize("profile", ["clean", "svp"])
def test_preflight_constructs_actual_pipeline_without_session_key_or_new_ledger(
    profile, monkeypatch, tmp_path, capsys
):
    if not runner.BASELINE.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    module = runner.importlib.import_module(f"videomasys.zerogpu.{profile}_prompts")
    monkeypatch.setattr(module, "PROMPT_VERSION", runner.PROFILES[profile])
    monkeypatch.setattr(runner, "ROOT", tmp_path / "not_created")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.RemotePipeline, "analysis_session",
                        lambda *a: pytest.fail("admitted session"))
    monkeypatch.setattr(runner.CPUEmbedder, "__call__", lambda *a: pytest.fail("embedding call"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("ledger opened"))
    assert runner.main(["--profile", profile, "--case", "sofa_mc"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["credential_read"] is False and output["model_called"] is False
    assert output["model"] == runner.GPT41_MINI_MODEL
    assert output["evaluation_index_signature"].endswith(runner.PROFILES[profile])
    assert not runner.ROOT.exists()


def test_preflight_denies_unknown_version_before_inputs_or_credentials(monkeypatch):
    monkeypatch.setattr(runner, "cache_seed", lambda *a: pytest.fail("read input"))
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.importlib, "import_module", lambda *a: SimpleNamespace(
        PROMPT_VERSION="svp-roles-v999"
    ))
    with pytest.raises(runner.EvaluationError, match="not authorized"):
        runner.preflight("svp", "sofa_mc")
    with pytest.raises(runner.EvaluationError, match="authorized profile"):
        runner.preflight("legacy", "sofa_mc")
    with pytest.raises(runner.EvaluationError, match="authorized profile"):
        runner.preflight("svp", "unbudgeted_case")


@pytest.fixture
def seed():
    return {
        "key": "fixture-video-hash", "duration": 4.0, "video_path": "/fixture/video.mp4",
        "video_name": "video.mp4", "question": "What is happening?", "options": [],
        "scenes": [{"index": 0, "start": 0.0, "end": 4.0, "frames": [],
                    "frame_times": [], "caption": "The person moves a cup."}],
        "vectors": [[1.0] + [0.0] * 383], "original_signature": "original-clean-index",
        "provenance": {"fixed_input_sha256": "immutable-inputs"},
    }


def test_seed_rebinds_exact_vectors_and_captions_without_mutating_input(seed, tmp_path):
    original = deepcopy(seed)
    bundle = runner.seed_bundle(seed, tmp_path, "isolated-svp-index")
    restored = runner.SceneCache(bundle["database"], "isolated-svp-index").load(bundle["scenes"])
    assert restored[0] == seed["scenes"]
    assert np.array_equal(restored[1], np.asarray(seed["vectors"], dtype=np.float32))
    assert runner.SceneCache(bundle["database"], "original-clean-index").load(
        bundle["scenes"]
    ) is None
    assert seed == original
    bundle["scenes"][0]["caption"] = "Edited working result"
    assert seed == original
    assert Path(bundle["frame_directory"]) == tmp_path / "frames"


def test_run_has_one_admission_shared_limits_closes_and_never_overwrites(
    seed, tmp_path, monkeypatch
):
    key = "sk-fixture-not-a-real-credential"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    calls = []

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["provider"] == "openai" and kwargs["token"] == key
            assert kwargs["vlm_factory"] is runner.ProvenanceVLM
            assert os.environ["VLM_MODEL"] == runner.GPT41_MINI_MODEL
            self.guard = kwargs["guard"]
            self.signature = kwargs["prompt_profile"] + "-index"
            calls.append(self)

        @contextmanager
        def analysis_session(self, bundle):
            allowance = self.guard.start_run()
            bundle["inference"] = {"calls": []}

            def qa(bundle, question, options):
                assert question == seed["question"] and options == []
                allowance.before_request()
                bundle["inference"]["calls"].append({
                    "attempts": [{"request_attempted": True}],
                })
                bundle["trace"].append({"stage": "Fixture QA", "status": "complete"})
                bundle["answer"] = {"answer": "Moving a cup.", "supporting_scenes": [0]}
                bundle["verification"] = {"accepted": True}
                yield bundle

            try:
                yield SimpleNamespace(signature=self.signature, qa=qa)
            finally:
                allowance.close()
                bundle["inference"]["closed_before_publication"] = True

    def preflight(profile, case):
        def index(bundle):
            restored = runner.SceneCache(bundle["database"], profile + "-index").load(
                bundle["scenes"]
            )
            assert restored is not None
            bundle["cached"] = True

        return deepcopy(seed), SimpleNamespace(
            signature=profile + "-index", embed=None, index=index,
            prompt_version=runner.PROFILES[profile],
        )

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    for profile in ("clean", "svp"):
        assert runner.evaluate(profile, "sofa_mc", run=True) == 0
        directory = tmp_path / f"{profile}__sofa_mc"
        report = json.loads((directory / "report.json").read_text())
        trace = json.loads((directory / "raw-trace.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        assert report["qa_completed"] is True and report["inference_invoked"] is True
        assert report["result_is_execution_not_correctness"] is True
        assert trace[-1]["closed_before_publication"] is True
        assert manifest["caption_generation_enabled"] is False
        assert manifest["original_index_signature"] == "original-clean-index"
        assert manifest["public_space_modified"] is False
        assert all(key not in path.read_text() for path in directory.glob("*.json"))
    snapshot = calls[-1].guard.snapshot()
    assert snapshot["runs_used"] == 2 and snapshot["request_attempts_used"] == 2
    assert snapshot["max_request_attempts"] == 60 and snapshot["max_runs"] == 4
    with pytest.raises(runner.EvaluationError, match="no rerun"):
        runner.evaluate("clean", "sofa_mc", run=True)
    assert len(calls) == 2


def test_budget_survives_reopening_and_denies_attempt_sixty_one(tmp_path):
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    first = guard.start_run()
    for _ in range(32):
        first.before_request()
    first.close()
    reopened = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    second = reopened.start_run()
    for _ in range(28):
        second.before_request()
    from videomasys.zerogpu.usage import UsageLimitError

    with pytest.raises(UsageLimitError):
        second.before_request()
    second.close()
    assert reopened.snapshot()["request_attempts_used"] == 60
    assert reopened.snapshot()["runs_used"] == 2


def test_four_run_limit_is_separate_from_request_allowance(tmp_path):
    from videomasys.zerogpu.usage import UsageLimitError

    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    for _ in range(4):
        allowance = guard.start_run()
        allowance.before_request()
        allowance.close()
    with pytest.raises(UsageLimitError):
        guard.start_run()
    assert guard.snapshot()["runs_used"] == 4
    assert guard.snapshot()["request_attempts_used"] == 4


def test_failed_cache_preflight_cannot_admit_or_read_credentials(seed, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("admitted run"))
    monkeypatch.setattr(runner, "preflight", lambda *a: (
        deepcopy(seed), SimpleNamespace(
            signature="isolated", prompt_version="svp-roles-v1",
            index=lambda bundle: bundle.update(cached=False),
        )
    ))
    with pytest.raises(runner.EvaluationError, match="exact historical cache hit"):
        runner.evaluate("svp", "sofa_mc", run=True)
    assert not (tmp_path / "usage.sqlite3").exists()


def used_batch(tmp_path, requests=36):
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    remaining = requests
    for _ in range(4):
        allowance = guard.start_run()
        count = min(15, remaining)
        for _ in range(count):
            allowance.before_request()
        remaining -= count
        allowance.close()
    for case in runner.CASES:
        directory = tmp_path / f"svp__{case}"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "case": case, "prompt_version": "svp-roles-v1",
            "source_sha256": {"src/videomasys/zerogpu/core.py": "historical-v1-hash"},
        }))
    return guard


def test_v2_protocol_requires_original_used_ledger_and_keeps_global_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(runner.EvaluationError):
        runner.protocol("svp-roles-v2", "sofa_mc")
    assert not (tmp_path / "usage.sqlite3").exists()
    guard = used_batch(tmp_path)
    before = guard.snapshot()
    limits, amendment = runner.protocol("svp-roles-v2", "sofa_mc")
    assert limits == {"max_runs": 6, "max_requests": 60, "max_requests_per_run": 12}
    assert amendment["remaining_global_request_attempts"] == 24
    assert amendment["max_attempts_this_run"] == 12
    assert amendment["original_limits"] == runner.LIMITS
    assert amendment["additional_request_allowance"] == 0
    assert amendment["ledger_reset"] is False
    assert guard.snapshot() == before
    assert amendment["prior_candidate_source_sha256"] == {
        "src/videomasys/zerogpu/core.py": "historical-v1-hash"
    }


@pytest.mark.parametrize("requests,expected", [(59, 1), (60, None)])
def test_v2_respects_remaining_requests_and_exhaustion(tmp_path, monkeypatch, requests, expected):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    guard = used_batch(tmp_path, requests)
    if expected is None:
        with pytest.raises(runner.EvaluationError, match="exhausted"):
            runner.protocol("svp-roles-v2", "sofa_mc")
    else:
        limits, amendment = runner.protocol("svp-roles-v2", "sofa_mc")
        assert limits["max_requests_per_run"] == expected
        assert amendment["max_attempts_this_run"] == expected
    assert guard.snapshot()["request_attempts_used"] == requests


def test_v2_runs_exclusive_in_same_ledger_and_retains_source_snapshots(seed, tmp_path, monkeypatch):
    key = "sk-fake-v2-only"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    guard = used_batch(tmp_path)
    old_hashes = {path: runner.file_sha(path) for path in tmp_path.glob("svp__*/manifest.json")}
    monkeypatch.setattr(runner, "preflight", lambda *args: (
        deepcopy(seed), SimpleNamespace(
            signature="v2-index", embed=None, prompt_version="svp-roles-v2",
            index=lambda bundle: None,
        )
    ))

    class Pipeline:
        def __init__(self, **kwargs):
            self.guard = kwargs["guard"]
            assert kwargs["token"] == key
            assert self.guard.path == tmp_path / "usage.sqlite3"
            assert self.guard.max_requests == 60
            assert self.guard.max_requests_per_run == 12

        @contextmanager
        def analysis_session(self, bundle):
            allowance = self.guard.start_run()
            bundle["inference"] = {"calls": []}

            def qa(bundle, question, options):
                for _ in range(13):
                    allowance.before_request()
                    bundle["inference"]["calls"].append({
                        "attempts": [{"request_attempted": True}],
                    })
                    bundle["trace"].append({"stage": "Fixture stage"})
                    yield bundle

            try:
                yield SimpleNamespace(signature="v2-index", qa=qa)
            finally:
                allowance.close()
                bundle["inference"]["closed_before_publication"] = True

    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    for case in runner.CASES:
        assert runner.evaluate("svp", case, run=True) == 1
        directory = runner.evaluation_directory("svp", "svp-roles-v2", case)
        assert directory == tmp_path / f"svp__svp-roles-v2__{case}"
        report = json.loads((directory / "report.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        snapshots = json.loads((directory / "source-snapshots.json").read_text())
        assert report["error_code"] == "allowance"
        assert manifest["protocol_amendment"]["additional_sessions"] == 2
        assert manifest["limits"]["max_requests_per_run"] == 12
        assert set(snapshots) == {
            "src/videomasys/zerogpu/core.py", "src/videomasys/zerogpu/svp_retrieval.py"
        }
        with pytest.raises(runner.EvaluationError, match="no rerun"):
            runner.evaluate("svp", case, run=True)
    assert all(runner.file_sha(path) == value for path, value in old_hashes.items())
    assert guard.snapshot()["request_attempts_used"] == 60
    assert guard.snapshot()["runs_used"] == 6
    with pytest.raises(runner.EvaluationError, match="exhausted"):
        runner.protocol("svp-roles-v2", "sofa_mc")


def test_regular_input_refuses_symlinks_and_outside_root(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(runner.EvaluationError):
        runner.regular_input(link, tmp_path)
    with pytest.raises(runner.EvaluationError):
        runner.regular_input(source, tmp_path / "other")
