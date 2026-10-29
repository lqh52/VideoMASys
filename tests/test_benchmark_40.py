"""Small offline checks for benchmark scoring and label isolation, not accuracy tests."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_benchmark_40.py"
spec = importlib.util.spec_from_file_location("benchmark40_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("fail", [False, True])
def test_labels_are_scoring_only_and_failed_drafts_do_not_count_correct(
    tmp_path, monkeypatch, fail,
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "sources", lambda: {})
    (tmp_path / "cases").mkdir()
    case = {"id": "nextqa-fixture", "dataset": "nextqa", "split": "val", "video": "test.mp4",
            "video_sha256": "fixture", "duration_seconds": 20, "question": "What happens?",
            "options": ["A", "B", "C", "D", "E"],
            "reference": {"correct_option": 1, "correct_choice": "B"}}
    monkeypatch.setattr(runner, "prepare_video", lambda *_: {
        "key": "fixture", "scenes": [], "trace": [], "cached": False,
    })

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["provider"] == "openai"
            assert kwargs["prompt_profile"] == "structured"

        @contextmanager
        def analysis_session(self, bundle):
            assert "reference" not in bundle and "correct_option" not in bundle
            yield self

        def index(self, bundle):
            assert "reference" not in bundle

        def save_index(self, bundle):
            bundle["inference"] = {"calls": []}

        def qa(self, bundle, question, options):
            assert question == case["question"] and options == case["options"]
            assert "reference" not in bundle
            bundle["answer"] = {"selected_option": 1, "selected_choice": "B", "answer": "B"}
            bundle["trace"].append({"stage": "Answer"})
            yield bundle
            if fail:
                raise RuntimeError("Synthetic review failure")
            bundle["verification"] = {"accepted": False}
            bundle["status"] = "unverified"
            yield bundle

    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    report = runner.run_case(case, "fixture-key", None, None, None, {})
    assert report["completed"] is (not fail)
    assert report["correct"] is (not fail)  # Not conditioned on reviewer acceptance.
    assert report["selected_option"] == 1
    saved = json.loads((tmp_path / "cases/nextqa-fixture/bundle.json").read_text())
    assert "reference" not in saved
    with pytest.raises(FileExistsError):
        runner.run_case(case, "fixture-key", None, None, None, {})


def test_manifest_preflight_has_exact_twenty_per_dataset_and_no_credentials(monkeypatch, capsys):
    if not (runner.ROOT / "manifest.json").exists():
        pytest.skip("Local-only evaluation manifest")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 40
    assert result["datasets"] == {"nextqa": 20, "egoschema": 20}
    assert result["max_usd"] == 4.0
    assert result["credential_read"] is False
