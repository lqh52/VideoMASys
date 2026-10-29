"""Evaluation tooling tests use explicit fixtures; they never invoke a model."""

import csv
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_zerogpu_library.py"
spec = importlib.util.spec_from_file_location("zerogpu_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture
def example():
    return {
        "id": "fixture",
        "number": 1,
        "video": "fixture.mp4",
        "question": "What happens?",
        "options": ["First", "Second"],
        "reference": {"correct_option": 1},
    }


def api_output(selected=1, accepted=False):
    return [
        "status",
        "answer",
        [],
        [
            {
                "stage": "Video cache",
                "status": "complete",
                "cache_hit": True,
                "scenes": [{"index": 0, "caption": "Real caption from supplied trace."}],
            },
            {
                "stage": "Retrieval",
                "status": "complete",
                "output": {
                    "selected": [{"index": 0, "start": 0, "end": 20, "caption": "Caption"}],
                    "calls": [{"id": "l1"}],
                },
            },
            {
                "stage": "Answer",
                "status": "complete",
                "output": {
                    "answer": "Fixture answer.",
                    "selected_option": selected,
                    "supporting_scenes": [0],
                },
                "attempts": [{"raw_response": "bad"}, {"raw_response": "fixed"}],
            },
            {
                "stage": "Verifier",
                "status": "complete",
                "output": {
                    "accepted": accepted,
                    "issues": [] if accepted else ["Not enough evidence"],
                },
            },
        ],
    ]


def test_inference_payload_excludes_reference(example):
    assert evaluation.inference_inputs(example, lambda p: {"path": p}) == (
        {"path": "fixture.mp4"},
        "What happens?",
        "First\nSecond",
    )
    assert "correct_option" not in str(evaluation.inference_inputs(example, str))


def test_reverse_choices_remaps_only_scoring_and_preserves_original_reference(example):
    original = json.loads(json.dumps(example))
    reordered = evaluation.reverse_options(example)
    assert example == original
    assert reordered["options"] == ["Second", "First"]
    assert reordered["reference"]["correct_option"] == 0
    assert reordered["reference"]["options"][0] == "Second"
    assert reordered["original_reference"] == example["reference"]
    assert reordered["option_order"]["presented_index_to_original_index"] == [1, 0]
    assert evaluation.inference_inputs(reordered, str) == (
        "fixture.mp4",
        "What happens?",
        "Second\nFirst",
    )
    assert evaluation.summarize_output(api_output(selected=0), reordered)["correct"] is True
    assert evaluation.summarize_output(api_output(selected=1), reordered)["correct"] is False


def test_rejected_correct_choice_is_not_a_failure(example):
    summary = evaluation.summarize_output(api_output(), example)
    assert summary["completed"] and summary["correct"]
    assert summary["accepted"] is False
    assert summary["repair_count"] == 1
    assert summary["retrieved_scene_ids"] == [0]
    assert summary["cached_scenes"][0]["caption"] == "Real caption from supplied trace."


def test_partial_and_malformed_outputs_are_not_completed(example):
    output = api_output()
    output[3].pop()
    assert evaluation.summarize_output(output, example)["completed"] is False


def test_independent_inspection_calls_are_not_counted_as_repairs(example):
    output = api_output()
    output[3].insert(
        2,
        {
            "stage": "Evidence inspection",
            "status": "complete",
            "attempts": [
                {"raw_response": "scene 0"},
                {"raw_response": "scene 1"},
                {"raw_response": "scene 2"},
            ],
            "repair_count": 0,
        },
    )
    summary = evaluation.summarize_output(output, example)
    assert summary["repair_count"] == 1  # Only the separate Answer stage repaired.
    inspection = next(m for m in summary["stage_metrics"] if m["stage"] == "Evidence inspection")
    assert inspection["repairs"] == 0 and inspection["attempts"] == 3
    assert evaluation.summarize_output(None, example)["correct"] is None
    output[3] = "not a trace"
    assert evaluation.summarize_output(output, example)["completed"] is False


def test_aggregate_distinguishes_wrong_acceptance_from_correct_rejection(example):
    records = [
        {
            "example": example,
            "summary": evaluation.summarize_output(output, example),
            "wall_seconds": 2,
            "error": None,
        }
        for output in [api_output(), api_output(selected=0, accepted=True)]
    ]
    result = evaluation.aggregate(records)
    assert result["correct"] == 1 and result["completed"] == 2
    assert result["accepted_wrong"] == result["correct_rejected"] == 1
    assert result["accuracy_over_attempted"] == 0.5


def test_hosted_failure_keeps_latest_real_generator_output(tmp_path, example):
    class Job:
        def done(self):
            return True

        def outputs(self):
            output = api_output()
            output[3].pop()
            return [output]

        def result(self, timeout):
            raise RuntimeError("Fixture disconnection")

        def cancel(self):
            return True

    client = SimpleNamespace(submit=lambda *args, **kwargs: Job())
    record = evaluation.run_example(client, example, tmp_path, 10, str)
    saved = json.loads((tmp_path / "01-fixture.json").read_text())
    assert "Fixture disconnection" in record["error"]
    assert saved["raw_output"][3][-1]["stage"] == "Answer"
    assert saved["summary"]["completed"] is False
    assert len(saved["events"]) == 1


def test_baseline_loader_restores_canonical_modules(tmp_path):
    import videomasys.domain.schemas as current_schemas
    import videomasys.zerogpu.core as current_core

    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for path in evaluation.SOURCE_FILES.values():
            target.add(SCRIPT.parents[1] / path, arcname=path)
    frozen_core, prompt, hashes = evaluation.load_local_source(archive)
    assert frozen_core is not current_core
    assert frozen_core.ReasoningPlan is not current_schemas.ReasoningPlan
    assert sys.modules["videomasys.domain.schemas"] is current_schemas
    assert sys.modules["videomasys.zerogpu.core"] is current_core
    assert "never fabricate observations or tool results" in prompt
    assert set(hashes) == set(evaluation.SOURCE_FILES)


def test_local_requests_use_real_api_format_and_no_reference(monkeypatch, tmp_path):
    from PIL import Image

    calls = []

    class HTTP:
        def __init__(self, **kwargs):
            pass

        def post(self, url, json):
            calls.append((url, json))
            data = (
                {"choices": [{"message": {"content": '{"answer":"text"}'}}]}
                if url.endswith("chat/completions")
                else {"data": [{"index": 0, "embedding": [3, 4]}]}
            )
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    import httpx

    monkeypatch.setattr(httpx, "Client", HTTP)
    core, prompt, _ = evaluation.load_local_source(None)
    args = SimpleNamespace(
        timeout=10,
        vlm_url="http://localhost:1/v1",
        embedding_url="http://localhost:2/v1",
        model_id="test-model",
        embedding_id="test-embedding",
    )
    pipeline, _ = evaluation.local_pipeline(core, prompt, args)
    frame = tmp_path / "frame.png"
    Image.new("RGB", (160, 90), "blue").save(frame)
    assert pipeline.generate("Requested facts", [str(frame)], 123) == '{"answer":"text"}'
    payload = calls[0][1]
    assert payload["temperature"] == 0 and payload["max_tokens"] == 123
    assert "response_format" not in payload and "json_schema" not in payload
    assert payload["messages"][0] == {"role": "system", "content": prompt}
    assert payload["messages"][1]["content"][0]["type"] == "image_url"
    assert payload["messages"][1]["content"][-1]["text"] == "Requested facts"
    np.testing.assert_allclose(pipeline.embed(["caption"]), [[0.6, 0.8]])

    inference = tmp_path / "inference.py"
    inference.write_text(
        "def generate(paths):\n"
        "    picture.thumbnail((512, 384) if len(paths) > 6 else (640, 480))\n"
        "    model.generate(repetition_penalty=1.08)\n"
    )
    archive = tmp_path / "inference.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(inference, arcname=evaluation.SOURCE_FILES["inference"])
    args.source_archive = archive
    penalized, _ = evaluation.local_pipeline(core, prompt, args)
    penalized.generate("Requested facts", [str(frame)], 123)
    assert calls[-1][1]["repetition_penalty"] == 1.08
    assert penalized.generation_settings == {"temperature": 0, "repetition_penalty": 1.08}


def test_references_checked_against_source_and_ui_aliases(tmp_path):
    library_root = tmp_path / "library"
    library_root.mkdir()
    nextqa_root = tmp_path / "nextqa"
    (nextqa_root / "nextqa").mkdir(parents=True)
    ego_root = tmp_path / "ego"
    ego_root.mkdir()
    library = []
    rows = []
    for example_id, alias in evaluation.NEXTQA_ALIASES.items():
        library.append(
            {
                "title": example_id,
                "file": "video.mp4",
                "video_id": "123",
                "question": alias["question"],
                "options": alias["options"],
            }
        )
        rows.append(
            {
                "video": "123",
                "qid": alias["qid"],
                "question": alias["source_question"],
                "answer": "2",
                **{f"a{i}": s for i, s in enumerate(alias["source_options"])},
            }
        )
    with (nextqa_root / "nextqa" / "val.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ego_rows = []
    for i in range(2):
        library.append(
            {
                "title": f"ego{i}",
                "file": "video.mp4",
                "video_id": str(i),
                "question": "Purpose?",
                "options": ["One", "Two"],
            }
        )
        ego_rows.append(
            {
                "video_uid": str(i),
                "question": "Purpose?",
                "options": ["O1: One", "O2: Two"],
                "ground_truth": "O2: Two",
            }
        )
    (library_root / "video.mp4").write_bytes(b"fixture; not decoded")
    (library_root / "library.json").write_text(json.dumps(library))
    (ego_root / "subset_first10_with_gt.json").write_text(json.dumps(ego_rows))
    loaded = evaluation.load_examples(library_root, nextqa_root, ego_root)
    assert [e["reference"]["correct_option"] for e in loaded] == [2, 2, 1, 1]
    library[0]["options"] = list(reversed(library[0]["options"]))
    (library_root / "library.json").write_text(json.dumps(library))
    with pytest.raises(ValueError, match="UI choices"):
        evaluation.load_examples(library_root, nextqa_root, ego_root)
