"""Fixtures test the local harness, not video/model accuracy."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_agent_icl", ROOT / "scripts/evaluate_agent_icl.py"
)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_read_actual_system_literal_without_import(tmp_path):
    source = tmp_path / "inference.py"
    source.write_text('raise RuntimeError("must not import")\nSYSTEM_PROMPT = "actual"\n')
    assert harness.read_system_prompt(source) == "actual"
    source.write_text('def f():\n return {"role": "system", "content": "nested " "literal"}\n')
    assert harness.read_system_prompt(source) == "nested literal"
    source.write_text("SYSTEM_PROMPT = unknown_variable\n")
    with pytest.raises(ValueError):
        harness.read_system_prompt(source)


def test_saved_case_does_not_propagate_reference_or_prior_outputs(tmp_path):
    source = tmp_path / "saved.json"
    source.write_text(
        json.dumps(
            {
                "reference": "NEVER_INCLUDE_REFERENCE",
                "example": {"question": "What happens?", "options": ["A", "B"], "answer": "SECRET"},
                "bundle": {
                    "answer": "SECRET",
                    "trace": [{"raw_response": "SECRET"}],
                    "scenes": [
                        {
                            "index": 0,
                            "start": 0,
                            "end": 20,
                            "caption": "Actual saved caption",
                            "frames": ["private.png"],
                        }
                    ],
                },
            }
        )
    )
    case = harness.load_saved_case(source, "test", "whole_video")
    serialized = json.dumps(case)
    assert "SECRET" not in serialized
    assert "NEVER_INCLUDE_REFERENCE" not in serialized
    assert "private.png" not in serialized
    assert case["scenes"][0]["caption"] == "Actual saved caption"


def test_two_yields_stop_before_retrieval(monkeypatch):
    core, prompts = harness.import_live()
    demonstration = prompts.DEMONSTRATIONS["global_workbench"]
    outputs = iter([json.dumps(demonstration["plan"]), json.dumps(demonstration["program"])])
    requests = []

    def fixture_generate(prompt, images, tokens):
        assert not images
        requests.append(prompt)
        return next(outputs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Retrieval/sampling must not execute")

    monkeypatch.setattr(core.SceneCache, "load", forbidden)
    monkeypatch.setattr(core, "sample_scene_frames", forbidden)
    case = {
        "id": "fixture_only",
        "question": demonstration["question"],
        "options": [],
        "scenes": [{"index": 0, "start": 0, "end": 20, "caption": "Fixture hint", "frames": []}],
        "expected_scope": "whole_video",
        "constraints": {},
    }
    result = harness.run_case(core, case, fixture_generate)
    assert result["contracts_passed"]
    assert [stage["stage"] for stage in result["trace"]] == ["Planner", "Tool agent"]
    assert len(requests) == 2
    assert result["answer_generated"] is False
    assert result["video_accuracy_evaluated"] is False
    context = harness.question_context(case["question"], [])
    assert requests[0].startswith(prompts.planner_prompt(context)[0])
    assert "Fixture hint" not in requests[0]
    compiler_context = core.tool_search_context(context, case["scenes"], "whole_video")
    validated_plan = core.QAPlan.model_validate(demonstration["plan"]).model_dump()
    assert requests[1].startswith(prompts.tool_prompt(compiler_context, validated_plan)[0])
    assert "groups_by_window_midpoint" in requests[1]


def test_direction_contract_rejects_after_analogy_for_before():
    _core, prompts = harness.import_live()
    pair = prompts.DEMONSTRATIONS["local_mat_purpose"]
    trace = [
        {"stage": "Planner", "output": pair["plan"]},
        {"stage": "Tool agent", "output": pair["program"]},
    ]
    case = {"expected_scope": "local_event", "constraints": {"require_relation": "before"}}
    assert "Expected a before temporal relation" in harness.constraint_issues(case, trace)


def test_token_budget_includes_live_schema_system_and_repair_reserve():
    core, prompts = harness.import_live()

    class FixtureTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0] == {"role": "system", "content": "actual system"}
            # Character IDs only for this helper test, never used by real evaluation.
            return list(range(len(messages[1]["content"][0]["text"])))

    plan = copy.deepcopy(prompts.DEMONSTRATIONS["global_workbench"]["plan"])
    rows = harness.budget_audit(
        core, prompts, FixtureTokenizer(), "actual system", {"fixture": plan}
    )
    assert len(rows) == 8
    assert all(
        row["total_tokens"] == row["input_tokens"] + row["reserved_output_tokens"] for row in rows
    )
    assert all("$defs" in row["prompt"] for row in rows)
    repairs = [row for row in rows if row["attempt"] == "max_length_repair_fixture"]
    assert repairs and all("Validation error to fix:" in row["prompt"] for row in repairs)
    assert all("budget_only_invalid_field" in row["prompt"] for row in repairs)
    assert all(
        "Fallible cached timeline" not in row["prompt"] for row in rows if row["stage"] == "Planner"
    )
    assert all(
        '"groups_by_window_midpoint"' in row["prompt"]
        for row in rows
        if row["stage"] == "Tool agent"
    )


def test_offline_case_set_marks_synthetic_vs_real_and_has_open_mode(tmp_path):
    fixture = {
        "example": {"question": "Fixture-only question?", "options": ["A", "B"]},
        "bundle": {
            "scenes": [{"index": 0, "start": 0, "end": 20, "caption": "Fixture cached caption"}]
        },
    }
    for name in ("01-nextqa-sofa.json", "03-egoschema-kitchen.json"):
        (tmp_path / name).write_text(json.dumps(fixture))
    values = harness.cases(tmp_path)
    assert len(values) == 5
    assert values[2]["id"] == "kitchen_open" and values[2]["options"] == []
    assert all(value["kind"] == "real_cached_video_captions" for value in values[:3])
    assert all(value["kind"] == "synthetic_timeline_contract_only" for value in values[3:])
