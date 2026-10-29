"""Offline fixtures exercise strict-callback routing, not provider/model accuracy."""

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from videomasys.zerogpu.core import Caption, LivePipeline


class Detail(BaseModel):
    scene: Literal[2, 4]
    note: str | None = None


class Evidence(BaseModel):
    evidence: list[Detail] = Field(min_length=1)


def forbidden(*_args, **_kwargs):
    raise AssertionError("This generation path must not be used")


def test_explicit_source_schema_images_tokens_and_raw_trace():
    requests = []
    raw = ' {"evidence":[{"scene":2,"note":null}]} '
    images = [Path("/isolated-fixture/frame-2.jpg")]

    def structured(prompt, passed_images, tokens, schema):
        requests.append((prompt, passed_images, tokens, schema))
        return raw

    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    assert pipeline.generate_structured is structured
    trace = []
    result = pipeline.structured(
        trace,
        "Evidence",
        "Observe visible activity.",
        Evidence,
        images,
        333,
        prompt_examples={"version": "fixture-only"},
    )
    prompt, passed_images, tokens, schema = requests[0]
    assert schema == Evidence.model_json_schema()
    assert schema["$defs"]["Detail"]["properties"]["scene"]["enum"] == [2, 4]
    assert schema["$defs"]["Detail"]["properties"]["note"]["default"] is None
    assert passed_images is images and tokens == 333
    assert "matching the supplied response schema" in prompt
    assert "$defs" not in prompt and '"properties"' not in prompt
    assert result.evidence[0].scene == 2
    assert trace[0]["status"] == "complete"
    assert trace[0]["prompt_examples"] == {"version": "fixture-only"}
    assert trace[0]["attempts"] == [
        {
            "prompt": prompt,
            "images": ["frame-2.jpg"],
            "max_new_tokens": 333,
            "response_schema": Evidence.model_json_schema(),
            "raw_response": raw,
        }
    ]


def test_semantic_validator_is_enforced_and_repaired_with_same_schema():
    outputs = iter(['{"caption":"An unsupported fixture."}', '{"caption":"A visible fixture."}'])
    requests = []

    def structured(prompt, images, tokens, schema):
        requests.append((prompt, schema))
        return next(outputs)

    def validate(caption):
        if "unsupported" in caption.caption:
            raise ValueError("Fixture grounding validation failed")

    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    trace = []
    result = pipeline.structured(trace, "Caption", "Describe", Caption, validate=validate)
    assert result.caption == "A visible fixture."
    assert len(requests) == 2
    assert requests[0][1] == requests[1][1] == Caption.model_json_schema()
    assert "Fixture grounding validation failed" in requests[1][0]
    assert 'Your previous invalid response: {"caption":"An unsupported fixture."}' in requests[1][0]
    assert '"properties"' not in requests[1][0]
    assert trace[0]["attempts"][0]["raw_response"] == '{"caption":"An unsupported fixture."}'
    assert trace[0]["attempts"][0]["validation_error"] == "Fixture grounding validation failed"


@pytest.mark.parametrize("raw", ['{"evidence":[{"scene":9}]}', '{"evidence":[]}', "not JSON"])
def test_strict_callback_does_not_bypass_pydantic_or_two_attempt_limit(raw):
    requests = []

    def structured(*args):
        requests.append(args)
        return raw

    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    trace = []
    with pytest.raises(ValueError, match="two attempts"):
        pipeline.structured(trace, "Evidence", "Observe", Evidence)
    assert len(requests) == 2
    assert trace[0]["status"] == "failed" and "output" not in trace[0]
    assert all(attempt["raw_response"] == raw for attempt in trace[0]["attempts"])
    assert all("validation_error" in attempt for attempt in trace[0]["attempts"])


def test_invalid_repetition_not_echoed_in_strict_repair_but_retained_in_trace():
    invalid = '{"caption":"' + "ISOLATED_LOOP_MARKER " * 1000
    responses = iter([invalid, '{"caption":"A concise visible fixture."}'])
    requests = []

    def structured(prompt, *_args):
        requests.append(prompt)
        return next(responses)

    trace = []
    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    pipeline.structured(trace, "Caption", "Describe", Caption)
    assert "ISOLATED_LOOP_MARKER" not in requests[1]
    assert invalid not in requests[1]
    assert "Start fresh" in requests[1]
    assert trace[0]["attempts"][0]["raw_response"] == invalid


def test_adapter_schema_mutations_cannot_change_trace_or_repair_contract():
    requests = []

    def structured(_prompt, _images, _tokens, schema):
        requests.append(json.loads(json.dumps(schema)))
        schema["properties"].clear()
        return "invalid" if len(requests) == 1 else '{"caption":"A valid isolated fixture."}'

    trace = []
    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    pipeline.structured(trace, "Caption", "Describe", Caption)
    assert requests == [Caption.model_json_schema()] * 2
    assert all(a["response_schema"] == Caption.model_json_schema() for a in trace[0]["attempts"])


def test_provider_exception_is_not_silently_replaced_or_retried():
    def structured(*_args):
        raise RuntimeError("isolated provider failure")

    pipeline = LivePipeline(forbidden, None, "test", "test", generate_structured=structured)
    trace = []
    with pytest.raises(RuntimeError, match="isolated provider failure"):
        pipeline.structured(trace, "Caption", "Describe", Caption)
    assert trace[0]["status"] == "failed"
    assert len(trace[0]["attempts"]) == 1
    assert trace[0]["attempts"][0]["error"] == "isolated provider failure"
    assert "raw_response" not in trace[0]["attempts"][0]


def test_legacy_prompt_and_repair_contract_unchanged():
    requests = []
    outputs = iter(["not JSON", '{"caption":"An isolated valid fixture."}'])

    def generate(*args):
        requests.append(args)
        return next(outputs)

    pipeline = LivePipeline(generate, None, "test", "test")
    assert pipeline.generate_structured is None
    trace = []
    pipeline.structured(trace, "Caption", "Describe", Caption)
    contract = json.dumps(Caption.model_json_schema(), separators=(",", ":"))
    assert requests[0] == (
        "Describe\nReturn one JSON object only, no markdown. Keep text fields concise."
        "\nThe required JSON schema is:\n" + contract,
        (),
        500,
    )
    expected_repair = (
        "Describe\nRequired schema: "
        + contract
        + "\nThe previous output was not valid JSON. Start fresh; describe each "
        "action only once and use at most two sentences per text field."
        + "\nValidation error to fix: "
        + trace[0]["attempts"][0]["validation_error"][:1000]
        + "\nReturn a complete corrected JSON object. Correct the field types; "
        "do not repeat the invalid structure. Return no explanation."
    )
    assert requests[1] == (expected_repair, (), 500)
    assert all("response_schema" not in attempt for attempt in trace[0]["attempts"])


def test_plaintext_choice_selection_ignores_structured_callback():
    calls = []

    def generate(*args):
        calls.append(args)
        return "First choice"

    pipeline = LivePipeline(generate, None, "test", "test", generate_structured=forbidden)
    trace = []
    assert (
        pipeline.choose_option(trace, "What happens?", ["First choice", "Second choice"], [])
        == "First choice"
    )
    assert len(calls) == 1 and calls[0][2] == 120
    assert "response_schema" not in trace[0]["attempts"][0]


@pytest.mark.parametrize("mode", ["advisory", "explanation", "answer", "incomplete"])
def test_structured_ui_progress_and_completion_respect_review_scope(tmp_path, monkeypatch, mode):
    """Authored UI callback state; no model/provider calls or altered pipeline decisions."""
    from types import SimpleNamespace

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu import ui

    state = {"trace": [], "scenes": [], "database": str(tmp_path / "cache.sqlite3"),
             "cached": False, "duration": 20}
    monkeypatch.setattr(ui, "prepare_video", lambda *_: state)
    monkeypatch.setattr(ui, "SceneCache", lambda *_: SimpleNamespace(load=lambda *_: ([], [])))

    def qa(bundle, _question, _options):
        review = {"review_scope": "answer_and_explanation", "accepted": mode != "answer",
                  "explanation_accepted": mode != "explanation", "issues": [],
                  "explanation_issues": ["Explain the uncertain location."]
                      if mode == "explanation" else [],
                  "advisories": ["An ancillary detail is uncertain."],
                  "revision_scope": mode if mode in ("explanation", "answer") else "none",
                  "needs_revision": mode in ("explanation", "answer"), "review_complete": True}
        bundle.update(answer={"answer": "A fixture answer.", "inference": "Fixture inference."},
                      verification=review,
                      status="verified" if mode == "advisory" else "unverified",
                      final_answer_complete=False, review_complete=True)
        bundle["trace"].append({"stage": "Verifier", "status": "complete", "revision": 0,
                                "output": review})
        yield bundle
        if mode == "incomplete":
            bundle.update(status="verified", retained_previous_draft=True, review_complete=False)
        else:
            bundle["final_answer_complete"] = True

    demo = ui.build_demo(SimpleNamespace(model_id="fixture", signature="fixture"),
                         lambda bundle: bundle, qa,
                         cache_root=tmp_path, library_root=tmp_path / "none")
    try:
        run = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "run")
        updates = list(run("fixture.mp4", "A fixture question?", "", None))
        messages = "\n".join(update[0] for update in updates)
        final, answer = updates[-1][:2]
        if mode == "explanation":
            assert "Refining the explanation using review feedback." in messages
            assert "Correcting the answer" not in messages
            assert "Answer checked · explanation needs review" in final
            assert "Answer concerns" not in answer
        elif mode == "answer":
            assert "Correcting the answer using review feedback." in messages
            assert "Answer needs review" in final
        elif mode == "advisory":
            assert "Correcting the answer" not in messages
            assert "Refining the explanation" not in messages
            assert "Model check passed" in final and "Review notes" in answer
        else:
            assert 'data-state="error"' in final and "Analysis stopped" in final
            assert "did not complete" in final
            assert "Model-checked" not in answer and "Answer checked" not in answer
        if mode != "incomplete":
            assert 'data-state="complete"' in final
    finally:
        demo.close()
