"""Pure presentation tests use explicit fixtures, never simulated live model calls."""

import copy

import pytest

from videomasys.zerogpu.presentation import (
    gallery_items,
    render_answer,
    render_evidence,
    render_flow,
    render_metrics,
)


def test_real_inference_and_uncertainty_are_visible_escaped_and_optional():
    bundle = {
        "answer": {
            "answer": "An isolated presentation fixture.",
            "inference": "Facts <support> it.",
        },
        "selected": [{"index": 0, "start": 0, "end": 20}],
        "trace": [
            {
                "stage": "Evidence inspection",
                "output": {
                    "evidence": [
                        {
                            "scene": 0,
                            "observation": "Visible test observation.",
                            "relevant": True,
                            "uncertainty": "The <edge> is outside the frame.",
                        }
                    ]
                },
            }
        ],
    }
    answer = render_answer(bundle)
    assert "Why this fits" in answer
    assert "Facts &lt;support&gt; it." in answer
    assert "<support>" not in answer
    for result in (render_evidence(bundle), render_flow(bundle)):
        assert "Uncertainty: The &lt;edge&gt; is outside the frame." in result
        assert "<edge>" not in result
    bundle["answer"]["inference"] = ""
    bundle["trace"][0]["output"]["evidence"][0]["uncertainty"] = ""
    assert "Why this fits" not in render_answer(bundle)
    assert "Uncertainty:" not in render_evidence(bundle)
    assert "Uncertainty:" not in render_flow(bundle)


def test_gallery_labels_use_supplied_times_with_safe_missing_time_fallback():
    bundle = {
        "selected": [
            {
                "index": 2,
                "start": 40,
                "end": 60,
                "frames": ["first.jpg", "last.jpg"],
                "frame_times": [40.25, 59.75],
            },
            {
                "index": 3,
                "start": 60,
                "end": 80,
                "frames": ["untimed.jpg", "invalid.jpg"],
                "frame_times": [None, float("nan")],
            },
            {"index": 4, "start": 80, "end": 100, "frames": ["legacy.jpg"]},
        ]
    }
    assert gallery_items(bundle) == [
        ("first.jpg", "Scene 2 · ≈0:40.2"),
        ("last.jpg", "Scene 2 · ≈0:59.8"),
        ("untimed.jpg", "Scene 3 · 1:00–1:20"),
        ("invalid.jpg", "Scene 3 · 1:00–1:20"),
        ("legacy.jpg", "Scene 4 · 1:20–1:40"),
    ]
    assert gallery_items(None) == []


def test_temporal_retrieval_exposes_separate_scores_and_candidate_not_proof():
    output = {
        "calls": [
            {
                "call": {"id": "r1", "source": "l1", "relation": "after"},
                "score_type": "semantic_cosine",
                "hits": [
                    {
                        "scene": 0,
                        "score": 0.8,
                        "semantic_cosine": 0.8,
                        "temporal_proximity": 1.0,
                        "anchor_overlap": True,
                        "window_relation": "anchor_overlap",
                    },
                    {
                        "scene": 1,
                        "score": 0.7,
                        "semantic_cosine": 0.7,
                        "temporal_proximity": 0.5,
                        "anchor_overlap": False,
                        "window_relation": "strict_after",
                    },
                ],
            }
        ],
        "selected": [{"index": 0, "start": 0, "end": 20, "caption": "Isolated test caption."}],
        "selection_reasons": [
            {
                "scene": 0,
                "reasons": [
                    {
                        "source": "r1",
                        "rank": 1,
                        "purpose": "evidence_need",
                        "evidence_need_ids": ["test_outcome"],
                        "anchor_overlap": True,
                    }
                ],
            }
        ],
        "uncovered_evidence_needs": ["test_missing"],
        "uncovered_temporal_outputs": ["r2"],
    }
    result = render_flow({"trace": [{"stage": "Retrieval", "output": output}]})
    assert "Cosine similarity: 0.8 · Temporal proximity: 1.0" in result
    assert "Temporal proximity: 0.8" not in result
    assert "Overlaps anchor window; event order unproven" in result
    assert "Window after anchor" in result
    assert "Selected via r1 (rank 1) · needs: test_outcome" in result
    assert "overlapping-window candidate, not confirmed event order" in result
    assert "No candidate retained for needs: test_missing" in result
    assert "No candidate retained from temporal outputs: r2" in result


def whole_video_retrieval_output():
    selected = [
        {"index": index, "start": start, "end": start + 20, "caption": "Fixture search hint."}
        for index, start in ((0, 0), (4, 80), (7, 140))
    ]
    return {
        "selected": selected,
        "calls": [],
        "temporal_coverage": {
            "complete": True,
            "required_positions": ["beginning", "middle", "end"],
            "missing_positions": [],
            "representatives": [
                {"position": position, "scene": scene["index"], "start": -1, "end": -1}
                for position, scene in zip(("beginning", "middle", "end"), selected, strict=True)
            ],
        },
    }


def render_retrieval_fixture(output):
    return render_flow({"trace": [{"stage": "Retrieval", "output": output}]})


def test_global_retrieval_shows_actual_selected_representative_spans_and_sampling_limit():
    output = whole_video_retrieval_output()
    result = render_retrieval_fixture(output)
    assert "Temporal coverage: 3/3 video thirds represented" in result
    assert "Beginning: Scene 0 · 0:00–0:20" in result
    assert "Middle: Scene 4 · 1:20–1:40" in result
    assert "End: Scene 7 · 2:20–2:40" in result
    assert "assigned by window midpoint" in result
    assert "Only sampled frames are inspected" in result
    assert "does not mean every event was seen" in result


def test_partial_global_coverage_shows_missing_period_without_claiming_success():
    output = whole_video_retrieval_output()
    output["selected"].pop(1)
    output["temporal_coverage"]["complete"] = False
    output["temporal_coverage"]["missing_positions"] = ["middle"]
    result = render_retrieval_fixture(output)
    assert "Temporal coverage incomplete: 2/3 video thirds represented" in result
    assert "Middle: No representative retained" in result
    assert "Middle: Scene 4" not in result


def test_global_labels_without_actual_selected_representatives_cannot_claim_coverage():
    output = whole_video_retrieval_output()
    output["selected"] = []
    result = render_retrieval_fixture(output)
    assert "Temporal coverage incomplete: 0/3 video thirds represented" in result
    assert result.count("No representative retained") == 3
    assert "Temporal coverage: 3/3" not in result


def test_global_representatives_must_be_distinct_and_controller_failure_stays_visible():
    output = whole_video_retrieval_output()
    for representative in output["temporal_coverage"]["representatives"]:
        representative["scene"] = 0
    result = render_retrieval_fixture(output)
    assert "Temporal coverage incomplete: 1/3 video thirds represented" in result
    output = whole_video_retrieval_output()
    output["temporal_coverage"]["complete"] = False
    assert "Temporal coverage incomplete:" in render_retrieval_fixture(output)


def test_local_retrieval_does_not_gain_a_global_coverage_panel_or_change_output():
    output = whole_video_retrieval_output()
    del output["temporal_coverage"]
    baseline = render_retrieval_fixture(output)
    assert "Temporal coverage" not in baseline
    altered = copy.deepcopy(output)
    altered["temporal_coverage"] = None
    assert render_retrieval_fixture(altered) == baseline


def test_multiple_window_calls_are_not_mislabeled_as_validation_repairs():
    calls = [
        {
            "scene": scene,
            "status": "complete",
            "attempts": [{"images": [f"{scene}.jpg"]}],
            "output": {"scene": scene, "observation": f"Real test observation {scene}."},
        }
        for scene in [2, 5, 8]
    ]
    entry = {
        "stage": "Evidence inspection",
        "status": "complete",
        "calls": calls,
        "repair_count": 0,
        "attempts": [
            {**call["attempts"][0], "scene": call["scene"], "model_call": index}
            for index, call in enumerate(calls, 1)
        ],
    }
    result = render_flow({"trace": [entry]})
    assert "3 model calls" in result
    assert "validation repair" not in result
    calls[1]["attempts"].insert(0, {"validation_error": "Wrong test scene ID."})
    entry["attempts"].insert(
        1, {"scene": 5, "model_call": 2, "validation_error": "Wrong test scene ID."}
    )
    entry["repair_count"] = 1
    result = render_flow({"trace": [entry]})
    assert "4 model calls" in result and "1 validation repair" in result
    assert "3 validation repairs" not in result
    assert "Scene 5 · attempt 1: Wrong test scene ID." in result
    del entry["repair_count"]  # Legacy grouped traces can derive repairs per call.
    assert "1 validation repair" in render_flow({"trace": [entry]})


def test_partial_inspection_failure_still_displays_completed_real_observations():
    bundle = {
        "selected": [
            {"index": scene, "start": scene * 20, "end": (scene + 1) * 20} for scene in [2, 5, 8]
        ],
        "trace": [
            {
                "stage": "Evidence inspection",
                "status": "failed",
                "repair_count": 0,
                "calls": [
                    {
                        "scene": 2,
                        "status": "complete",
                        "attempts": [{}],
                        "output": {
                            "scene": 2,
                            "observation": "Completed <test> observation.",
                            "relevant": True,
                            "uncertainty": "An edge is obscured.",
                        },
                    },
                    {"scene": 5, "status": "failed", "attempts": [{"error": "GPU failed"}]},
                ],
                "attempts": [
                    {"scene": 2, "model_call": 1},
                    {"scene": 5, "model_call": 2, "error": "GPU failed"},
                ],
            }
        ],
    }
    evidence = render_evidence(bundle)
    flow = render_flow(bundle)
    assert "Completed &lt;test&gt; observation." in evidence
    assert evidence.count("Inspection did not complete") == 2
    assert "Inspection pending" not in evidence
    assert "An edge is obscured." in evidence
    assert "Completed &lt;test&gt; observation." in flow
    assert "Scene 5 · attempt 1: GPU failed" in flow
    assert "2 model calls" in flow
    assert "validation repair" not in flow


def test_model_option_comparisons_appear_only_in_expanded_answer_stage():
    answer = {
        "answer": "An isolated answer.",
        "selected_choice": "Test <A>",
        "supporting_scenes": [2],
        "option_comparisons": [
            {
                "choice": "Test <A>",
                "status": "best_supported",
                "reason": "Unique <A> comparison rationale.",
            },
            {
                "choice": "Test B",
                "status": "compatible",
                "reason": "Unique B comparison rationale.",
            },
        ],
    }
    bundle = {
        "answer": answer,
        "trace": [{"stage": "Answer", "status": "complete", "output": answer}],
    }
    result = render_answer(bundle)
    flow = render_flow(bundle)
    assert "Model option comparisons" not in result
    assert "comparison rationale" not in result
    assert "Model option comparisons" in flow
    assert "Test &lt;A&gt; · best supported" in flow
    assert "Test B · compatible" in flow
    assert flow.count("Unique &lt;A&gt; comparison rationale.") == 1
    assert flow.count("Unique B comparison rationale.") == 1
    assert "<A>" not in flow


def test_empty_state_has_no_claimed_result_or_run():
    for render in (render_answer, render_evidence, render_flow, render_metrics):
        assert render(None) == ""
        assert render({}) == ""


def test_answer_preserves_text_choices_timestamps_and_review_limits():
    bundle = {
        "video_name": "painting <clip>.mp4",
        "question": "Why does she move the <cup>?",
        "answer": {
            "answer": "To rinse the brush.\nOnly two frames were sampled.",
            "selected_option": 1,
            "supporting_scenes": [3],
            "limitations": ["Motion between samples is not visible."],
        },
        "options": ["Drink water", "Rinse the brush"],
        "selected": [{"index": 3, "start": 60, "end": 79.5}],
        "verification": {"accepted": True, "issues": []},
        "status": "verified",
    }
    result = render_answer(bundle)
    assert "Model-checked" in result
    assert "Verified answer" not in result
    assert "Why does she move the &lt;cup&gt;?" in result
    assert "painting &lt;clip&gt;.mp4" in result
    assert '<p class="answer-text">' in result
    assert "To rinse the brush.<br>Only two frames were sampled." in result
    assert "Choice 2: Rinse the brush" in result
    assert "Scene 3 · 1:00–1:19.5" in result
    assert "Motion between samples is not visible." in result
    assert "Model review checks the draft" not in result
    assert "not a guarantee of correctness" not in result


def test_rejected_or_partial_answer_stays_a_draft():
    bundle = {
        "answer": {"answer": "A possible explanation.", "supporting_scenes": [5]},
        "verification": {"accepted": False, "issues": ["No observed outcome supports this."]},
        "status": "unverified",
    }
    result = render_answer(bundle)
    assert "Draft · not verified" in result
    assert "Model-checked" not in result
    assert "No observed outcome supports this." in result
    assert "Model review checks the draft" not in result
    assert "Scene 5" in result
    del bundle["status"]
    assert "Draft · not verified" in render_answer(bundle)


def scoped_review_fixture(*, accepted=True, explanation=True):
    review = {
        "review_scope": "answer_and_explanation", "accepted": accepted,
        "explanation_accepted": explanation, "review_complete": True,
        "issues": [] if accepted else ["Selected action is not supported."],
        "explanation_issues": [] if explanation else ["The location is uncertain."],
        "advisories": ["An unchosen option has a <minor> wording disagreement."],
        "revision_scope": "none", "needs_revision": False,
    }
    return {
        "answer": {"answer": "A fixture answer.", "selected_option": 0,
                   "inference": "A fixture explanation.", "limitations": []},
        "options": ["A fixture answer", "Another answer"],
        "verification": review, "final_answer_complete": True, "review_complete": True,
        "retained_previous_draft": False,
        "status": "verified" if accepted and explanation else "unverified",
        "trace": [{"stage": "Verifier", "status": "complete", "output": review}],
    }


def test_structured_advisories_are_separate_notes_not_answer_rejection():
    bundle = scoped_review_fixture()
    original = copy.deepcopy(bundle)
    result, flow = render_answer(bundle), render_flow(bundle)
    assert "Model-checked" in result
    for html in (result, flow):
        assert "Review notes" in html
        assert "&lt;minor&gt;" in html and "<minor>" not in html
        assert "Answer concerns" not in html and "Explanation concerns" not in html
        assert "Draft rejected" not in html and "Answer needs review" not in html
    assert "Main answer accepted by the model reviewer." in flow
    assert bundle == original


@pytest.mark.parametrize("accepted,explanation,badge,heading", [
    (True, False, "Answer checked · explanation needs review", "Explanation concerns"),
    (False, True, "Answer needs review", "Answer concerns"),
])
def test_structured_answer_and_explanation_concerns_are_distinguished(
    accepted, explanation, badge, heading,
):
    bundle = scoped_review_fixture(accepted=accepted, explanation=explanation)
    result, flow = render_answer(bundle), render_flow(bundle)
    assert badge in result and "Model-checked" not in result
    assert heading in result and heading in flow
    assert "A fixture answer." in result
    if accepted:
        assert "Answer concerns" not in result
        assert "Main answer accepted" in flow and "Main answer needs review" not in flow
    else:
        assert "Main answer needs review" in flow


@pytest.mark.parametrize("field,value", [
    ("final_answer_complete", False), ("review_complete", False),
    ("retained_previous_draft", True),
])
def test_incomplete_structured_draft_never_receives_checked_badge(field, value):
    bundle = scoped_review_fixture()
    bundle[field] = value
    result = render_answer(bundle)
    assert "Draft · not verified" in result
    assert "Model-checked" not in result and "Answer checked" not in result


def test_scoped_review_marker_is_required_and_explanation_refinement_is_labeled():
    bundle = scoped_review_fixture()
    bundle["verification"].pop("review_scope")
    bundle["verification"]["accepted"] = False
    flow = render_flow(bundle)
    assert "Draft rejected by the model checker." in flow
    assert "Review notes" not in flow
    bundle = scoped_review_fixture(explanation=False)
    bundle["trace"][0].update(revision=1, revision_scope="explanation")
    flow = render_flow(bundle)
    assert "Explanation refinement 1 of 1" in flow
    assert "Correction 1 of 1" not in flow


def test_failed_optional_refinement_is_explicit_without_promoting_incomplete_draft():
    bundle = scoped_review_fixture(explanation=False)
    bundle.update(explanation_repair_failed=True, retained_reviewed_answer=True)
    result = render_answer(bundle)
    assert "Answer checked · explanation needs review" in result
    assert "Explanation refinement did not complete; the reviewed answer is unchanged." in result
    assert "Model-checked" not in result
    bundle.update(final_answer_complete=False, retained_previous_draft=True)
    result = render_answer(bundle)
    assert "Draft · not verified" in result and "Answer checked" not in result
    assert "the reviewed answer is unchanged" not in result


def test_optional_refinement_warning_appears_once_when_runtime_supplies_advisory():
    bundle = scoped_review_fixture(explanation=False)
    bundle.update(explanation_repair_failed=True, retained_reviewed_answer=True)
    note = "Explanation refinement did not complete; the previously reviewed answer is unchanged."
    bundle["verification"].update(explanation_repair_failed=True, advisories=[note])
    result = render_answer(bundle)
    assert result.count("Explanation refinement did not complete;") == 1
    assert note in result


def test_review_correction_is_labeled_without_hiding_the_original_draft():
    bundle = {
        "trace": [
            {
                "stage": "Answer",
                "status": "complete",
                "output": {"answer": "Original unsupported draft."},
            },
            {
                "stage": "Verifier",
                "status": "complete",
                "output": {
                    "accepted": False,
                    "issues": ["A fact was not observed."],
                    "needs_revision": True,
                },
            },
            {
                "stage": "Answer",
                "status": "complete",
                "revision": 1,
                "output": {"answer": "Corrected evidence-limited draft."},
            },
            {
                "stage": "Verifier",
                "status": "complete",
                "revision": 1,
                "output": {"accepted": False, "issues": ["The cause is still uncertain."]},
            },
        ]
    }
    rendered = render_flow(bundle)
    assert "Original unsupported draft." in rendered
    assert "Corrected evidence-limited draft." in rendered
    assert "A fact was not observed." in rendered
    assert "The cause is still uncertain." in rendered
    assert rendered.count("Correction 1 of 1") == 2


def test_answer_review_notes_deduplicate_exact_matches_without_rewriting():
    bundle = {
        "answer": {
            "answer": "An unverified draft.",
            "limitations": ["Only sampled frames are visible.", "The outcome is unclear."],
        },
        "verification": {
            "accepted": False,
            "issues": [
                "Only sampled frames are visible.",
                "The outcome is unclear.",
                "Review needed.",
            ],
        },
    }
    result = render_answer(bundle)
    assert result.count("Only sampled frames are visible.") == 1
    assert result.count("The outcome is unclear.") == 1
    assert (
        result.index("Only sampled frames")
        < result.index("The outcome")
        < result.index("Review needed")
    )
    assert bundle["verification"]["issues"] == [
        "Only sampled frames are visible.",
        "The outcome is unclear.",
        "Review needed.",
    ]


def test_metrics_only_use_known_measurements_and_union_selected_spans():
    assert render_metrics({"cached": False}) == ""
    result = render_metrics(
        {
            "duration": 90,
            "selected": [
                {"index": 0, "start": 0, "end": 20},
                {"index": 1, "start": 10, "end": 30},
            ],
            "elapsed_seconds": 12.25,
            "trace": [{"stage": "Video cache", "cache_hit": True}],
        }
    )
    assert "1:30" in result
    assert "Selected scenes" in result
    assert ">2<" in result
    assert "Evidence spans" in result
    assert ">30s<" in result
    assert "Reused" in result
    assert "12.2s" in result
    assert "latency" not in result.lower()
    assert "Elapsed" not in render_metrics({"duration": 10})
    assert "Evidence spans" not in render_metrics({"selected": [{"index": 0}]})


def test_plan_and_tool_flow_is_real_output_not_generic_descriptions():
    result = render_flow(
        {
            "trace": [
                {
                    "stage": "Planner",
                    "status": "complete",
                    "latency_ms": 1234,
                    "output": {
                        "objective": "Explain the reason for opening the present.",
                        "evidence_needs": [
                            {
                                "id": "outcome",
                                "description": "What the boy does after receiving the gift.",
                                "temporal_role": "outcome",
                                "depends_on": ["anchor"],
                            }
                        ],
                        "temporal_constraints": ["The opening follows receipt."],
                        "stop_condition": "A visible later action explains the choice.",
                        "reasoning": "The question asks about a later response.",
                        "max_intervals": 2,
                        "max_inspected_seconds": 40,
                    },
                },
                {
                    "stage": "Tool agent",
                    "status": "complete",
                    "output": {
                        "rationale": "Find receipt, then its outcome.",
                        "locate": [{"id": "l1", "query": "boy receives gift", "limit": 2}],
                        "relate": [{"id": "r1", "source": "l1", "relation": "after"}],
                        "inspect": {"sources": ["l1", "r1"]},
                    },
                },
            ]
        }
    )
    assert result.count('<details class="flow-step"') == 2
    assert "Explain the reason for opening the present." in result
    assert "What the boy does after receiving the gift." in result
    assert "Depends on: anchor" in result
    assert "The opening follows receipt." in result
    assert "1.2s" in result
    assert "l1 = locate(" in result
    assert "r1 = relate(" in result
    assert "inspect(" in result
    assert "boy receives gift" in result
    assert "Find receipt, then its outcome." in result


def test_timeline_positions_are_derived_from_clamped_metadata_and_escaped():
    result = render_metrics(
        {
            "duration": 80,
            "selected": [
                {"index": 1, "start": 20, "end": 40},
                {"index": '<img src="x">', "start": -10, "end": 10},
                {"index": 3, "start": 70, "end": 100},
                {"index": 4, "start": 90, "end": 100},
            ],
        }
    )
    assert "Retrieved windows" in result
    assert 'style="left:25%;width:25%"' in result
    assert 'style="left:0%;width:12.5%"' in result
    assert 'style="left:87.5%;width:12.5%"' in result
    assert result.count('class="timeline-window"') == 3
    assert "Scene 1 · 0:20–0:40" in result
    assert "&lt;img src=&quot;x&quot;&gt;" in result
    assert "<img" not in result
    assert "1:20" in result
    assert "Sampled frames from highlighted spans" in result


def test_timeline_handles_empty_retrieval_and_invalid_duration():
    result = render_metrics({"duration": 80, "selected": []})
    assert 'class="timeline-track"' in result
    assert 'class="timeline-window"' not in result
    assert "evidence-timeline" not in render_metrics({"duration": 0, "selected": []})
    assert "evidence-timeline" not in render_metrics({"duration": float("nan"), "selected": []})
    assert "evidence-timeline" not in render_metrics({"duration": float("inf"), "selected": []})
    assert "evidence-timeline" not in render_metrics({"duration": 80})


def test_retrieval_scores_captions_and_observations_are_distinguished():
    result = render_flow(
        {
            "trace": [
                {
                    "stage": "Retrieval",
                    "status": "complete",
                    "output": {
                        "calls": [
                            {
                                "call": {"id": "l1", "query": "opening gift"},
                                "hits": [{"scene": 2, "score": 0.812345}],
                            },
                            {
                                "call": {"id": "r1", "source": "l1", "relation": "after"},
                                "hits": [{"scene": 3, "score": 1.0}],
                            },
                            {"call": "inspect", "selected": [2, 3], "inspected_seconds": 40},
                        ],
                        "selected": [
                            {"index": 2, "start": 40, "end": 60, "caption": "A gift is held."}
                        ],
                    },
                },
                {
                    "stage": "Evidence inspection",
                    "status": "complete",
                    "output": {
                        "evidence": [
                            {"scene": 2, "observation": "Wrapping is torn.", "relevant": True}
                        ]
                    },
                },
            ]
        }
    )
    assert "Cosine similarity: 0.812345" in result
    assert "Temporal proximity: 1.0" in result
    assert "Cached caption — search hint" in result
    assert "A gift is held." in result
    assert "Sampled-frame observation" in result
    assert "Wrapping is torn." in result
    assert "not every frame" in result


def test_failures_and_repairs_visible_but_raw_prompts_are_not_repeated():
    result = render_flow(
        {
            "trace": [
                {
                    "stage": "Planner",
                    "status": "failed",
                    "attempts": [
                        {"prompt": "private raw prompt", "validation_error": "invalid id"},
                        {
                            "raw_response": "unparsed raw response",
                            "validation_error": "duplicate id",
                        },
                    ],
                    "latency_ms": 5000,
                },
                {"stage": "Run", "status": "failed", "error": "Planner could not complete."},
            ]
        }
    )
    assert 'data-step-key="step-0" open' in result
    assert "1 validation repair" in result
    assert "Attempt 1: invalid id" in result
    assert "Attempt 2: duplicate id" in result
    assert "Planner could not complete." in result
    assert "5s" in result
    assert "private raw prompt" not in result
    assert "unparsed raw response" not in result
    assert "Tool agent" not in result
    assert "Verifier" not in result


def test_all_model_and_trace_strings_are_html_escaped():
    attack = '<img src=x onerror="alert(1)">'
    bundle = {
        "question": attack,
        "answer": {"answer": attack, "limitations": [attack]},
        "trace": [
            {"stage": attack, "status": attack, "error": attack, "model": attack},
            {
                "stage": "Tool agent",
                "output": {"locate": [{"id": attack, "query": attack}], "inspect": {}},
            },
            {"stage": "Verifier", "output": {"accepted": False, "issues": [attack]}},
        ],
    }
    for result in (render_answer(bundle), render_flow(bundle)):
        assert "<img" not in result
        assert "&lt;img" in result
        assert "&quot;alert(1)&quot;" in result


def test_evidence_cards_distinguish_pending_observations_and_citations():
    bundle = {
        "selected": [
            {"index": 0, "start": 0, "end": 20, "caption": "A cup <appears>."},
            {"index": 1, "start": 20, "end": 40, "caption": "A brush moves."},
        ],
        "answer": {"supporting_scenes": [1]},
        "trace": [
            {
                "stage": "Evidence inspection",
                "output": {
                    "evidence": [
                        {"scene": 1, "observation": "The brush enters <water>.", "relevant": True}
                    ]
                },
            }
        ],
    }
    result = render_evidence(bundle)
    assert result.startswith('<div class="evidence-grid">')
    assert result.count('<section class="evidence-card">') == 2
    assert "Scene 0 · 0:00–0:20" in result
    assert "Scene 1 · 0:20–0:40" in result
    assert result.count("Cited in answer") == 1
    assert result.count("Inspection pending") == 1
    assert "Sampled-frame observation" in result
    assert "Marked relevant" in result
    assert "The brush enters &lt;water&gt;." in result
    assert "Cached caption · search hint" in result
    assert "A cup &lt;appears&gt;." in result
    assert "<appears>" not in result
    assert "<water>" not in result
    bundle["trace"] = [{"stage": "Evidence inspection", "status": "failed"}]
    assert "Inspection pending" not in render_evidence(bundle)
    assert render_evidence(bundle).count("Inspection did not complete") == 2
    bundle["trace"] = [{"stage": "Run", "status": "failed", "error": "GPU unavailable"}]
    assert "Inspection pending" not in render_evidence(bundle)
    assert render_evidence(bundle).count("Inspection did not complete") == 2
    bundle["selected"] = []
    assert render_evidence(bundle) == ""


def test_consecutive_caption_rows_share_one_group_without_changing_trace():
    import copy

    bundle = {
        "trace": [
            {
                "stage": f"Caption scene {index}",
                "status": "complete",
                "latency_ms": 1234,
                "output": {"caption": f"Actual caption {index}: <literal> text."},
            }
            for index in range(3)
        ]
        + [{"stage": "Video cache", "status": "complete", "cache_hit": False}]
        + [{"stage": "Planner", "status": "running"}]
    }
    original = copy.deepcopy(bundle)
    result = render_flow(bundle)
    assert result.count('<details class="flow-step"') == 3
    assert "Scene captioning · 3 windows" in result
    assert "3 complete" in result
    for index in range(3):
        assert f"Caption scene {index}" in result
        assert f"Actual caption {index}: &lt;literal&gt; text." in result
    assert result.count("1.2s") == 3
    assert result.index("Actual caption 2") < result.index("Video cache") < result.index("Planner")
    assert bundle == original


def test_caption_groups_preserve_partial_errors_and_only_group_adjacent_entries():
    result = render_flow(
        {
            "trace": [
                {"stage": "Caption scene 0", "status": "complete", "output": {"caption": "One"}},
                {
                    "stage": "Caption scene 1",
                    "status": "failed",
                    "attempts": [
                        {"validation_error": "Missing caption"},
                        {"error": "GPU <unavailable>"},
                    ],
                },
                {"stage": "Run", "status": "failed", "error": "Stopped"},
                {"stage": "Caption scene 2", "status": "running"},
            ]
        }
    )
    assert result.count("Scene captioning") == 2
    assert result.count('<details class="flow-step"') == 3
    assert '<details class="flow-step" data-step-key="step-0" open>' in result
    assert "1 complete · 1 failed" in result
    assert "Scene captioning · 1 window" in result
    assert "1 running" in result
    assert "1 validation repair" in result
    assert "Missing caption" in result
    assert "GPU &lt;unavailable&gt;" in result
    assert (
        result.index("Caption scene 1") < result.index("Stopped") < result.index("Caption scene 2")
    )
