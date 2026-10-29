"""Retrieval tests use deterministic vector fixtures, never benchmark answer labels."""

import numpy as np
import pytest

from videomasys.domain.schemas import ToolProgram
from videomasys.zerogpu.retrieval import execute_program


def scenes(count=6, span=20):
    return [
        {
            "index": i + 10,
            "start": float(i * span),
            "end": float((i + 1) * span),
            "caption": f"Isolated retrieval fixture {i}",
            "frames": [],
        }
        for i in range(count)
    ]


def program(**updates):
    return ToolProgram.model_validate(
        {
            "rationale": "Find an anchor and inspect its subsequent outcome.",
            "max_intervals": 3,
            "max_inspected_seconds": 60,
            "locate": [
                {
                    "id": "l1",
                    "query": "Initial action",
                    "limit": 4,
                    "evidence_need_ids": ["anchor"],
                },
                {
                    "id": "l2",
                    "query": "Subsequent activity",
                    "limit": 4,
                    "evidence_need_ids": ["outcome"],
                },
            ],
            "relate": [
                {
                    "id": "r1",
                    "source": "l1",
                    "relation": "after",
                    "limit": 4,
                    "horizon_seconds": 20,
                    "evidence_need_ids": ["outcome"],
                }
            ],
            "inspect": {"sources": ["l1", "l2", "r1"], "evidence_need_ids": ["anchor", "outcome"]},
            **updates,
        }
    )


def vectors():
    # The globally closest outcome (row 0) is BEFORE the anchor. Row 3 is the
    # semantically strongest temporally eligible outcome, not its nearest neighbor.
    return np.array([[0, 1], [1, 0], [0.8, 0.6], [0.1, 0.995], [-0.3, 0.95], [-1, 0]])


def run(prog=None, clip=None, embeddings=None, queries=None):
    return execute_program(
        prog or program(),
        clip or scenes(),
        vectors() if embeddings is None else embeddings,
        [[1, 0], [0, 1]] if queries is None else queries,
    )


def test_real_cosine_normalizes_embeddings_and_exposes_score_domain():
    embeddings = vectors()
    embeddings[1] *= 7
    result = run(embeddings=embeddings, queries=[[5, 0], [0, 3]])
    locate = result["calls"][0]
    assert locate["score_type"] == "semantic_cosine"
    assert locate["hits"][0] == {"scene": 11, "score": 1.0, "semantic_cosine": 1.0}


def test_temporal_reranking_uses_target_query_not_anchor_or_proximity_score():
    temporal = run()["calls"][2]
    assert temporal["anchor_scene"] == 11
    assert temporal["target_query_sources"] == ["l2"]
    assert temporal["ranking"] == ["semantic_cosine", "temporal_proximity"]
    assert temporal["hits"][0]["scene"] == 13
    assert temporal["hits"][0]["semantic_cosine"] > 0.99
    assert temporal["hits"][0]["temporal_proximity"] == 0.0
    assert temporal["hits"][1]["scene"] == 12
    assert temporal["hits"][1]["temporal_proximity"] == 1.0
    assert 10 not in [hit["scene"] for hit in temporal["hits"]]


@pytest.mark.parametrize("relation", ["before", "after", "around", "during"])
def test_anchor_overlap_is_eligible_but_never_claims_event_ordering(relation):
    prog = program()
    prog.relate[0].relation = relation
    temporal = run(prog)["calls"][2]
    anchor = next(hit for hit in temporal["hits"] if hit["scene"] == 11)
    assert anchor["anchor_overlap"] is True
    assert anchor["window_relation"] == "anchor_overlap"
    assert "does not establish event ordering" in temporal["ordering_scope"]
    if relation == "during":
        assert [hit["scene"] for hit in temporal["hits"]] == [11]
    elif relation == "before":
        assert {hit["scene"] for hit in temporal["hits"]} == {10, 11}


def test_inspection_preserves_anchor_and_temporal_target_not_unconstrained_runner_up():
    result = run(program(max_intervals=2, max_inspected_seconds=40))
    assert [scene["index"] for scene in result["selected"]] == [11, 13]
    assert not result["uncovered_evidence_needs"]
    assert not result["uncovered_temporal_outputs"]
    outcome = result["selection_reasons"][1]["reasons"][0]
    assert outcome["source"] == "r1"
    assert outcome["evidence_need_ids"] == ["outcome"]


def test_spare_slot_prefers_relevant_temporal_candidate_to_unrelated_earlier_scene():
    result = run()
    assert [scene["index"] for scene in result["selected"]] == [11, 12, 13]
    assert result["calls"][-1]["inspected_seconds"] == 60
    assert result["selection_reasons"][1]["reasons"][0]["source"] == "r1"


def test_uncovered_need_and_temporal_output_are_explicit_when_budget_is_too_small():
    prog = program(max_intervals=1, max_inspected_seconds=20)
    prog.relate[0].limit = 2  # The anchor isn't a candidate for this target's top two.
    result = run(prog)
    assert [scene["index"] for scene in result["selected"]] == [11]
    assert result["uncovered_evidence_needs"] == ["outcome"]
    assert result["uncovered_temporal_outputs"] == ["r1"]


def test_hard_demo_budgets_cannot_be_relaxed_by_program():
    result = run(program(max_intervals=8, max_inspected_seconds=240))
    assert len(result["selected"]) == 3
    assert result["calls"][-1]["budget_intervals"] == 3
    assert result["calls"][-1]["budget_seconds"] == 60
    tiny = run(program(max_inspected_seconds=5))
    assert tiny["selected"] == []
    assert tiny["uncovered_evidence_needs"] == ["anchor", "outcome"]


def test_duplicate_candidates_do_not_use_extra_budget_and_output_is_chronological():
    result = run(queries=[[1, 0], [1, 0]])
    selected = result["selected"]
    assert len({scene["index"] for scene in selected}) == len(selected)
    assert [scene["start"] for scene in selected] == sorted(s["start"] for s in selected)
    assert result["calls"][-1]["inspected_seconds"] == sum(s["end"] - s["start"] for s in selected)


@pytest.mark.parametrize(
    "position,expected", [("beginning", [10, 11]), ("middle", [12, 13]), ("end", [14, 15])]
)
def test_locate_can_restrict_search_to_a_video_third(position, expected):
    prog = program()
    prog.locate[0].temporal_position = position
    result = run(prog, embeddings=np.ones((6, 2)))
    assert [hit["scene"] for hit in result["calls"][0]["hits"]] == expected
    assert result["calls"][0]["temporal_filter"]["position"] == position


@pytest.mark.parametrize("position", ["beginning", "middle", "end"])
def test_coarse_single_window_is_not_lost_by_video_third_filter(position):
    prog = program()
    prog.locate[0].temporal_position = position
    result = run(prog, clip=scenes(1), embeddings=[[1, 1]])
    assert [hit["scene"] for hit in result["calls"][0]["hits"]] == [10]


def test_relation_without_target_query_reports_only_temporal_proximity():
    prog = program()
    prog.locate = prog.locate[:1]
    prog.inspect.sources = ["l1", "r1"]
    temporal = run(prog, queries=[[1, 0]])["calls"][1]
    assert temporal["score_type"] == "temporal_proximity"
    assert temporal["target_query_sources"] == []
    assert all("semantic_cosine" not in hit and "score" not in hit for hit in temporal["hits"])


def test_relation_chain_follows_best_reranked_output():
    prog = program()
    prog.relate.append(prog.relate[0].model_copy(update={"id": "r2", "source": "r1"}))
    prog.inspect.sources.append("r2")
    result = run(prog)
    assert result["calls"][2]["hits"][0]["scene"] == 13
    assert result["calls"][3]["anchor_scene"] == 13


@pytest.mark.parametrize("bad", [np.full((6, 2), np.nan), np.zeros((6, 2)), np.ones((5, 2))])
def test_invalid_embedding_data_fails_explicitly(bad):
    with pytest.raises(ValueError, match="Scene embeddings"):
        run(embeddings=bad)


def test_execution_is_deterministic_and_does_not_mutate_inputs():
    prog, clip, embedding = program(), scenes(), vectors()
    original = prog.model_dump()
    first = run(prog, clip, embedding)
    assert first == run(prog, clip, embedding)
    assert prog.model_dump() == original
    assert all("selection_reasons" not in scene for scene in clip)


def whole_program(**updates):
    return program(
        locate=[
            {
                "id": f"l{index}",
                "query": f"Representative actions in the {position}",
                "limit": 4,
                "evidence_need_ids": [need],
                "temporal_position": position,
            }
            for index, (need, position) in enumerate(
                zip(("early", "middle", "late"), ("beginning", "middle", "end"), strict=True),
                start=1,
            )
        ],
        relate=[],
        inspect={"sources": ["l1", "l2", "l3"], "evidence_need_ids": ["early", "middle", "late"]},
        **updates,
    )


def whole_run(prog=None, clip=None, embeddings=None, queries=None):
    clip = scenes(9) if clip is None else clip
    return execute_program(
        whole_program() if prog is None else prog,
        clip,
        np.ones((len(clip), 3)) if embeddings is None else embeddings,
        np.eye(3) if queries is None else queries,
        scope="whole_video",
    )


def spans(*intervals):
    return [
        {**scene, "start": start, "end": end}
        for scene, (start, end) in zip(scenes(len(intervals)), intervals, strict=True)
    ]


def test_whole_video_requires_real_distinct_representatives_not_just_need_labels():
    result = whole_run()
    assert [scene["index"] for scene in result["selected"]] == [10, 13, 16]
    assert result["temporal_coverage"]["complete"] is True
    assert result["temporal_coverage"]["missing_positions"] == []
    assert result["temporal_coverage"] == result["calls"][-1]["temporal_coverage"]
    assert [row["position"] for row in result["temporal_coverage"]["representatives"]] == [
        "beginning",
        "middle",
        "end",
    ]
    assert all(
        call["temporal_filter"]["semantics"] == "scene_midpoint" for call in result["calls"][:3]
    )
    assert not result["uncovered_evidence_needs"]


def test_global_semantic_preference_cannot_escape_a_target_third():
    embeddings = np.array([[1, 0, 0]] * 3 + [[0, 1, 1]] * 3 + [[0, 0, -1]] * 3)
    result = whole_run(embeddings=embeddings)
    assert [scene["index"] for scene in result["selected"]] == [10, 13, 16]
    assert result["calls"][2]["hits"][0]["semantic_cosine"] == -1
    assert result["temporal_coverage"]["complete"] is True


def test_midpoints_on_third_boundaries_belong_only_to_the_later_third():
    # D=90; midpoint30 is middle and midpoint60 is end, despite both crossing boundaries.
    result = whole_run(clip=spans((0, 10), (20, 40), (50, 70), (80, 90)))
    assert [[hit["scene"] for hit in call["hits"]] for call in result["calls"][:3]] == [
        [10],
        [11],
        [12, 13],
    ]
    assert result["temporal_coverage"]["complete"] is True


@pytest.mark.parametrize("clip", [spans((0, 30)), spans((0, 20), (20, 40))])
def test_one_or_two_indexed_windows_cannot_fake_full_coverage(clip):
    result = whole_run(clip=clip)
    assert result["temporal_coverage"]["complete"] is False
    assert result["temporal_coverage"]["missing_positions"]
    assert result["uncovered_evidence_needs"]
    assert len(result["selected"]) == len(clip)
    assert len({scene["index"] for scene in result["selected"]}) == len(clip)


def test_short_video_with_three_actual_small_windows_can_have_full_coverage():
    result = whole_run(clip=scenes(3, span=1))
    assert result["temporal_coverage"]["complete"] is True
    assert result["calls"][-1]["inspected_seconds"] == 3


def test_overlapping_straddler_cannot_satisfy_multiple_periods():
    result = whole_run(clip=spans((0, 10), (10, 80), (80, 90)))
    assert [call["temporal_filter"]["semantics"] for call in result["calls"][:3]] == [
        "scene_midpoint",
        "scene_midpoint",
        "scene_midpoint",
    ]
    assert [[hit["scene"] for hit in call["hits"]] for call in result["calls"][:3]] == [
        [10],
        [11],
        [12],
    ]
    # The 70-second middle window cannot fit: keeping two other windows is not success.
    assert result["temporal_coverage"]["complete"] is False
    assert result["temporal_coverage"]["missing_positions"] == ["middle"]
    assert result["uncovered_evidence_needs"] == ["middle"]


def test_joint_selection_can_use_shorter_runner_up_when_greedy_top_hit_would_fail():
    clip = spans((0, 30), (30, 40), (60, 80), (140, 160))
    # D=160; the first two are both beginning. Only the runner-up allows all 3 in 60s.
    result = whole_run(clip=clip, embeddings=[[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]])
    assert result["calls"][0]["hits"][0]["scene"] == 10
    assert result["temporal_coverage"]["complete"] is True
    assert [scene["index"] for scene in result["selected"]] == [11, 12, 13]
    assert result["selection_reasons"][0]["reasons"][0]["rank"] == 2
    assert result["calls"][-1]["inspected_seconds"] == 50


@pytest.mark.parametrize("updates", [{"max_intervals": 2}, {"max_inspected_seconds": 40}])
def test_whole_video_budget_shortfall_is_reported_even_with_all_query_candidates(updates):
    result = whole_run(whole_program(**updates))
    assert result["temporal_coverage"]["candidate_counts"] == {
        "beginning": 3,
        "middle": 3,
        "end": 3,
    }
    assert result["temporal_coverage"]["complete"] is False
    assert len(result["selected"]) == 2
    assert len(result["uncovered_evidence_needs"]) == 1


def test_whole_video_does_not_relax_demo_budget_or_silently_expand_top_k():
    prog = whole_program(max_intervals=8, max_inspected_seconds=240)
    # The only short beginning candidate is rank 3; limit 2 must still be respected.
    prog.locate[0].limit = 2
    clip = spans((0, 30), (1, 31), (30, 40), (60, 80), (140, 160))
    result = whole_run(prog, clip, [[1, 0, 0], [0.99, 0.1, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]])
    assert len(result["calls"][0]["hits"]) == 2
    assert result["temporal_coverage"]["complete"] is False
    assert result["calls"][-1]["budget_seconds"] == 60
    assert result["calls"][-1]["budget_intervals"] == 3


@pytest.mark.parametrize(
    "problem", ["relation", "wrong_position", "multiple_needs", "missing_inspect"]
)
def test_executor_defensively_rejects_malformed_global_programs(problem):
    prog = whole_program()
    if problem == "relation":
        prog.relate = program().relate
    elif problem == "wrong_position":
        prog.locate[2].temporal_position = "middle"
    elif problem == "multiple_needs":
        prog.locate[0].evidence_need_ids.append("late")
    else:
        prog.inspect.sources.pop()
    with pytest.raises(ValueError, match="Whole-video retrieval requires exactly"):
        whole_run(prog)


def test_default_local_retrieval_is_unchanged_and_scope_is_never_inferred():
    prog = whole_program()
    clip = spans((0, 30))
    result = execute_program(prog, clip, [[1, 1, 1]], np.eye(3))
    assert "temporal_coverage" not in result
    assert all(
        call["temporal_filter"]["semantics"] == "window_overlap" for call in result["calls"][:3]
    )
    assert result["uncovered_evidence_needs"] == []
    assert len(result["selected"]) == 1


def test_whole_video_is_deterministic_and_preserves_inputs():
    prog, clip, embeddings = whole_program(), scenes(9), np.ones((9, 3))
    before = prog.model_dump()
    first = whole_run(prog, clip, embeddings)
    assert first == whole_run(prog, clip, embeddings)
    assert prog.model_dump() == before
    assert all("selection_reasons" not in scene for scene in clip)


@pytest.mark.parametrize("span,complete", [(20.0000000001, True), (20.00000001, False)])
def test_whole_video_seconds_budget_uses_only_existing_numerical_tolerance(span, complete):
    result = whole_run(clip=scenes(3, span=span))
    assert result["temporal_coverage"]["complete"] is complete
    assert result["calls"][-1]["inspected_seconds"] <= 60 + 1e-9


def test_global_all_candidates_over_budget_returns_full_failure_diagnostics():
    result = whole_run(whole_program(max_inspected_seconds=5))
    assert result["selected"] == []
    assert result["temporal_coverage"]["complete"] is False
    assert result["temporal_coverage"]["candidate_scope"] == "returned_locate_hits"
    assert result["temporal_coverage"]["missing_positions"] == ["beginning", "middle", "end"]
    assert result["uncovered_evidence_needs"] == ["early", "middle", "late"]
    assert result["calls"][-1]["inspected_seconds"] == 0


def test_unknown_retrieval_scope_is_not_silently_treated_as_local():
    with pytest.raises(ValueError, match="Unknown retrieval scope"):
        execute_program(whole_program(), scenes(3), np.ones((3, 3)), np.eye(3), scope="typo")
