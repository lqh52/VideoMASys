"""Synthetic, offline v3 dataflow tests; no benchmark labels or model calls."""

from copy import deepcopy

import numpy as np
import pytest

from videomasys.zerogpu import svp_retrieval as runtime
from videomasys.zerogpu.svp_retrieval import (
    SVPPlan,
    SVPProgram,
    SVPProgramValidator,
    execute_svp_program,
    validate_svp_task_program,
)


def clips(count=6, span=10):
    return [
        {
            "index": index + 10,
            "start": float(index * span),
            "end": float((index + 1) * span),
            "caption": f"Synthetic scene {index}",
            "frames": [],
        }
        for index in range(count)
    ]


def plan(**updates):
    return SVPPlan.model_validate(
        {
            "scope": "local_event",
            "objective": "Find the activity after an arrival",
            "evidence_needs": [
                {"id": "arrival", "description": "Initial arrival", "temporal_role": "anchor"},
                {
                    "id": "activity",
                    "description": "Subsequent activity",
                    "temporal_role": "outcome",
                    "depends_on": ["arrival"],
                },
            ],
            "stop_condition": "Relevant activity is visible",
            "reasoning": "Follow the event",
            **updates,
        }
    )


def payload(**updates):
    return {
        "version": "3.0",
        "rationale": "Narrow an anchor then inspect its subsequent activity",
        "steps": [
            {
                "op": "locate",
                "id": "arrival",
                "source": "video",
                "query": "Arrival event",
                "limit": 1,
                "evidence_need_ids": ["arrival"],
            },
            {
                "op": "relate",
                "id": "after_arrival",
                "source": "arrival",
                "relation": "after",
                "horizon_seconds": 20,
                "limit": 4,
                "evidence_need_ids": [],
            },
            {
                "op": "locate",
                "id": "activity",
                "source": "after_arrival",
                "query": "Activity",
                "limit": 1,
                "evidence_need_ids": ["activity"],
            },
        ],
        "inspect": {"sources": ["activity"], "evidence_need_ids": ["arrival", "activity"]},
        **updates,
    }


def program(**updates):
    return SVPProgram.model_validate(payload(**updates))


def embeddings():
    # Globally closest activity is BEFORE the anchor, so a scoped search must
    # reject it. The distant scene 4 is more semantic than the nearer scene 2.
    return np.array([[0, 1], [1, 0], [0.8, 0.6], [0.7, 0.7], [0.1, 0.99], [-1, 0]])


def run(prog=None, scenes=None, vectors=None, queries=None, **kwargs):
    return execute_svp_program(
        prog or program(),
        clips() if scenes is None else scenes,
        embeddings() if vectors is None else vectors,
        [[1, 0], [0, 1]] if queries is None else queries,
        **kwargs,
    )


def test_dependency_can_be_satisfied_by_relation_then_scoped_locate():
    assert SVPProgramValidator().validate(program(), plan()).version == "3.0"
    assert [step.id for step in program().locate] == ["arrival", "activity"]
    assert [step.id for step in program().relate] == ["after_arrival"]


@pytest.mark.parametrize("role", ["anchor", "context"])
def test_directionless_dependency_accepts_real_scoped_search_without_added_relation(role):
    value = plan()
    value.evidence_needs[1].temporal_role = role
    data = payload()
    data["steps"] = [data["steps"][0], data["steps"][2]]
    data["steps"][1]["source"] = "arrival"
    prog = SVPProgram.model_validate(data)
    original = deepcopy(prog.model_dump())
    assert SVPProgramValidator().validate(prog, value) is prog
    assert prog.model_dump() == original
    result = run(prog)
    # Refinement searches the real anchor output only. It neither adds a
    # temporal hop nor retrieves the globally better activity scene 14.
    assert len(result["calls"]) == 3  # Two searches and inspect.
    assert result["calls"][1]["candidate_scope_scene_ids"] == [11]
    assert result["calls"][1]["ancestry"] == ["arrival", "activity"]
    assert [scene["index"] for scene in result["selected"]] == [11]


@pytest.mark.parametrize("same_step_tags", [False, True])
def test_context_dependency_rejects_independent_search_even_with_same_step_tags(same_step_tags):
    value = plan()
    value.evidence_needs[1].temporal_role = "context"
    data = payload()
    data["steps"] = [data["steps"][0], data["steps"][2]]
    data["steps"][1]["source"] = "video"
    if same_step_tags:
        data["steps"][1]["evidence_need_ids"] = ["arrival", "activity"]
    data["inspect"]["sources"] = ["arrival", "activity"]
    with pytest.raises(ValueError, match="requires a source-scoped target"):
        SVPProgramValidator().validate(SVPProgram.model_validate(data), value)


@pytest.mark.parametrize("role,direction", [("before", "before"), ("after", "after"),
                                           ("outcome", "after")])
def test_directional_dependency_cannot_be_replaced_by_scoped_locate(role, direction):
    value = plan()
    value.evidence_needs[1].temporal_role = role
    data = payload()
    data["steps"] = [data["steps"][0], data["steps"][2]]
    data["steps"][1]["source"] = "arrival"
    with pytest.raises(ValueError, match=rf"requires relate\({direction}\)"):
        SVPProgramValidator().validate(SVPProgram.model_validate(data), value)


def test_dependency_repair_reports_all_edges_and_keeps_the_second_temporal_hop():
    value = plan(evidence_needs=[
        {"id": "arrival", "description": "Initial event", "temporal_role": "anchor"},
        {"id": "activity", "description": "Second event", "temporal_role": "after",
         "depends_on": ["arrival"]},
        {"id": "result", "description": "Later result", "temporal_role": "outcome",
         "depends_on": ["arrival", "activity"]},
    ])
    data = payload()
    last = {"op": "locate", "id": "result", "source": "activity", "query": "Later result",
            "evidence_need_ids": ["result"]}
    data["steps"] = [data["steps"][0], data["steps"][2], last]
    data["steps"][1]["source"] = "arrival"
    data["inspect"] = {"sources": ["result"],
                       "evidence_need_ids": ["arrival", "activity", "result"]}
    with pytest.raises(ValueError) as error:
        SVPProgramValidator().validate(SVPProgram.model_validate(data), value)
    for edge in ("arrival -> activity", "arrival -> result", "activity -> result"):
        assert f"Dependency {edge} requires relate(after)" in str(error.value)
    assert len(str(error.value)) < 1000  # Fits the existing bounded repair feedback.
    data["steps"] = [*payload()["steps"], last]
    with pytest.raises(ValueError, match=r"activity -> result requires relate\(after\)"):
        SVPProgramValidator().validate(SVPProgram.model_validate(data), value)
    data["steps"].insert(-1, {"op": "relate", "id": "after_activity", "source": "activity",
                              "relation": "after", "evidence_need_ids": []})
    data["steps"][-1]["source"] = "after_activity"
    SVPProgramValidator().validate(SVPProgram.model_validate(data), value)


def test_scoped_search_does_not_broaden_to_globally_best_hit():
    result = run()
    assert result["calls"][0]["hits"][0]["scene"] == 11
    assert result["calls"][2]["source_scene_ids"] == [14, 13, 12, 11]
    assert result["calls"][2]["candidate_scope_scene_ids"] == [14, 13, 12, 11]
    assert result["calls"][2]["hits"][0]["scene"] == 14
    assert [scene["index"] for scene in result["selected"]] == [14]
    assert 10 not in result["calls"][2]["candidate_scope_scene_ids"]


def test_ancestor_is_navigation_not_forced_visual_inspection():
    result = run()
    assert result["coverage"]["navigation_only_needs"] == ["arrival"]
    assert result["coverage"]["need_sources"] == {"arrival": ["activity"], "activity": ["activity"]}
    assert result["uncovered_evidence_needs"] == []
    assert result["uncovered_sources"] == []
    assert result["uncovered_temporal_outputs"] == []
    reason = result["selection_reasons"][0]["reasons"][0]
    assert reason["evidence_need_ids"] == ["activity"]
    assert reason["lineage_evidence_need_ids"] == ["arrival", "activity"]
    assert reason["ancestry"] == ["arrival", "after_arrival", "activity"]
    assert "not_visual_confirmation" in result["coverage"]["scope"]


def test_temporal_candidates_rank_the_declared_child_target_before_limiting():
    result = run()
    relation = result["calls"][1]
    assert relation["anchor_scene"] == 11
    assert [hit["scene"] for hit in relation["hits"]] == [14, 13, 12, 11]
    assert relation["score_type"] == "semantic_cosine"
    assert relation["target_query_sources"] == ["activity"]
    assert relation["limit_policy"] == "rank_eligible_candidates_before_limit"
    assert relation["ranking"] == ["semantic_cosine", "temporal_proximity", "strict_before_overlap"]
    for hit in relation["hits"]:
        assert hit["semantic_cosine"] == hit["target_query_cosines"]["activity"]
    changed = run(queries=[[1, 0], [-1, 0]])
    assert changed["calls"][1] != relation
    assert changed["calls"][1]["anchor_scene"] == 11
    assert set(changed["calls"][1]["candidate_scope_scene_ids"]) == {11, 12, 13, 14}


def test_late_semantic_match_survives_small_relation_limit_without_widening_horizon():
    data = payload()
    data["steps"][1]["limit"] = 1
    result = run(SVPProgram.model_validate(data))
    assert [hit["scene"] for hit in result["calls"][1]["hits"]] == [14]
    assert result["calls"][2]["source_scene_ids"] == [14]
    assert [scene["index"] for scene in result["selected"]] == [14]
    # The even stronger global match is before the anchor and never eligible.
    assert 10 not in result["calls"][1]["candidate_scope_scene_ids"]
    data["steps"][1]["horizon_seconds"] = 10
    shorter = run(SVPProgram.model_validate(data))
    assert 14 not in shorter["calls"][1]["candidate_scope_scene_ids"]
    assert [scene["index"] for scene in shorter["selected"]] == [13]


def test_unrelated_query_does_not_change_temporal_ranking():
    data = payload()
    data["steps"].append(
        {
            "op": "locate",
            "id": "unrelated",
            "source": "video",
            "query": "Unrelated appearance",
            "evidence_need_ids": ["arrival"],
        }
    )
    data["inspect"]["sources"].append("unrelated")
    prog = SVPProgram.model_validate(data)
    first = run(prog, queries=[[1, 0], [0, 1], [1, 1]])
    changed = run(prog, queries=[[1, 0], [0, 1], [-1, -1]])
    assert first["calls"][1] == changed["calls"][1]
    assert first["calls"][1]["target_query_sources"] == ["activity"]


def test_same_need_video_locate_can_supply_old_style_temporal_target():
    data = payload()
    anchor, relation, target = data["steps"]
    target["source"] = "video"
    relation.update(evidence_need_ids=["activity"], limit=1)
    data["steps"] = [anchor, target, relation]
    data["inspect"]["sources"] = ["arrival", "activity", "after_arrival"]
    result = run(SVPProgram.model_validate(data))
    assert result["calls"][2]["target_query_sources"] == ["activity"]
    assert [hit["scene"] for hit in result["calls"][2]["hits"]] == [14]
    assert result["coverage"]["need_sources"]["activity"] == ["after_arrival"]


def test_relation_without_a_declared_target_retains_proximity_ranking():
    data = payload()
    data["steps"] = data["steps"][:2]
    data["steps"][1]["evidence_need_ids"] = ["activity"]
    data["inspect"]["sources"] = ["after_arrival"]
    result = run(SVPProgram.model_validate(data), queries=[[1, 0]])
    relation = result["calls"][1]
    assert relation["target_query_sources"] == []
    assert relation["score_type"] == "temporal_proximity"
    assert [hit["scene"] for hit in relation["hits"]] == [12, 11, 13, 14]
    assert all("semantic_cosine" not in hit for hit in relation["hits"])


@pytest.mark.parametrize("relation", ["before", "after", "during", "around"])
def test_anchor_overlap_never_claims_event_order(relation):
    data = payload()
    data["steps"][1]["relation"] = relation
    data["steps"][1]["limit"] = 12
    result = run(SVPProgram.model_validate(data))
    trace = result["calls"][1]
    overlap = next(hit for hit in trace["hits"] if hit["scene"] == 11)
    assert overlap["anchor_overlap"] is True
    assert overlap["window_relation"] == "anchor_overlap"
    assert "does not establish event ordering" in trace["ordering_scope"]
    if relation == "during":
        assert [hit["scene"] for hit in trace["hits"]] == [11]


def test_final_hit_retains_temporal_source_provenance():
    hit = run()["calls"][2]["hits"][0]
    assert hit["source_hit"]["anchor_scene"] == 11
    assert hit["source_hit"]["window_relation"] == "strict_after"
    assert "position" not in hit["source_hit"]


def test_scoped_locate_can_follow_locate_and_use_empty_intermediate_labels():
    data = payload()
    data["steps"] = [data["steps"][0], data["steps"][2]]
    data["steps"][0].update(limit=2, evidence_need_ids=[])
    data["steps"][1].update(source="arrival", evidence_need_ids=["activity"])
    data["inspect"]["evidence_need_ids"] = ["activity"]
    result = run(SVPProgram.model_validate(data))
    assert result["calls"][1]["source_scene_ids"] == [11, 12]
    assert [scene["index"] for scene in result["selected"]] == [12]


def test_empty_temporal_source_propagates_without_global_fallback(monkeypatch):
    # Inject an empty retrieval result at the boundary; later operations must
    # preserve that real result rather than widening the search to video.
    monkeypatch.setattr(runtime, "_relation_hits", lambda *_: [])
    result = run()
    assert result["calls"][2]["empty_source"] is True
    assert result["calls"][2]["source_scene_ids"] == []
    assert result["calls"][2]["hits"] == []
    assert result["selected"] == []
    assert result["uncovered_evidence_needs"] == ["arrival", "activity"]
    assert result["uncovered_sources"] == ["activity"]
    assert result["uncovered_temporal_outputs"] == ["after_arrival"]


def test_empty_overview_and_relation_helpers_do_not_expand_scope():
    assert runtime._overview([], embeddings(), clips(), 3) == []
    assert runtime._relation_hits(program().relate[0], [], clips()) == []


@pytest.mark.parametrize(
    "edit,error",
    [
        (lambda data: data["steps"][0].update(source="activity"), "unknown or later"),
        (lambda data: data["steps"][1].update(id="arrival"), "unique"),
        (lambda data: data["steps"][0].update(id="video"), "reserved"),
        (lambda data: data["steps"][1].update(source="video"), "earlier result"),
        (lambda data: data["steps"][0].update(id="Bad-ID"), "pattern"),
        (lambda data: data["steps"][0].update(id="x" * 33), "pattern"),
        (lambda data: data["steps"][0].update(source="arrival"), "unknown or later"),
        (lambda data: data["steps"][0].update(unknown="ignored?"), "Extra inputs"),
        (lambda data: data["steps"][0].update(limit="3"), "valid integer"),
        (lambda data: data["steps"][0].update(limit=0), "greater than or equal"),
        (lambda data: data["steps"][0].update(limit=13), "less than or equal"),
        (lambda data: data["steps"][1].update(horizon_seconds=0), "greater than"),
        (lambda data: data["steps"][1].update(horizon_seconds=181), "less than or equal"),
        (lambda data: data["steps"][0].update(evidence_need_ids=["other"]), "unknown evidence"),
        (lambda data: data["steps"][0].update(evidence_need_ids=["arrival", "arrival"]), "unique"),
        (lambda data: data["inspect"].update(sources=["missing"]), "unknown retrieval"),
        (lambda data: data["inspect"].update(sources=["activity", "activity"]), "unique"),
        (lambda data: data["steps"][0].update(evidence_need_ids=[]), "missing evidence needs"),
        (lambda data: data.update(max_intervals=4), "less than or equal"),
        (lambda data: data.update(max_inspected_seconds=61), "less than or equal"),
    ],
)
def test_invalid_graphs_and_untyped_payloads_fail_closed(edit, error):
    data = payload()
    edit(data)
    with pytest.raises(ValueError, match=error):
        SVPProgram.model_validate(data)


def test_dead_outputs_are_rejected_not_silently_deleted():
    data = payload()
    data["steps"].append({"op": "overview", "id": "unused", "source": "video"})
    with pytest.raises(ValueError, match="Unused retrieval outputs: unused"):
        SVPProgram.model_validate(data)


def test_inspect_labels_cannot_substitute_for_live_lineage_labels():
    data = payload()
    data["steps"][-1]["evidence_need_ids"] = []
    with pytest.raises(ValueError, match="missing evidence needs: activity"):
        SVPProgram.model_validate(data)


@pytest.mark.parametrize("relation", ["before", "during", "around"])
def test_outcome_dependency_requires_after_relation(relation):
    data = payload()
    data["steps"][1]["relation"] = relation
    with pytest.raises(ValueError, match=r"requires relate\(after\)"):
        SVPProgramValidator().validate(SVPProgram.model_validate(data), plan())


def test_unrelated_relation_branch_cannot_satisfy_dependency():
    data = payload()
    data["steps"][-1]["source"] = "video"
    data["inspect"]["sources"] = ["after_arrival", "activity"]
    with pytest.raises(ValueError, match="target ancestry"):
        SVPProgramValidator().validate(SVPProgram.model_validate(data), plan())


@pytest.mark.parametrize("field,value", [("max_intervals", 2), ("max_inspected_seconds", 30)])
def test_program_cannot_exceed_its_plan_budget(field, value):
    with pytest.raises(ValueError, match="exceeds the plan"):
        SVPProgramValidator().validate(program(), plan(**{field: value}))


def test_program_cannot_add_or_omit_plan_needs():
    changed = plan(
        evidence_needs=[
            {
                "id": "arrival",
                "description": "Initial action",
                "temporal_role": "context",
            }
        ]
    )
    with pytest.raises(ValueError, match="all and only"):
        SVPProgramValidator().validate(program(), changed)


def test_mutated_program_is_revalidated_before_execution():
    prog = program()
    prog.max_intervals = 100
    with pytest.raises(ValueError, match="less than or equal"):
        run(prog)


def overview_program(limit=12, **updates):
    return program(
        steps=[
            {
                "op": "overview",
                "id": "summary",
                "source": "video",
                "limit": limit,
                "evidence_need_ids": ["context"],
            }
        ],
        inspect={"sources": ["summary"], "evidence_need_ids": ["context"]},
        **updates,
    )


def test_whole_video_can_use_one_context_need_not_forced_thirds():
    global_plan = plan(
        scope="whole_video",
        evidence_needs=[
            {
                "id": "context",
                "description": "Representative activity",
                "temporal_role": "context",
            }
        ],
    )
    prog = overview_program()
    SVPProgramValidator().validate(prog, global_plan)
    result = run(prog, queries=[], scope="whole_video")
    assert len(result["selected"]) == 3
    assert result["uncovered_evidence_needs"] == []
    assert "temporal_coverage" not in result
    assert "not_required_thirds" in result["calls"][0]["coverage_policy"]
    assert prog.locate == []


def test_overview_exposes_real_novelty_and_temporal_spread():
    result = run(overview_program(), queries=[])
    hits = result["calls"][0]["hits"]
    assert hits[0]["selection_method"] == "semantic_medoid"
    assert all(hit["selection_method"] == "novelty_and_temporal_spread" for hit in hits[1:])
    for hit in hits[1:]:
        assert hit["representative_score"] == pytest.approx(
            0.6 * hit["semantic_novelty"] + 0.4 * hit["temporal_spread"]
        )
    assert hits[1]["semantic_novelty"] > 0.5  # The distinct opposing vector matters.


def test_overview_is_scoped_to_previous_results():
    data = payload()
    data["steps"] = [
        data["steps"][0],
        {
            "op": "overview",
            "id": "summary",
            "source": "arrival",
            "limit": 12,
            "evidence_need_ids": ["activity"],
        },
    ]
    data["inspect"]["sources"] = ["summary"]
    result = run(SVPProgram.model_validate(data), queries=[[1, 0]])
    assert result["calls"][1]["source_scene_ids"] == [11]
    assert [scene["index"] for scene in result["selected"]] == [11]


def test_global_overview_limits_intervals_and_real_seconds():
    result = run(overview_program(), scenes=clips(span=25), queries=[], scope="whole_video")
    assert len(result["selected"]) == 2
    assert result["calls"][-1]["inspected_seconds"] == 50
    tiny = run(overview_program(max_inspected_seconds=5), queries=[])
    assert tiny["selected"] == []
    assert tiny["uncovered_sources"] == ["summary"]


def test_no_feasible_window_is_reported_not_replaced_with_full_video():
    result = run(scenes=clips(span=70))
    assert result["selected"] == []
    assert result["uncovered_evidence_needs"] == ["arrival", "activity"]
    assert result["calls"][-1]["inspected_seconds"] == 0


def test_cosine_normalization_is_real_and_zero_queries_allowed_only_without_locates():
    vectors = embeddings() * 7
    result = run(vectors=vectors, queries=[[3, 0], [0, 5]])
    assert result["calls"][0]["hits"][0]["semantic_cosine"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Query embeddings"):
        run(queries=[])
    assert run(overview_program(), queries=np.empty((0, 2)))["selected"]


@pytest.mark.parametrize("bad", [np.ones((5, 2)), np.zeros((6, 2)), np.full((6, 2), np.nan)])
def test_invalid_scene_embeddings_fail(bad):
    with pytest.raises(ValueError, match="Scene embeddings"):
        run(vectors=bad)


def test_mismatched_query_dimensions_fail():
    with pytest.raises(ValueError, match="same dimension"):
        run(queries=np.ones((2, 3)))


@pytest.mark.parametrize(
    "edit,error",
    [
        (lambda scenes: scenes[1].update(index=10), "unique"),
        (lambda scenes: scenes[1].update(index=True), "integers"),
        (lambda scenes: scenes[1].update(start=-1), "finite, positive"),
        (lambda scenes: scenes[1].update(end=float("inf")), "finite, positive"),
        (lambda scenes: scenes[1].update(start=30), "finite, positive"),
    ],
)
def test_invalid_scene_metadata_fails(edit, error):
    scenes = clips()
    edit(scenes)
    with pytest.raises(ValueError, match=error):
        run(scenes=scenes)


def test_empty_video_index_fails_and_does_not_invent_candidates():
    with pytest.raises(ValueError, match="empty video index"):
        run(scenes=[])


def test_execution_does_not_mutate_sources_and_is_deterministic():
    prog, scenes, vectors = program(), clips(), embeddings()
    original = deepcopy((prog.model_dump(), scenes, vectors.tolist()))
    first = run(prog, scenes, vectors)
    assert first == run(prog, scenes, vectors)
    assert (prog.model_dump(), scenes, vectors.tolist()) == original


def test_final_budget_reports_each_uncovered_terminal_source():
    prog = program(
        steps=[
            {
                "op": "locate",
                "id": "first",
                "source": "video",
                "query": "First activity",
                "limit": 1,
                "evidence_need_ids": ["first"],
            },
            {
                "op": "locate",
                "id": "second",
                "source": "video",
                "query": "Second activity",
                "limit": 1,
                "evidence_need_ids": ["second"],
            },
        ],
        inspect={"sources": ["first", "second"], "evidence_need_ids": ["first", "second"]},
        max_intervals=1,
    )
    result = run(prog)
    assert len(result["selected"]) == 1
    assert len(result["uncovered_sources"]) == 1
    assert len(result["uncovered_evidence_needs"]) == 1


def test_v3_schema_uses_supported_strict_hosted_anyof_not_oneof():
    from videomasys.zerogpu.openai import strict_json_schema

    for model in (SVPPlan, SVPProgram):
        schema = strict_json_schema(model.model_json_schema())
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    branches = schema["properties"]["steps"]["items"]
    assert "anyOf" in branches
    assert "oneOf" not in branches
    assert "discriminator" not in branches
    for name in ("LocateStep", "RelateStep", "OverviewStep"):
        assert "op" in schema["$defs"][name]["required"]


@pytest.mark.parametrize("op", [None, "exec", "find"])
def test_operator_must_be_present_and_known(op):
    data = payload()
    if op is None:
        del data["steps"][0]["op"]
    else:
        data["steps"][0]["op"] = op
    with pytest.raises(ValueError):
        SVPProgram.model_validate(data)


@pytest.mark.parametrize("value", [1e308, 1e-308])
def test_finite_but_overflowing_or_underflowing_norm_is_rejected(value):
    with pytest.raises(ValueError, match="Scene embeddings"):
        run(vectors=np.full((6, 2), value))


def test_negative_scene_index_is_rejected():
    scenes = clips()
    scenes[0]["index"] = -1
    with pytest.raises(ValueError, match="nonnegative integers"):
        run(scenes=scenes)


def test_plan_preserves_three_need_limit_without_forced_global_thirds():
    needs = [
        {"id": f"need_{index}", "description": "Independent evidence", "temporal_role": "context"}
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="at most 3"):
        plan(evidence_needs=needs)


def test_step_budget_is_eight_and_does_not_silently_truncate():
    data = payload()
    data["steps"] = [
        {
            "op": "overview",
            "id": f"step_{index}",
            "source": "video" if index == 0 else f"step_{index - 1}",
            "evidence_need_ids": ["arrival", "activity"] if index == 8 else [],
        }
        for index in range(9)
    ]
    data["inspect"]["sources"] = ["step_8"]
    with pytest.raises(ValueError, match="at most 8"):
        SVPProgram.model_validate(data)


def test_temporal_chain_uses_preceding_refined_result_not_original_anchor():
    data = payload()
    data["steps"].append(
        {
            "op": "relate",
            "id": "context",
            "source": "activity",
            "relation": "before",
            "horizon_seconds": 10,
            "limit": 1,
            "evidence_need_ids": [],
        }
    )
    data["inspect"]["sources"] = ["context"]
    result = run(SVPProgram.model_validate(data))
    assert result["calls"][3]["anchor_scene"] == 14
    assert result["calls"][3]["source_scene_ids"] == [14]
    assert result["calls"][3]["hits"][0]["scene"] == 13
    assert result["calls"][3]["ancestry"] == ["arrival", "after_arrival", "activity", "context"]


def test_inspection_searches_feasible_combinations_not_greedy_source_order():
    prog = program(
        max_intervals=1,
        steps=[
            {
                "op": "locate",
                "id": "one",
                "source": "video",
                "query": "First need",
                "limit": 2,
                "evidence_need_ids": ["one"],
            },
            {
                "op": "locate",
                "id": "two",
                "source": "video",
                "query": "Second need",
                "limit": 1,
                "evidence_need_ids": ["two"],
            },
        ],
        inspect={"sources": ["one", "two"], "evidence_need_ids": ["one", "two"]},
    )
    result = run(
        prog, scenes=clips(3), vectors=[[1, 0], [0.8, 0.6], [-1, -1]], queries=[[1, 0], [0, 1]]
    )
    assert [scene["index"] for scene in result["selected"]] == [11]
    assert result["uncovered_sources"] == []
    assert result["uncovered_evidence_needs"] == []


def test_duplicate_terminal_queries_cannot_vote_a_scene_into_selection():
    data = payload(
        max_intervals=1,
        steps=[
            {
                "op": "locate",
                "id": "one",
                "source": "video",
                "query": "First search",
                "limit": 2,
                "evidence_need_ids": ["activity"],
            },
            {
                "op": "locate",
                "id": "two",
                "source": "video",
                "query": "Second search",
                "limit": 2,
                "evidence_need_ids": ["activity"],
            },
        ],
        inspect={"sources": ["one", "two"], "evidence_need_ids": ["activity"]},
    )
    vectors = [[1, 0], [0, 1], [0.7, 0.7]]
    baseline = run(SVPProgram.model_validate(data), scenes=clips(3), vectors=vectors)
    duplicate = {**data["steps"][1], "id": "duplicate"}
    data["steps"].append(duplicate)
    data["inspect"]["sources"].append("duplicate")
    repeated = run(
        SVPProgram.model_validate(data),
        scenes=clips(3),
        vectors=vectors,
        queries=[[1, 0], [0, 1], [0, 1]],
    )
    assert [scene["index"] for scene in baseline["selected"]] == [10]
    assert repeated["selected"] == baseline["selected"]


def test_need_first_selection_spends_spare_slots_on_temporal_targets_not_anchor_runners_up():
    data = payload()
    data["steps"][0]["limit"] = 3
    data["steps"][1].update(horizon_seconds=60, limit=3)
    data["steps"][2]["limit"] = 3
    data["inspect"]["sources"] = ["arrival", "activity"]
    result = run(
        SVPProgram.model_validate(data),
        vectors=[[1, 0, 0], [0.9, 0, 0.1], [0.8, 0, 0.2], [0, 1, 0], [0, 0.9, 0.1], [0, 0.8, 0.2]],
        queries=[[1, 0, 0], [0, 1, 0]],
    )
    assert [scene["index"] for scene in result["selected"]] == [10, 13, 14]
    assert result["coverage"]["need_sources"] == {"arrival": ["arrival"], "activity": ["activity"]}
    assert result["uncovered_sources"] == []
    assert result["uncovered_evidence_needs"] == []
    assert result["calls"][-1]["inspected_seconds"] == 30


def test_navigation_label_cannot_replace_explicitly_inspected_anchor_under_tight_budget():
    data = payload(max_intervals=1)
    data["inspect"]["sources"] = ["arrival", "activity"]
    result = run(SVPProgram.model_validate(data))
    assert len(result["selected"]) == 1
    assert len(result["uncovered_evidence_needs"]) == 1
    assert result["coverage"]["need_sources"] == {"arrival": ["arrival"], "activity": ["activity"]}


def test_whitespace_search_is_rejected():
    data = payload()
    data["steps"][0]["query"] = "   "
    with pytest.raises(ValueError, match="non-whitespace"):
        SVPProgram.model_validate(data)


def global_plan(**updates):
    return plan(
        scope="whole_video",
        evidence_needs=[
            {
                "id": "context",
                "description": "Representative activity",
                "temporal_role": "context",
            }
        ],
        **updates,
    )


def test_task_global_gate_does_not_change_generic_compositional_validation():
    prog, global_event_plan = program(), plan(scope="whole_video")
    assert SVPProgramValidator().validate(prog, global_event_plan) is prog
    with pytest.raises(ValueError, match="directly in inspect.sources"):
        validate_svp_task_program(prog, global_event_plan, clips())


def test_task_global_gate_rejects_overview_scoped_to_one_anchor():
    data = payload()
    data["steps"].append(
        {
            "op": "overview",
            "id": "scoped_summary",
            "source": "activity",
            "limit": 3,
        }
    )
    data["inspect"]["sources"] = ["scoped_summary"]
    prog = SVPProgram.model_validate(data)
    with pytest.raises(ValueError, match="scoped to one retrieved event"):
        validate_svp_task_program(prog, plan(scope="whole_video"), clips())


def test_task_global_gate_rejects_root_overview_used_only_as_navigation():
    prog = program(
        steps=[
            {
                "op": "overview",
                "id": "overview",
                "source": "video",
                "limit": 3,
                "evidence_need_ids": ["context"],
            },
            {
                "op": "locate",
                "id": "specific",
                "source": "overview",
                "limit": 1,
                "query": "Specific activity",
                "evidence_need_ids": [],
            },
        ],
        inspect={"sources": ["specific"], "evidence_need_ids": ["context"]},
    )
    with pytest.raises(ValueError, match="directly in inspect.sources"):
        validate_svp_task_program(prog, global_plan(), clips())


def test_task_overview_candidate_count_must_cover_available_interval_slots():
    with pytest.raises(ValueError, match="limit >= 3"):
        validate_svp_task_program(overview_program(limit=2), global_plan(), clips())
    prog = overview_program(limit=2, max_intervals=2)
    assert validate_svp_task_program(prog, global_plan(), clips()) is prog


def test_task_global_gate_handles_short_single_window_video_without_three_probes():
    prog = overview_program(limit=1)
    validate_svp_task_program(prog, global_plan(), clips(1))
    result = run(prog, scenes=clips(1), vectors=[[1, 1]], queries=[], scope="whole_video")
    assert result["coverage"]["global_representatives_required"] == 1
    assert result["coverage"]["global_representatives_selected"] == 1
    assert result["coverage"]["global_coverage_complete"] is True


def test_global_selector_retains_representatives_before_unrelated_source_utility():
    base = run(overview_program(limit=3), queries=[], scope="whole_video")
    representatives = {scene["index"] for scene in base["selected"]}
    outside = next(index for index in range(6) if index + 10 not in representatives)
    prog = program(
        steps=[
            {
                "op": "overview",
                "id": "summary",
                "source": "video",
                "limit": 3,
                "evidence_need_ids": ["context"],
            },
            {
                "op": "locate",
                "id": "detail",
                "source": "video",
                "limit": 1,
                "query": "Separate detail",
                "evidence_need_ids": ["detail"],
            },
        ],
        inspect={"sources": ["summary", "detail"], "evidence_need_ids": ["context", "detail"]},
    )
    result = run(prog, queries=[embeddings()[outside]], scope="whole_video")
    assert {scene["index"] for scene in result["selected"]} == representatives
    assert result["coverage"]["global_representatives_selected"] == 3
    assert result["coverage"]["global_coverage_complete"] is True
    # Do not falsely claim the separate targeted source was covered as well.
    assert result["uncovered_sources"] == ["detail"]
    assert result["uncovered_evidence_needs"] == ["detail"]


def test_global_selector_preserves_ranked_diversity_not_arbitrary_membership_in_large_pool():
    base = run(overview_program(limit=6), queries=[], scope="whole_video")
    ranked = [hit["scene"] for hit in base["calls"][0]["hits"]]
    assert {scene["index"] for scene in base["selected"]} == set(ranked[:3])
    outside = ranked[-1] - 10
    data = payload(
        steps=[overview_program(limit=6).steps[0].model_dump()],
        inspect={"sources": ["summary"], "evidence_need_ids": ["context"]},
    )
    for name in ("detail", "repeated_detail"):
        data["steps"].append(
            {
                "op": "locate",
                "id": name,
                "source": "video",
                "query": "A repeated detail",
                "limit": 1,
                "evidence_need_ids": ["context"],
            }
        )
        data["inspect"]["sources"].append(name)
    result = run(
        SVPProgram.model_validate(data),
        queries=[embeddings()[outside]] * 2,
        scope="whole_video",
    )
    assert result["selected"] == base["selected"]
    assert result["coverage"]["global_coverage_complete"] is True
    assert result["uncovered_sources"] == ["detail", "repeated_detail"]


def test_global_selector_covers_required_detail_in_pool_before_overview_rank_tiebreak():
    base = run(overview_program(limit=6), queries=[], scope="whole_video")
    ranked = [hit["scene"] for hit in base["calls"][0]["hits"]]
    detail_position = ranked[-1] - 10
    data = payload(
        steps=[
            overview_program(limit=6).steps[0].model_dump(),
            {
                "op": "locate",
                "id": "detail",
                "source": "video",
                "query": "Required distinguishing detail",
                "limit": 1,
                "evidence_need_ids": ["detail"],
            },
        ],
        inspect={"sources": ["summary", "detail"], "evidence_need_ids": ["context", "detail"]},
    )
    result = run(
        SVPProgram.model_validate(data),
        queries=[embeddings()[detail_position]],
        scope="whole_video",
    )
    assert {scene["index"] for scene in result["selected"]} == {ranked[0], ranked[1], ranked[-1]}
    assert result["coverage"]["global_representatives_selected"] == 3
    assert result["coverage"]["global_coverage_complete"] is True
    assert result["uncovered_sources"] == []
    assert result["uncovered_evidence_needs"] == []


def test_insufficient_seconds_report_incomplete_global_coverage_without_overrun():
    prog = overview_program(limit=3)
    result = run(prog, scenes=clips(span=25), queries=[], scope="whole_video")
    assert result["coverage"]["global_representatives_required"] == 3
    assert result["coverage"]["global_representatives_selected"] == 2
    assert result["coverage"]["global_coverage_complete"] is False
    assert result["calls"][-1]["inspected_seconds"] == 50
    assert len(result["selected"]) == 2


def test_direct_runtime_does_not_invent_global_coverage_when_task_gate_is_skipped():
    result = run(scope="whole_video")
    assert result["coverage"]["global_representative_sources"] == []
    assert result["coverage"]["global_representatives_required"] == 3
    assert result["coverage"]["global_representatives_selected"] == 0
    assert result["coverage"]["global_coverage_complete"] is False


def test_task_gate_does_not_ban_short_horizons_or_relation_chains_for_local_tasks():
    data = payload()
    data["steps"][1]["horizon_seconds"] = 1
    data["steps"].append(
        {
            "op": "relate",
            "id": "nested",
            "source": "after_arrival",
            "relation": "around",
            "horizon_seconds": 1,
            "evidence_need_ids": [],
        }
    )
    data["inspect"]["sources"].append("nested")
    prog = SVPProgram.model_validate(data)
    assert validate_svp_task_program(prog, plan(), clips()) is prog


def test_task_gate_rejects_empty_index_before_any_retrieval():
    with pytest.raises(ValueError, match="empty video index"):
        validate_svp_task_program(program(), plan(), [])


@pytest.mark.parametrize("duplicate", [False, True])
def test_selection_keeps_all_temporal_branches_when_feasible_without_source_votes(duplicate):
    steps = [
        payload()["steps"][0],
        *[{"op": "relate", "id": direction, "source": "arrival", "relation": direction,
           "horizon_seconds": 60, "limit": 2, "evidence_need_ids": ["activity"]}
          for direction in ("before", "after")],
        {"op": "locate", "id": "broad", "source": "video", "query": "Extra context",
         "limit": 1, "evidence_need_ids": []},
    ]
    def hit(position):
        return {"position": position, "scene": position + 10}

    # Isolated selector fixture: the broad query boosts a before-scene's best
    # rank, but must not displace the other temporal branch's only candidate.
    outputs = {"arrival": [hit(2)], "before": [hit(0), hit(1)],
               "after": [hit(3)], "broad": [hit(1)]}
    if duplicate:
        steps.append({**steps[1], "id": "before_again"})
        outputs["before_again"] = deepcopy(outputs["before"])
    prog = program(steps=steps, inspect={"sources": list(outputs),
                                        "evidence_need_ids": ["arrival", "activity"]})
    calls, lineage = runtime._graph(prog)
    selected, _, coverage = runtime._select(prog, clips(4), outputs, calls, lineage)
    assert selected == [0, 2, 3]
    assert coverage["uncovered_evidence_needs"] == []
    assert coverage["uncovered_temporal_outputs"] == []
