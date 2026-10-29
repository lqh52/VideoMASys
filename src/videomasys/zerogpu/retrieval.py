"""Typed, auditable retrieval over real scene-caption embeddings.

Temporal operators constrain *windows*, not events inside those windows. In
particular, including the anchor window does not establish event ordering.
"""

from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np

from videomasys.domain.schemas import ToolProgram


def _unit_rows(values, rows: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) != rows or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with {rows} rows")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError(f"{name} contains an empty embedding")
    return array / norms


def _temporal_hit(scene: dict, anchor: dict, relation: str, horizon: float) -> dict | None:
    overlap = scene["start"] < anchor["end"] and scene["end"] > anchor["start"]
    if overlap:
        gap = 0.0
        kind = "anchor_overlap"
    elif relation == "during":
        return None
    elif relation == "after":
        gap = scene["start"] - anchor["end"]
        kind = "strict_after"
    elif relation == "before":
        gap = anchor["start"] - scene["end"]
        kind = "strict_before"
    else:
        gap = max(scene["start"] - anchor["end"], anchor["start"] - scene["end"])
        kind = "strict_after" if scene["start"] >= anchor["end"] else "strict_before"
    if not 0 <= gap <= horizon:
        return None
    return {
        "anchor_overlap": overlap,
        "window_relation": kind,
        "gap_seconds": float(gap),
        "temporal_proximity": 1.0 - float(gap) / horizon,
    }


def execute_program(
    program: ToolProgram,
    scenes: list[dict],
    vectors,
    query_vectors,
    *,
    scope: Literal["local_event", "whole_video"] = "local_event",
) -> dict:
    """Execute cosine search and target-aware temporal retrieval within hard budgets.

    For each evidence need, prefer its inspected relation outputs over unrestricted
    locate outputs. Admit the best fitting candidate first, then cover remaining
    temporal outputs, then fill spare slots with diverse (nonduplicate) candidates.
    Coverage means retrieval candidate coverage, not visual proof of the need.
    Whole-video mode instead selects distinct midpoint-based temporal representatives;
    callers must stop before answering when temporal_coverage.complete is false.
    """
    if scope not in {"local_event", "whole_video"}:
        raise ValueError(f"Unknown retrieval scope: {scope}")
    if scope == "whole_video":
        _validate_whole_video_shape(program)
    if not scenes:
        raise ValueError("Cannot retrieve from an empty video index")
    if len({scene["index"] for scene in scenes}) != len(scenes):
        raise ValueError("Scene indices must be unique")
    for scene in scenes:
        if (
            not np.isfinite([scene["start"], scene["end"]]).all()
            or scene["start"] < 0
            or scene["end"] <= scene["start"]
        ):
            raise ValueError("Scene spans must be finite, positive video intervals")
    scene_vectors = _unit_rows(vectors, len(scenes), "Scene embeddings")
    queries = _unit_rows(query_vectors, len(program.locate), "Query embeddings")
    if scene_vectors.shape[1] != queries.shape[1]:
        raise ValueError("Scene and query embeddings must have the same dimension")
    cosine = np.clip(scene_vectors @ queries.T, -1.0, 1.0)
    outputs: dict[str, list[dict]] = {}
    trace = []
    calls = {call.id: call for call in [*program.locate, *program.relate]}
    video_end = max(scene["end"] for scene in scenes)

    for query_index, call in enumerate(program.locate):
        bounds = {
            "any": (0.0, video_end),
            "beginning": (0.0, video_end / 3),
            "middle": (video_end / 3, 2 * video_end / 3),
            "end": (2 * video_end / 3, video_end),
        }[call.temporal_position]
        if scope == "whole_video":
            # A straddling window belongs to exactly one third, never several.
            eligible = [
                index
                for index, scene in enumerate(scenes)
                if bounds[0] <= (scene["start"] + scene["end"]) / 2 < bounds[1]
            ]
        else:
            eligible = [
                index
                for index, scene in enumerate(scenes)
                if scene["start"] < bounds[1] and scene["end"] > bounds[0]
            ]
        ordered = sorted(eligible, key=lambda index: (-cosine[index, query_index], index))
        hits = [
            {
                "position": index,
                "scene": scenes[index]["index"],
                "score": float(cosine[index, query_index]),
                "semantic_cosine": float(cosine[index, query_index]),
            }
            for index in ordered[: call.limit]
        ]
        outputs[call.id] = hits
        trace.append(
            {
                "call": call.model_dump(),
                "score_type": "semantic_cosine",
                "temporal_filter": {
                    "position": call.temporal_position,
                    "start": bounds[0],
                    "end": bounds[1],
                    "semantics": "scene_midpoint" if scope == "whole_video" else "window_overlap",
                },
                "hits": [_public_hit(hit) for hit in hits],
            }
        )

    if scope == "whole_video":
        return _inspect_whole_video(program, scenes, outputs, trace)

    for call in program.relate:
        if call.source not in outputs:
            raise ValueError(f"Unknown temporal source: {call.source}")
        source_hits = outputs[call.source]
        target_queries = [
            index
            for index, locate in enumerate(program.locate)
            if set(locate.evidence_need_ids) & set(call.evidence_need_ids)
        ]
        hits = []
        anchor = scenes[source_hits[0]["position"]] if source_hits else None
        if anchor is not None:
            for index, scene in enumerate(scenes):
                temporal = _temporal_hit(scene, anchor, call.relation, call.horizon_seconds)
                if temporal is None:
                    continue
                hit = {"position": index, "scene": scene["index"], **temporal}
                if target_queries:
                    # Every number remains in the cosine domain; proximity is a
                    # tie-break, never added to or compared against cosine.
                    semantic = float(np.max(cosine[index, target_queries]))
                    hit.update(score=semantic, semantic_cosine=semantic)
                hits.append(hit)
        hits.sort(
            key=lambda hit: (
                -hit.get("semantic_cosine", 0.0),
                -hit["temporal_proximity"],
                hit["position"],
            )
        )
        outputs[call.id] = hits[: call.limit]
        trace.append(
            {
                "call": call.model_dump(),
                "anchor_scene": anchor["index"] if anchor is not None else None,
                "target_query_sources": [program.locate[index].id for index in target_queries],
                "ranking": (
                    ["semantic_cosine", "temporal_proximity"]
                    if target_queries
                    else ["temporal_proximity"]
                ),
                "score_type": "semantic_cosine" if target_queries else "temporal_proximity",
                "ordering_scope": "windows_only; anchor_overlap does not establish event ordering",
                "hits": [_public_hit(hit) for hit in outputs[call.id]],
            }
        )

    inspected = program.inspect.sources
    if any(source not in outputs for source in inspected):
        raise ValueError("Inspect references an unknown retrieval output")
    relation_ids = {call.id for call in program.relate}
    required_relations = [source for source in inspected if source in relation_ids]
    need_sources = {}
    for need in program.inspect.evidence_need_ids:
        sources = [source for source in inspected if need in calls[source].evidence_need_ids]
        temporal = [source for source in sources if source in relation_ids]
        need_sources[need] = temporal or sources

    interval_limit = min(3, program.max_intervals)
    second_limit = min(60.0, program.max_inspected_seconds)
    selected: list[int] = []
    reasons: dict[int, list[dict]] = {}
    seconds = 0.0

    def admit(source: str, hit: dict, rank: int, purpose: str, needs: list[str]) -> bool:
        nonlocal seconds
        index = hit["position"]
        if index not in selected:
            span = scenes[index]["end"] - scenes[index]["start"]
            if len(selected) >= interval_limit or seconds + span > second_limit + 1e-9:
                return False
            selected.append(index)
            seconds += span
        reason = {
            "source": source,
            "rank": rank + 1,
            "purpose": purpose,
            "evidence_need_ids": needs,
        }
        reason.update(
            {
                key: hit[key]
                for key in (
                    "semantic_cosine",
                    "temporal_proximity",
                    "anchor_overlap",
                    "window_relation",
                )
                if key in hit
            }
        )
        if reason not in reasons.setdefault(index, []):
            reasons[index].append(reason)
        return True

    for need, sources in need_sources.items():
        for source, rank, hit in _round_robin(sources, outputs):
            if admit(source, hit, rank, "evidence_need", [need]):
                break

    # A second dependency on the same target must not become ceremonial merely
    # because another source already admitted a candidate carrying that need ID.
    for source in required_relations:
        if any(hit["position"] in selected for hit in outputs[source]):
            continue
        for rank, hit in enumerate(outputs[source]):
            if admit(source, hit, rank, "temporal_dependency", calls[source].evidence_need_ids):
                break

    # Spend any remaining slots on temporal candidates first, avoiding unrelated
    # anchor-search runners-up while relevant constrained evidence is available.
    for sources in [required_relations, [s for s in inspected if s not in relation_ids]]:
        for source, rank, hit in _round_robin(sources, outputs):
            if hit["position"] not in selected:
                admit(source, hit, rank, "additional_candidate", calls[source].evidence_need_ids)

    def has_candidate(sources):
        return any(hit["position"] in selected for source in sources for hit in outputs[source])

    uncovered = [need for need, sources in need_sources.items() if not has_candidate(sources)]
    uncovered_relations = [source for source in required_relations if not has_candidate([source])]
    selected.sort(key=lambda index: (scenes[index]["start"], index))
    selection_reasons = [
        {"scene": scenes[index]["index"], "reasons": reasons[index]} for index in selected
    ]
    trace.append(
        {
            "call": "inspect",
            "selected": [scenes[index]["index"] for index in selected],
            "inspected_seconds": seconds,
            "budget_seconds": second_limit,
            "budget_intervals": interval_limit,
            "coverage_scope": "retrieval_candidates_not_visual_confirmation",
            "selection_reasons": selection_reasons,
            "uncovered_evidence_needs": uncovered,
            "uncovered_temporal_outputs": uncovered_relations,
        }
    )
    return {
        "selected": [scenes[index] for index in selected],
        "calls": trace,
        "selection_reasons": selection_reasons,
        "uncovered_evidence_needs": uncovered,
        "uncovered_temporal_outputs": uncovered_relations,
    }


def _validate_whole_video_shape(program: ToolProgram) -> None:
    """Defend direct executor callers too; plan-to-need mapping is checked upstream."""
    positions = ("beginning", "middle", "end")
    needs = [call.evidence_need_ids for call in program.locate]
    if (
        program.relate
        or len(program.locate) != 3
        or [call.id for call in program.locate] != ["l1", "l2", "l3"]
        or [call.temporal_position for call in program.locate] != list(positions)
        or any(len(ids) != 1 for ids in needs)
        or len({need for ids in needs for need in ids}) != 3
        or len(program.inspect.sources) != 3
        or set(program.inspect.sources) != {"l1", "l2", "l3"}
        or len(program.inspect.evidence_need_ids) != 3
        or set(program.inspect.evidence_need_ids) != {need for ids in needs for need in ids}
    ):
        raise ValueError(
            "Whole-video retrieval requires exactly l1,l2,l3 for beginning,middle,end, "
            "one distinct evidence need each, relate=[], and inspection of exactly "
            "those outputs/IDs."
        )


def _inspect_whole_video(program, scenes, outputs, trace) -> dict:
    """Choose a feasible joint set, not a greedy prefix that can exhaust the budget."""
    interval_limit = min(3, program.max_intervals)
    second_limit = min(60.0, program.max_inspected_seconds)
    candidates = [[*enumerate(outputs[call.id]), None] for call in program.locate]
    best, best_key, seconds = None, None, 0.0
    for combination in product(*candidates):
        hits = [candidate[1] for candidate in combination if candidate is not None]
        indices = [hit["position"] for hit in hits]
        duration = sum(scenes[index]["end"] - scenes[index]["start"] for index in indices)
        if (
            len(indices) > interval_limit
            or len(indices) != len(set(indices))
            or duration > second_limit + 1e-9
        ):
            continue
        key = (
            len(hits),
            sum(hit["semantic_cosine"] for hit in hits),
            -duration,
            tuple(
                -candidate[0] if candidate is not None else -len(scenes)
                for candidate in combination
            ),
        )
        if best_key is None or key > best_key:
            best, best_key, seconds = combination, key, duration
    # The all-None combination always fits and preserves honest failure diagnostics.
    assert best is not None
    selected, reasons, representatives, missing, uncovered = [], [], [], [], []
    for call, candidate in zip(program.locate, best, strict=True):
        if candidate is None:
            missing.append(call.temporal_position)
            uncovered.extend(call.evidence_need_ids)
            continue
        rank, hit = candidate
        scene = scenes[hit["position"]]
        selected.append(scene)
        reasons.append(
            {
                "scene": scene["index"],
                "reasons": [
                    {
                        "source": call.id,
                        "rank": rank + 1,
                        "purpose": "temporal_representative",
                        "evidence_need_ids": call.evidence_need_ids,
                        "temporal_position": call.temporal_position,
                        "semantic_cosine": hit["semantic_cosine"],
                    }
                ],
            }
        )
        representatives.append(
            {
                "position": call.temporal_position,
                "scene": scene["index"],
                "start": scene["start"],
                "end": scene["end"],
                "midpoint": (scene["start"] + scene["end"]) / 2,
            }
        )
    selected.sort(key=lambda scene: (scene["start"], scene["index"]))
    coverage = {
        "complete": not missing,
        "semantics": "distinct_scene_midpoints_in_video_thirds",
        "candidate_scope": "returned_locate_hits",
        "required_positions": ["beginning", "middle", "end"],
        "missing_positions": missing,
        "candidate_counts": {
            call.temporal_position: len(outputs[call.id]) for call in program.locate
        },
        "representatives": representatives,
    }
    trace.append(
        {
            "call": "inspect",
            "selected": [scene["index"] for scene in selected],
            "inspected_seconds": seconds,
            "budget_seconds": second_limit,
            "budget_intervals": interval_limit,
            "coverage_scope": "retrieval_candidates_not_visual_confirmation",
            "selection_reasons": reasons,
            "uncovered_evidence_needs": uncovered,
            "uncovered_temporal_outputs": [],
            "temporal_coverage": coverage,
        }
    )
    return {
        "selected": selected,
        "calls": trace,
        "selection_reasons": reasons,
        "uncovered_evidence_needs": uncovered,
        "uncovered_temporal_outputs": [],
        "temporal_coverage": coverage,
    }


def _public_hit(hit: dict) -> dict:
    return {key: value for key, value in hit.items() if key != "position"}


def _round_robin(sources: list[str], outputs: dict):
    for rank in range(max((len(outputs[source]) for source in sources), default=0)):
        for source in sources:
            if rank < len(outputs[source]):
                yield source, rank, outputs[source][rank]
