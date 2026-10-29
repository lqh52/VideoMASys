"""Local-only, compositional scene retrieval inspired by SVP.

This is a separate v3 contract. It does not execute Python, change the v2
retriever, or pretend that scene windows are an action/object memory hierarchy.
Lineage coverage is navigation provenance, never confirmation of a visual fact.
"""

from __future__ import annotations

from itertools import combinations
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from videomasys.domain.schemas import EvidenceNeed, InspectCall, ReasoningPlan

Variable = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]


class SVPEvidenceNeed(EvidenceNeed):
    model_config = ConfigDict(extra="forbid", strict=True)


class SVPPlan(ReasoningPlan):
    """Evidence requirements without a forced early/middle/late decomposition."""

    model_config = ConfigDict(extra="forbid", strict=True)
    scope: Literal["local_event", "whole_video"]
    evidence_needs: list[SVPEvidenceNeed] = Field(min_length=1, max_length=3)
    max_intervals: int = Field(default=3, ge=1, le=3)
    max_inspected_seconds: float = Field(default=60.0, ge=5.0, le=60.0)

    @model_validator(mode="after")
    def unique_dependencies(self):
        for need in self.evidence_needs:
            _unique(need.depends_on, f"{need.id} dependencies")
        return self


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: Variable
    source: Variable
    limit: int = Field(default=3, ge=1, le=12)
    evidence_need_ids: list[Variable] = Field(default_factory=list, max_length=3)


class LocateStep(_Step):
    op: Literal["locate"]
    query: str = Field(min_length=3, max_length=500)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value):
        if not value.strip():
            raise ValueError("locate query must contain non-whitespace text")
        return value


class RelateStep(_Step):
    op: Literal["relate"]
    relation: Literal["before", "after", "during", "around"]
    horizon_seconds: float = Field(default=60.0, gt=0.0, le=180.0)


class OverviewStep(_Step):
    op: Literal["overview"]


class SVPInspectCall(InspectCall):
    model_config = ConfigDict(extra="forbid", strict=True)
    sources: list[Variable] = Field(min_length=1, max_length=8)
    evidence_need_ids: list[Variable] = Field(min_length=1, max_length=3)


class SVPProgram(BaseModel):
    """An ordered typed DAG; source references can only name earlier outputs."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["3.0"] = "3.0"
    compiler: Literal["model", "model_repaired"] = "model"
    rationale: str = Field(min_length=3, max_length=800)
    max_intervals: int = Field(default=3, ge=1, le=3)
    max_inspected_seconds: float = Field(default=60.0, ge=5.0, le=60.0)
    # Required disjoint op literals are unambiguous and produce the anyOf form
    # supported by the existing hosted strict schema normalizer (not oneOf).
    steps: list[LocateStep | RelateStep | OverviewStep] = Field(min_length=1, max_length=8)
    inspect: SVPInspectCall

    @property
    def locate(self) -> list[LocateStep]:
        return [step for step in self.steps if isinstance(step, LocateStep)]

    @property
    def relate(self) -> list[RelateStep]:
        return [step for step in self.steps if isinstance(step, RelateStep)]

    @model_validator(mode="after")
    def validate_graph(self):
        _graph(self)
        return self


def _unique(values: list[str], owner: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{owner} must be unique")


def _graph(program: SVPProgram) -> tuple[dict, dict[str, list[str]]]:
    known_needs = set(program.inspect.evidence_need_ids)
    _unique(program.inspect.sources, "inspect sources")
    _unique(program.inspect.evidence_need_ids, "inspect evidence need ids")
    calls, lineage = {}, {}
    for step in program.steps:
        if step.id == "video" or step.id in calls:
            raise ValueError("Step IDs must be unique; video is a reserved source")
        if step.source == "video":
            if isinstance(step, RelateStep):
                raise ValueError(f"{step.id}: relate requires an earlier result, not video")
            parents = []
        elif step.source not in calls:
            raise ValueError(f"{step.id} references an unknown or later source: {step.source}")
        else:
            parents = lineage[step.source]
        _unique(step.evidence_need_ids, f"{step.id} evidence need ids")
        if not set(step.evidence_need_ids) <= known_needs:
            raise ValueError(f"{step.id} references an unknown evidence need")
        calls[step.id] = step
        lineage[step.id] = [*parents, step.id]
    if any(source not in calls for source in program.inspect.sources):
        raise ValueError("inspect references an unknown retrieval output")
    reachable = {ancestor for source in program.inspect.sources for ancestor in lineage[source]}
    unused = set(calls) - reachable
    if unused:
        raise ValueError("Unused retrieval outputs: " + ", ".join(sorted(unused)))
    covered = {need for node in reachable for need in calls[node].evidence_need_ids}
    if covered != known_needs:
        raise ValueError(
            "Final source lineage is missing evidence needs: "
            + ", ".join(sorted(known_needs - covered))
        )
    return calls, lineage


class SVPProgramValidator:
    """Verify plan linkage without forcing separate global searches per need."""

    def validate(self, program: SVPProgram, plan: SVPPlan) -> SVPProgram:
        # Revalidate to reject invalid objects mutated after initial construction.
        SVPProgram.model_validate(program.model_dump())
        SVPPlan.model_validate(plan.model_dump())
        if program.max_intervals > plan.max_intervals:
            raise ValueError("Tool program exceeds the plan's interval budget")
        if program.max_inspected_seconds > plan.max_inspected_seconds:
            raise ValueError("Tool program exceeds the plan's video-seconds budget")
        if set(program.inspect.evidence_need_ids) != {need.id for need in plan.evidence_needs}:
            raise ValueError("Inspect IDs must match all and only the plan's evidence needs")
        calls, lineage = _graph(program)
        missing_dependencies = []
        for need in plan.evidence_needs:
            direction = {"before": "before", "after": "after", "outcome": "after"}.get(
                need.temporal_role
            )
            targets = [node for node, call in calls.items() if need.id in call.evidence_need_ids]
            for dependency in need.depends_on:
                # A context/anchor need can refine an existing event's candidate
                # scope without expanding time. Require a genuine earlier source
                # edge: co-tagging one call or searching video independently does
                # not establish this dependency. Directional needs still require
                # the explicit temporal operator below.
                if direction is None and any(
                    dependency in calls[parent].evidence_need_ids
                    for target in targets
                    for parent in lineage[target][:-1]
                ):
                    continue
                matches = [
                    relation
                    for relation in program.relate
                    if (direction is None or relation.relation == direction)
                    and any(relation.id in lineage[target] for target in targets)
                    and any(
                        dependency in calls[parent].evidence_need_ids
                        for parent in lineage[relation.source]
                    )
                ]
                if not matches:
                    missing_dependencies.append(
                        f"Dependency {dependency} -> {need.id} requires " + (
                            f"relate({direction}) in target ancestry, with the dependency "
                            "tagged in its source ancestry"
                            if direction else
                            "a source-scoped target descended from an earlier output "
                            "tagged with the dependency; independent searches and "
                            "same-step tags do not establish a dependency"
                        )
                    )
        if missing_dependencies:
            # One bounded model repair must see every missing edge, not repair
            # the first only to discover a second defect on its final attempt.
            raise ValueError("; ".join(missing_dependencies))
        return program


def validate_program(program: SVPProgram, plan: SVPPlan) -> SVPProgram:
    return SVPProgramValidator().validate(program, plan)


def validate_svp_task_program(program: SVPProgram, plan: SVPPlan, scenes: list[dict]) -> SVPProgram:
    """Opt-in demo task gate, separate from the general compositional contract.

    Whole-video requests need a direct-video overview kept as a final inspection
    source. A scoped overview of one temporal anchor is not a global overview.
    No restriction is added to dependency chains or temporal horizon choices.
    """
    SVPProgramValidator().validate(program, plan)
    if not scenes:
        raise ValueError("Cannot retrieve from an empty video index")
    if plan.scope == "whole_video":
        required = min(program.max_intervals, len(scenes))
        roots = [
            step
            for step in program.steps
            if isinstance(step, OverviewStep)
            and step.source == "video"
            and step.id in program.inspect.sources
        ]
        if not roots:
            raise ValueError(
                "whole_video requires overview(source='video') directly in inspect.sources; "
                "an overview scoped to one retrieved event is not whole-video coverage"
            )
        if not any(step.limit >= required for step in roots):
            raise ValueError(
                f"whole_video overview needs limit >= {required} to supply distinct "
                "representatives within the program's interval budget"
            )
    return program


def _matrix(values, rows: int, name: str, *, dimension: int | None = None):
    array = np.asarray(values, dtype=np.float64)
    if rows == 0 and array.size == 0 and array.ndim == 1:
        array = np.empty((0, dimension or 0), dtype=np.float64)
    if (
        array.ndim != 2
        or len(array) != rows
        or (rows and array.shape[1] == 0)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite matrix with {rows} rows")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("Scene and query embeddings must have the same dimension")
    with np.errstate(over="ignore", under="ignore"):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{name} contains an empty or non-finite-norm embedding")
    return array / norms


def _public(hit: dict) -> dict:
    return {key: value for key, value in hit.items() if key != "position"}


def _overview(positions: list[int], vectors, scenes, limit: int) -> list[dict]:
    """Pick a semantic medoid, then novel, temporally spread representatives."""
    if not positions:
        return []
    similarities = np.clip(vectors[positions] @ vectors[positions].T, -1.0, 1.0)
    centrality = similarities.mean(axis=1)
    mids = np.array([(scenes[p]["start"] + scenes[p]["end"]) / 2 for p in positions])
    spread = max(float(np.ptp(mids)), 1.0)
    selected, hits = [], []
    for _ in range(min(limit, len(positions))):
        scores = {}
        for local, position in enumerate(positions):
            if local in selected:
                continue
            novelty = (1.0 - float(np.max(similarities[local, selected]))) / 2 if selected else 0.0
            temporal_spread = (
                float(np.min(np.abs(mids[local] - mids[selected]))) / spread if selected else 0.0
            )
            score = 0.6 * novelty + 0.4 * temporal_spread if selected else float(centrality[local])
            scores[local] = (score, novelty, temporal_spread, position)
        best = max(scores, key=lambda local: (scores[local][0], -scores[local][3]))
        score, novelty, temporal_spread, position = scores[best]
        hits.append(
            {
                "position": position,
                "scene": scenes[position]["index"],
                "representative_score": score,
                "selection_method": "novelty_and_temporal_spread"
                if selected
                else "semantic_medoid",
                "semantic_centrality": float(centrality[best]),
                "semantic_novelty": novelty,
                "temporal_spread": temporal_spread,
            }
        )
        selected.append(best)
    return hits


def _relation_hits(step: RelateStep, source_hits: list[dict], scenes) -> list[dict]:
    if not source_hits:
        return []
    anchor = scenes[source_hits[0]["position"]]
    hits = []
    for position, scene in enumerate(scenes):
        overlaps = scene["start"] < anchor["end"] and scene["end"] > anchor["start"]
        if overlaps:
            gap, kind = 0.0, "anchor_overlap"
        elif step.relation == "during":
            continue
        elif step.relation == "after":
            gap, kind = scene["start"] - anchor["end"], "strict_after"
        elif step.relation == "before":
            gap, kind = anchor["start"] - scene["end"], "strict_before"
        else:
            gap = max(scene["start"] - anchor["end"], anchor["start"] - scene["end"])
            kind = "strict_after" if scene["start"] >= anchor["end"] else "strict_before"
        if not 0 <= gap <= step.horizon_seconds:
            continue
        hits.append(
            {
                "position": position,
                "scene": scene["index"],
                "anchor_scene": anchor["index"],
                "anchor_overlap": overlaps,
                "window_relation": kind,
                "gap_seconds": float(gap),
                "temporal_proximity": 1.0 - float(gap) / step.horizon_seconds,
            }
        )
    # Proximity is a candidate-generation rule, not a semantic similarity score.
    # Strict neighbors win equal-distance ties; overlap remains available and flagged.
    hits.sort(key=lambda hit: (hit["gap_seconds"], hit["anchor_overlap"], hit["position"]))
    return hits


def _relation_targets(step: RelateStep, program: SVPProgram) -> list[LocateStep]:
    """Use the declared target query, not a query from an unrelated branch.

    Embedding a direct child's query does not execute that child early: its
    actual locate still searches only the relation's returned, limited scope.
    A same-need video search supplies the older direct-search/relate pattern.
    """
    children = [locate for locate in program.locate if locate.source == step.id]
    return children or [
        locate
        for locate in program.locate
        if locate.source == "video" and set(locate.evidence_need_ids) & set(step.evidence_need_ids)
    ]


def _select(program, scenes, outputs, calls, lineage, *, scope="local_event"):
    sources = program.inspect.sources
    needs = program.inspect.evidence_need_ids
    inherited = {
        source: [
            need
            for need in needs
            if any(need in calls[node].evidence_need_ids for node in lineage[source])
        ]
        for source in sources
    }
    positions = sorted({hit["position"] for source in sources for hit in outputs[source]})
    global_sources = (
        [
            source
            for source in sources
            if isinstance(calls[source], OverviewStep) and calls[source].source == "video"
        ]
        if scope == "whole_video"
        else []
    )
    global_positions = {hit["position"] for source in global_sources for hit in outputs[source]}
    global_required = min(program.max_intervals, len(scenes)) if scope == "whole_video" else 0
    duration = {
        position: scenes[position]["end"] - scenes[position]["start"] for position in positions
    }
    temporal_sources = {
        source
        for source in sources
        if any(isinstance(calls[node], RelateStep) for node in lineage[source])
    }
    required_relations = {step.id for step in program.relate}
    temporal_lineage = {
        position: {
            node for source in sources
            if any(hit["position"] == position for hit in outputs[source])
            for node in lineage[source] if node in required_relations
        }
        for position in positions
    }
    need_sources = {}
    for need in needs:
        # An explicitly inspected anchor must not be "covered" merely because
        # an outcome inherited its navigation label. For each actual target,
        # prefer constrained temporal evidence over a broad direct search.
        direct = [source for source in sources if need in calls[source].evidence_need_ids]
        candidates = direct or [source for source in sources if need in inherited[source]]
        temporal = [source for source in candidates if source in temporal_sources]
        need_sources[need] = temporal or candidates
    ranks = {
        source: {hit["position"]: 1.0 / (rank + 1) for rank, hit in enumerate(outputs[source])}
        for source in sources
    }
    need_ranks, utilities, global_utilities = {}, {}, {}
    temporal_positions = {hit["position"] for source in temporal_sources for hit in outputs[source]}
    for position in positions:
        need_ranks[position] = {
            need: max(
                (ranks[source].get(position, 0.0) for source in need_sources[need]), default=0.0
            )
            for need in needs
        }
        # One scene gets its best rank, never one vote per duplicate source.
        utilities[position] = max(ranks[source].get(position, 0.0) for source in sources)
        global_utilities[position] = max(
            (ranks[source].get(position, 0.0) for source in global_sources), default=0.0
        )
    # At most 8 * 12 terminal candidates and 3 final windows: bounded enumeration
    # avoids greedy admission missing feasible need coverage. Root overview
    # rank protects its actual diversity order, not just membership in its pool.
    best, best_score = (), None
    for count in range(min(3, program.max_intervals, len(positions)) + 1):
        for candidate in combinations(positions, count):
            seconds = sum(duration[position] for position in candidate)
            if seconds > min(60.0, program.max_inspected_seconds) + 1e-9:
                continue
            best_need_ranks = [
                max((need_ranks[position][need] for position in candidate), default=0.0)
                for need in needs
            ]
            score = (
                len(set(candidate) & global_positions),
                sum(rank > 0 for rank in best_need_ranks),
                required_relations.issubset({
                    node for position in candidate for node in temporal_lineage[position]
                }),
                sum(global_utilities[position] for position in candidate),
                sum(best_need_ranks),
                len(set(candidate) & temporal_positions),
                sum(utilities[position] for position in candidate),
                -seconds,
            )
            if best_score is None or score > best_score:
                best, best_score = candidate, score
    selected = sorted(best, key=lambda position: (scenes[position]["start"], position))
    selected_sources = [
        source for source in sources if any(hit["position"] in selected for hit in outputs[source])
    ]
    covered_needs = {
        need for need in needs if any(need_ranks[position][need] > 0 for position in selected)
    }
    covered_lineage = {node for source in selected_sources for node in lineage[source]}
    reasons = []
    for position in selected:
        entries = []
        for source in sources:
            for rank, hit in enumerate(outputs[source]):
                if hit["position"] == position:
                    entries.append(
                        {
                            "source": source,
                            "rank": rank + 1,
                            "purpose": "terminal_evidence",
                            "evidence_need_ids": calls[source].evidence_need_ids,
                            "lineage_evidence_need_ids": inherited[source],
                            "ancestry": lineage[source],
                            **{key: value for key, value in _public(hit).items() if key != "scene"},
                        }
                    )
        reasons.append({"scene": scenes[position]["index"], "reasons": entries})
    uncovered = [need for need in needs if need not in covered_needs]
    uncovered_sources = [source for source in sources if source not in selected_sources]
    uncovered_relations = [step.id for step in program.relate if step.id not in covered_lineage]
    coverage = {
        "scope": "retrieval_candidate_lineage_not_visual_confirmation",
        "need_sources": need_sources,
        "selection_policy": (
            "global_count_then_need_and_temporal_coverage_then_overview_rank_then_unique_hits"
        ),
        "selected_sources": selected_sources,
        "uncovered_sources": uncovered_sources,
        "uncovered_evidence_needs": uncovered,
        "uncovered_temporal_outputs": uncovered_relations,
        "global_representative_sources": global_sources,
        "global_representatives_required": global_required,
        "global_representatives_selected": len(set(selected) & global_positions),
        "global_coverage_complete": len(set(selected) & global_positions) >= global_required,
        "navigation_only_needs": [
            need
            for need in needs
            if need in covered_needs
            and not any(need in calls[source].evidence_need_ids for source in selected_sources)
        ],
    }
    return selected, reasons, coverage


def execute_svp_program(
    program: SVPProgram,
    scenes: list[dict],
    vectors,
    query_vectors,
    *,
    scope: Literal["local_event", "whole_video"] = "local_event",
) -> dict:
    """Execute ordered source-scoped operations; only final leaves cost video time.

    ``query_vectors`` follows ``program.locate`` order. Overview-only programs
    accept an empty query matrix. Cached retrieval intermediates do not cause
    frame inspection. An empty source stays empty; no fallback broadens it.
    """
    if scope not in {"local_event", "whole_video"}:
        raise ValueError(f"Unknown retrieval scope: {scope}")
    SVPProgram.model_validate(program.model_dump())
    calls, lineage = _graph(program)
    if not scenes:
        raise ValueError("Cannot retrieve from an empty video index")
    if any(
        not isinstance(scene["index"], int)
        or isinstance(scene["index"], bool)
        or scene["index"] < 0
        for scene in scenes
    ):
        raise ValueError("Scene indices must be nonnegative integers")
    if len({scene["index"] for scene in scenes}) != len(scenes):
        raise ValueError("Scene indices must be unique")
    for scene in scenes:
        if (
            not np.isfinite([scene["start"], scene["end"]]).all()
            or scene["start"] < 0
            or scene["end"] <= scene["start"]
        ):
            raise ValueError("Scene spans must be finite, positive video intervals")
    matrix = _matrix(vectors, len(scenes), "Scene embeddings")
    queries = _matrix(
        query_vectors, len(program.locate), "Query embeddings", dimension=matrix.shape[1]
    )
    cosine = np.clip(matrix @ queries.T, -1.0, 1.0)
    query_indices = {step.id: index for index, step in enumerate(program.locate)}
    outputs, trace = {}, []
    for step in program.steps:
        source_hits = outputs[step.source] if step.source != "video" else []
        positions = (
            list(range(len(scenes)))
            if step.source == "video"
            else [hit["position"] for hit in source_hits]
        )
        record = {
            "call": step.model_dump(),
            "ancestry": lineage[step.id],
            "source": step.source,
            "source_input_hits": [_public(hit) for hit in source_hits],
            "source_scene_ids": [scenes[position]["index"] for position in positions],
        }
        if isinstance(step, LocateStep):
            query_index = query_indices[step.id]
            ordered = sorted(
                positions, key=lambda position: (-cosine[position, query_index], position)
            )
            hits = []
            prior = {hit["position"]: hit for hit in source_hits}
            for position in ordered[: step.limit]:
                hit = {
                    "position": position,
                    "scene": scenes[position]["index"],
                    "score": float(cosine[position, query_index]),
                    "semantic_cosine": float(cosine[position, query_index]),
                }
                if position in prior:
                    hit["source_hit"] = _public(prior[position])
                hits.append(hit)
            record.update(score_type="semantic_cosine", ranking=["semantic_cosine"])
            record["candidate_scope_scene_ids"] = record["source_scene_ids"]
        elif isinstance(step, OverviewStep):
            hits = _overview(positions, matrix, scenes, step.limit)
            record.update(
                score_type="representative_diversity",
                ranking=["semantic_medoid", "0.6 * semantic_novelty + 0.4 * temporal_spread"],
                candidate_scope_scene_ids=record["source_scene_ids"],
                coverage_policy="semantic_representatives_with_temporal_spread_not_required_thirds",
            )
        else:
            candidates = _relation_hits(step, source_hits, scenes)
            targets = _relation_targets(step, program)
            for hit in candidates:
                if targets:
                    target_scores = {
                        target.id: float(cosine[hit["position"], query_indices[target.id]])
                        for target in targets
                    }
                    semantic = max(target_scores.values())
                    hit.update(
                        score=semantic,
                        semantic_cosine=semantic,
                        target_query_cosines=target_scores,
                    )
            if targets:
                candidates.sort(
                    key=lambda hit: (
                        -hit["semantic_cosine"],
                        hit["gap_seconds"],
                        hit["anchor_overlap"],
                        hit["position"],
                    )
                )
            hits = candidates[: step.limit]
            record.update(
                score_type="semantic_cosine" if targets else "temporal_proximity",
                ranking=(["semantic_cosine"] if targets else [])
                + ["temporal_proximity", "strict_before_overlap"],
                target_query_sources=[target.id for target in targets],
                target_query_policy="direct_child_else_same_need_video_locate",
                limit_policy="rank_eligible_candidates_before_limit",
                anchor_policy="top_ranked_source",
                anchor_scene=source_hits[0]["scene"] if source_hits else None,
                candidate_scope="video_temporal_neighborhood_of_source_anchor",
                candidate_scope_scene_ids=[hit["scene"] for hit in candidates],
                ordering_scope="windows_only; anchor_overlap does not establish event ordering",
            )
        outputs[step.id] = hits
        record["hits"] = [_public(hit) for hit in hits]
        record["empty_source"] = step.source != "video" and not source_hits
        trace.append(record)
    selected, reasons, coverage = _select(program, scenes, outputs, calls, lineage, scope=scope)
    seconds = sum(scenes[position]["end"] - scenes[position]["start"] for position in selected)
    trace.append(
        {
            "call": "inspect",
            "sources": program.inspect.sources,
            "source_input_hits": {
                source: [_public(hit) for hit in outputs[source]]
                for source in program.inspect.sources
            },
            "selected": [scenes[position]["index"] for position in selected],
            "inspected_seconds": seconds,
            "budget_seconds": min(60.0, program.max_inspected_seconds),
            "budget_intervals": min(3, program.max_intervals),
            "coverage_scope": coverage["scope"],
            "selection_reasons": reasons,
            **coverage,
        }
    )
    return {
        "selected": [scenes[position] for position in selected],
        "calls": trace,
        "selection_reasons": reasons,
        "coverage": coverage,
        "uncovered_evidence_needs": coverage["uncovered_evidence_needs"],
        "uncovered_temporal_outputs": coverage["uncovered_temporal_outputs"],
        "uncovered_sources": coverage["uncovered_sources"],
        "scope": scope,
    }
