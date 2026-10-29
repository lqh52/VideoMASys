"""Typed, bounded visual assertions and their provenance; never visual truth proofs.

Only controller-provided frame registries identify pixels and approximate seek
times. Observations remain model interpretations. Inferences retain their direct
and dependency provenance; reviewer coverage is checked independently of content.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$"
)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalText = Annotated[str, StringConstraints(strip_whitespace=True)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class FrameRef(_StrictModel):
    id: Identifier
    scene: int = Field(ge=0)
    seek_seconds: float = Field(ge=0)
    path: Text = Field(max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservedFact(_StrictModel):
    kind: Literal["state", "action", "change"]
    subject: Text = Field(max_length=160)
    predicate: Text = Field(max_length=220)
    object: Text | None = Field(max_length=160)
    frame_ids: list[Identifier] = Field(min_length=1, max_length=4)
    need_ids: list[Identifier] = Field(min_length=1, max_length=3)
    uncertainty: OptionalText = Field(max_length=220)


class UnresolvedNeed(_StrictModel):
    need_id: Identifier
    reason: Text = Field(max_length=220)


class WindowObservation(_StrictModel):
    facts: list[ObservedFact] = Field(max_length=3)
    unresolved: list[UnresolvedNeed] = Field(max_length=3)
    refine_frame_id: Identifier | None


class ObservationRecord(ObservedFact):
    id: Identifier
    scene: int = Field(ge=0)
    source: Literal["model_interpretation_of_sampled_frames"] = (
        "model_interpretation_of_sampled_frames"
    )


class Claim(_StrictModel):
    """Common claim interface; typed answer variants expose kind-specific constraints."""

    id: Identifier
    kind: Literal["observed", "inferred", "limitation"]
    statement: Text = Field(max_length=300)
    frame_ids: list[Identifier] = Field(max_length=6)
    observation_ids: list[Identifier] = Field(max_length=6)
    depends_on: list[Identifier] = Field(max_length=3)
    need_ids: list[Identifier] = Field(min_length=1, max_length=3)
    rationale: OptionalText = Field(max_length=220)

    @field_validator("frame_ids", "observation_ids", "depends_on", "need_ids")
    @classmethod
    def ordered_reference_sets(cls, values):
        # Run after field bounds/types: repeated references add no evidence or
        # edges, but must not let an oversized response bypass its budget.
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def valid_kind(self):
        if self.kind != "inferred" and self.depends_on:
            raise ValueError("Only inferred claims may depend on other claims")
        if self.kind == "inferred" and not self.rationale:
            raise ValueError("An inferred claim requires a rationale connecting visible evidence")
        return self


class ObservedClaim(Claim):
    kind: Literal["observed"]
    depends_on: list[Identifier] = Field(max_length=0)
    rationale: OptionalText = Field(max_length=0)


class InferredClaim(Claim):
    kind: Literal["inferred"]
    rationale: Text = Field(max_length=220)


class LimitationClaim(Claim):
    kind: Literal["limitation"]
    depends_on: list[Identifier] = Field(max_length=0)
    rationale: OptionalText = Field(max_length=0)


class OptionAssessment(_StrictModel):
    choice: Text = Field(max_length=300)
    # Unsupported means insufficient support in this evidence, not disproof.
    # Selection is separate: more than one choice may have some support.
    status: Literal["supported", "unsupported"]
    claim_ids: list[Identifier] = Field(max_length=4)

    @field_validator("claim_ids")
    @classmethod
    def ordered_reference_set(cls, values):
        return list(dict.fromkeys(values))


class StructuredAnswer(_StrictModel):
    # Disjoint kind literals generate anyOf, not a provider-incompatible oneOf
    # discriminator; non-inferred dependencies must be [] in the wire schema.
    claims: list[ObservedClaim | InferredClaim | LimitationClaim] = Field(
        min_length=1, max_length=5
    )
    answer_claim_ids: list[Identifier] = Field(min_length=1, max_length=3)
    selected_choice: Text | None = Field(max_length=300)
    choice_change_reason: OptionalText = Field(default="", max_length=300)
    option_comparisons: list[OptionAssessment] = Field(max_length=5)

    @field_validator("answer_claim_ids")
    @classmethod
    def ordered_reference_set(cls, values):
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def valid_graph(self):
        _answer_graph(self)
        return self


class ReviewCheck(_StrictModel):
    target_id: Identifier
    status: Literal["pass", "revise"]
    reason: Text = Field(max_length=200)


class StructuredReview(_StrictModel):
    checks: list[ReviewCheck] = Field(min_length=1, max_length=20)


class DirectOptionReview(_StrictModel):
    choice: Text = Field(max_length=300)
    status: Literal["supported", "unsupported"]
    reason: Text = Field(max_length=200)


class AnswerJudgment(_StrictModel):
    status: Literal["pass", "revise"]
    selected_choice: Text | None = Field(max_length=300)
    reason: Text = Field(max_length=200)


class DirectStructuredReview(_StrictModel):
    """Review claims, option support, and answer sufficiency as separate decisions."""

    claim_checks: list[ReviewCheck] = Field(min_length=1, max_length=5)
    option_assessments: list[DirectOptionReview] = Field(max_length=5)
    answer_judgment: AnswerJudgment


def _unique(values, label):
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must be nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _refs(values, allowed, label):
    _unique(values, label)
    if not set(values) <= set(allowed):
        raise ValueError(f"{label} contain unknown IDs")


def _model(schema, value):
    # Revalidate even models mutated after initial construction.
    return schema.model_validate(value.model_dump() if isinstance(value, BaseModel) else value)


def _frames(values):
    values = [_model(FrameRef, value) for value in values]
    _unique([value.id for value in values], "Frame IDs")
    if not values:
        raise ValueError("An actual nonempty frame registry is required")
    return {value.id: value for value in values}


def _fact(value, frames, need_ids):
    _refs(value.frame_ids, frames, "Observation frame references")
    _refs(value.need_ids, need_ids, "Observation need references")
    times = [frames[frame_id].seek_seconds for frame_id in value.frame_ids]
    if value.kind == "change" and (len(times) < 2 or any(
        right <= left for left, right in zip(times, times[1:], strict=False)
    )):
        raise ValueError("A change requires at least two distinct chronological frames")
    if value.kind == "change" and len({frames[name].sha256 for name in value.frame_ids}) < 2:
        raise ValueError("A change requires at least two distinct frame image hashes")


def validate_window(value, frames: list[FrameRef], need_ids: list[str], allow_refine=True):
    """Bind one model response to exactly its visual request, not other cached frames."""
    value = _model(WindowObservation, value)
    frames = _frames(frames)
    _unique(need_ids, "Requested need IDs")
    if not need_ids or len({frame.scene for frame in frames.values()}) != 1:
        raise ValueError("Window inspection requires requested needs and frames from one scene")
    covered = set()
    for fact in value.facts:
        _fact(fact, frames, need_ids)
        covered.update(fact.need_ids)
    unresolved = [item.need_id for item in value.unresolved]
    _refs(unresolved, need_ids, "Unresolved need references")
    if covered | set(unresolved) != set(need_ids):
        raise ValueError("Every requested need requires an observation or an unresolved entry")
    if value.refine_frame_id is not None:
        if not allow_refine:
            raise ValueError("Refinement responses must have refine_frame_id=null")
        _refs([value.refine_frame_id], frames, "Refinement frame reference")
    return value


def _answer_graph(value):
    ids = [claim.id for claim in value.claims]
    _unique(ids, "Claim IDs")
    if any(name.startswith(("option_", "obs_")) for name in ids):
        raise ValueError("Claim IDs cannot use reserved option_ or obs_ review target prefixes")
    earlier = set()
    for claim in value.claims:
        _refs(claim.depends_on, earlier, "Claim dependencies (earlier claims only)")
        for name in ("frame_ids", "observation_ids", "need_ids"):
            _unique(getattr(claim, name), f"Claim {name}")
        earlier.add(claim.id)
    _refs(value.answer_claim_ids, ids, "Answer claim references")
    for option in value.option_comparisons:
        _refs(option.claim_ids, ids, "Option claim references")


def _claim_closure(claims, roots):
    reachable = set(roots)
    for claim in reversed(claims):
        if claim.id in reachable:
            reachable.update(claim.depends_on)
    return reachable


def claim_partition(value):
    """Expose existing proof and context without deleting claims or adding edges.

    Context still needs valid provenance and review. Its presence does not satisfy
    answer need coverage or establish that it is irrelevant to the question.
    """
    value = _model(StructuredAnswer, value)
    answer_ids = _claim_closure(value.claims, value.answer_claim_ids)
    option_ids = _claim_closure(value.claims, [
        name for option in value.option_comparisons for name in option.claim_ids
    ])
    return {
        "answer_lineage_ids": [claim.id for claim in value.claims if claim.id in answer_ids],
        "option_lineage_ids": [claim.id for claim in value.claims if claim.id in option_ids],
        "context_claim_ids": [claim.id for claim in value.claims
                              if claim.id not in answer_ids | option_ids],
        "limitation_claim_ids": [claim.id for claim in value.claims
                                 if claim.kind == "limitation"],
    }


def answer_need_coverage(value, need_ids):
    """Report planner labels addressed by answer lineage or displayed limitations.

    Coverage is bookkeeping, not proof that a need is satisfied. Unaddressed
    search-plan needs do not invalidate an otherwise supported answer.
    """
    value = _model(StructuredAnswer, value)
    _unique(need_ids, "Requested need IDs")
    for claim in value.claims:
        _refs(claim.need_ids, need_ids, "Claim need references")
    displayed = _claim_closure(value.claims, value.answer_claim_ids) | {
        claim.id for claim in value.claims if claim.kind == "limitation"
    }
    covered = {need for claim in value.claims if claim.id in displayed for need in claim.need_ids}
    return {
        "covered_need_ids": [need for need in need_ids if need in covered],
        "unaddressed_need_ids": [need for need in need_ids if need not in covered],
    }


def _positive_claim_ids(claims):
    """Follow positive evidence paths in an already validated, acyclic claim graph."""
    positive = set()
    for claim in claims:
        if claim.kind in {"observed", "inferred"} and (
            claim.frame_ids or claim.observation_ids or positive.intersection(claim.depends_on)
        ):
            positive.add(claim.id)
    return positive


def claim_provenance(value: StructuredAnswer, observations: list[ObservationRecord]):
    """Resolve existing references without rewriting claims or inventing evidence.

    ``validate_answer`` checks active registry membership before this expansion.
    Observation links already identify pixels: models need not copy those frame
    IDs into each claim. Dependencies inherit the same resolved provenance.
    """
    value = _model(StructuredAnswer, value)
    records = [_model(ObservationRecord, record) for record in observations]
    _unique([record.id for record in records], "Observation IDs")
    records = {record.id: record for record in records}
    result = {}
    for claim in value.claims:
        _refs(claim.observation_ids, records, "Claim observation references")
        direct = {
            "frame_ids": claim.frame_ids + [
                frame for name in claim.observation_ids for frame in records[name].frame_ids
            ],
            "observation_ids": claim.observation_ids,
        }
        result[claim.id] = {
            field: list(dict.fromkeys(
                direct[field]
                + [reference for parent in claim.depends_on for reference in result[parent][field]]
            ))
            for field in ("frame_ids", "observation_ids")
        }
        if not result[claim.id]["frame_ids"]:
            raise ValueError("Every claim requires nonempty resolved provenance from actual frames")
    return result


def validate_answer(value, frames, observations, need_ids, options):
    """Validate references and reasoning structure, not the semantic truth of a claim."""
    value = _model(StructuredAnswer, value)
    frames = _frames(frames)
    _unique(need_ids, "Requested need IDs")
    _unique(options, "Supplied options")
    records = [_model(ObservationRecord, record) for record in observations]
    _unique([record.id for record in records], "Observation IDs")
    records = {record.id: record for record in records}
    for record in records.values():
        _fact(record, frames, need_ids)
        if any(frames[name].scene != record.scene for name in record.frame_ids):
            raise ValueError("Observation scene must match its actual frame references")
    if set(records) & {claim.id for claim in value.claims}:
        raise ValueError("Claim and observation IDs must be distinct review targets")
    claims = {claim.id: claim for claim in value.claims}
    for claim in value.claims:
        _refs(claim.frame_ids, frames, "Claim frame references")
        _refs(claim.observation_ids, records, "Claim observation references")
        _refs(claim.need_ids, need_ids, "Claim need references")
    claim_provenance(value, list(records.values()))
    answer_lineage = _claim_closure(value.claims, value.answer_claim_ids)
    compared = [item.choice for item in value.option_comparisons]
    _unique(compared, "Compared options")
    if set(compared) != set(options):
        raise ValueError("Compare every supplied option exactly once, and no other options")
    # Choice identity, not response order, binds each assessment. Canonicalize
    # only after exact coverage/uniqueness; the caller's raw response is untouched.
    by_choice = {item.choice: item for item in value.option_comparisons}
    value.option_comparisons = [by_choice[choice] for choice in options]
    if value.selected_choice is None:
        if options and not any(
            claims[name].kind == "limitation" for name in value.answer_claim_ids
        ):
            raise ValueError("An unresolved multiple-choice answer requires an explicit limitation")
    elif value.selected_choice not in options:
        raise ValueError("selected_choice must be a supplied option")
    else:
        selected = next(
            item for item in value.option_comparisons if item.choice == value.selected_choice
        )
        if selected.status != "supported":
            raise ValueError("The selected option must be supported")
        if not set(selected.claim_ids) <= answer_lineage:
            raise ValueError("The selected option must be justified by the answer's claim lineage")
    positive = _positive_claim_ids(value.claims)
    for option in value.option_comparisons:
        kinds = {claims[name].kind for name in option.claim_ids}
        if option.status == "supported" and (
            not kinds or not kinds <= {"observed", "inferred"}
        ):
            raise ValueError(
                "A supported option requires observed or inferred claims, not limitations"
            )
        if option.status == "supported" and not set(option.claim_ids) <= positive:
            raise ValueError(
                "A supported option requires a positive evidence path; inferred claims "
                "cannot inherit their only support from limitations"
            )
    return value


def validate_review(value, expected_targets):
    """Require exactly one review of every specified claim/option/observation target."""
    value = _model(StructuredReview, value)
    _unique(expected_targets, "Expected review targets")
    actual = [check.target_id for check in value.checks]
    _unique(actual, "Review targets")
    if set(actual) != set(expected_targets):
        raise ValueError(
            "Review must cover every expected target exactly once, without unknown targets"
        )
    return value


def validate_choice_change(value, previous_choice):
    """Require an explicit reason for a changed choice, not proof the reason is true."""
    value = _model(StructuredAnswer, value)
    if value.selected_choice != previous_choice and not value.choice_change_reason:
        raise ValueError(
            "Changing selected_choice requires a concise evidence-based choice_change_reason, "
            "including when changing to or from abstention"
        )
    return value


def validate_direct_review(value, draft, options):
    """Validate complete direct judgments; the controller derives agreement."""
    value = _model(DirectStructuredReview, value)
    draft = _model(StructuredAnswer, draft)
    _unique(options, "Supplied options")
    expected = [claim.id for claim in draft.claims]
    actual = [check.target_id for check in value.claim_checks]
    _unique(actual, "Claim review targets")
    if set(actual) != set(expected):
        raise ValueError("Review must cover every claim, including context, exactly once")
    compared = [item.choice for item in value.option_assessments]
    _unique(compared, "Reviewed options")
    if set(compared) != set(options):
        raise ValueError("Review must assess every supplied option exactly once, and no others")
    by_choice = {item.choice: item for item in value.option_assessments}
    value.option_assessments = [by_choice[choice] for choice in options]
    judgment = value.answer_judgment
    if judgment.selected_choice is not None and judgment.selected_choice not in options:
        raise ValueError("Reviewed selected_choice must be a supplied option or null")
    if judgment.selected_choice is not None:
        selected = next(item for item in value.option_assessments
                        if item.choice == judgment.selected_choice)
        if selected.status != "supported":
            raise ValueError(
                f"Reviewed selected_choice {judgment.selected_choice!r} must be marked "
                "supported in this review's option_assessments; correct the conflicting "
                "assessment or selection using the evidence."
            )
    if judgment.status == "pass" and judgment.selected_choice != draft.selected_choice:
        raise ValueError("A passing answer judgment cannot change the draft selected_choice")
    return value


def answer_diagnostics(value, frames, observations, need_ids, options):
    """Collect actionable repair hints, never repair or accept an invalid answer.

    The regular validator remains authoritative. Best-effort inspection of a raw
    response lets one bounded repair see reference and selection errors together.
    Unaddressed planner needs are advisory, never an instruction to invent support.
    """
    data = value.model_dump() if isinstance(value, BaseModel) else value
    issues = []

    def coverage_advisories(unaddressed):
        return [{
            "code": "unaddressed_planner_needs", "ids": unaddressed,
            "message": "These search-plan needs are not addressed by the answer lineage or "
            "limitations. This is advisory, not an answer-validation failure.",
        }] if unaddressed else []

    def issue(code, path, message, ids=None):
        item = {"code": code, "path": path, "message": message}
        if ids is not None:
            item["ids"] = ids
        issues.append(item)

    try:
        validate_answer(value, frames, observations, need_ids, options)
    except ValueError as exc:
        if hasattr(exc, "errors"):
            for error in exc.errors(include_url=False):
                issue("validation", ".".join(map(str, error["loc"])), error["msg"])
        else:
            issue("validation", "", str(exc))
    if not isinstance(data, dict):
        return {"issues": issues, "covered_need_ids": [],
                "unaddressed_need_ids": list(need_ids),
                "advisories": coverage_advisories(list(need_ids)),
                "context_claim_ids": [], "available_observation_ids_by_need": {}}

    def strings(values):
        return ([item for item in values if isinstance(item, str)]
                if isinstance(values, list) else [])

    def records(values):
        return [item.model_dump() if isinstance(item, BaseModel) else item
                for item in values if isinstance(item, (dict, BaseModel))]

    claims = records(data.get("claims", []) if isinstance(data.get("claims"), list) else [])
    comparisons = records(data.get("option_comparisons", []) if
                          isinstance(data.get("option_comparisons"), list) else [])
    frame_ids = {item.get("id") for item in records(frames)}
    observed = {item.get("id"): item for item in records(observations)}
    known_claims = {item["id"]: item for item in claims if isinstance(item.get("id"), str)}
    try:
        positive = _positive_claim_ids(_model(StructuredAnswer, data).claims)
    except ValueError:
        positive = None  # The authoritative validation error explains a malformed graph.

    def refs(path, values, allowed):
        values = strings(values)
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            issue("invalid_reference", path, "Use only supplied or earlier valid IDs.", unknown)

    earlier = set()
    for index, item in enumerate(claims):
        for field, allowed in (("frame_ids", frame_ids), ("observation_ids", observed),
                               ("need_ids", need_ids), ("depends_on", earlier)):
            refs(f"claims[{index}].{field}", item.get(field), allowed)
        if isinstance(item.get("id"), str):
            earlier.add(item["id"])
    roots = strings(data.get("answer_claim_ids"))
    refs("answer_claim_ids", roots, known_claims)
    for index, item in enumerate(comparisons):
        refs(f"option_comparisons[{index}].claim_ids", item.get("claim_ids"), known_claims)
        cited = strings(item.get("claim_ids"))
        if item.get("status") == "supported" and (not cited or any(
            known_claims.get(name, {}).get("kind") not in {"observed", "inferred"}
            or (positive is not None and name not in positive)
            for name in cited
        )):
            issue("invalid_option_support", f"option_comparisons[{index}].claim_ids",
                  "Support requires observed or inferred claims with direct evidence or a "
                  "positive supporting ancestry, not only limitations. Revise support or "
                  "selection from the evidence; do not invent links.")

    def closure(roots):
        reached, pending = set(), list(roots)
        while pending:
            name = pending.pop()
            if name in reached or name not in known_claims:
                continue
            reached.add(name)
            pending.extend(strings(known_claims[name].get("depends_on")))
        return reached

    lineage = closure(roots)
    displayed = lineage | {name for name, item in known_claims.items()
                           if item.get("kind") == "limitation"}
    covered = {need for name in displayed for need in strings(known_claims[name].get("need_ids"))}
    unaddressed = [need for need in need_ids if need not in covered]
    selected = data.get("selected_choice")
    if selected is None and options and not any(
        known_claims.get(name, {}).get("kind") == "limitation" for name in roots
    ):
        issue("missing_abstention_limitation", "answer_claim_ids",
              "An unresolved selection needs a frame-grounded limitation in answer_claim_ids, "
              "not only observations. Use an existing limitation below if appropriate, or "
              "state the evidence gap. Replace premise-only roots within the three-root limit; "
              "do not add dependency edges or force a choice.",
              [name for name, item in known_claims.items() if item.get("kind") == "limitation"])
    elif selected is not None:
        selected_option = next((item for item in comparisons if item.get("choice") == selected), {})
        if selected not in options:
            issue("invalid_choice", "selected_choice", "Select a supplied choice or null.")
        elif selected_option.get("status") != "supported":
            issue("selected_status_conflict", "selected_choice",
                  "The selected choice lacks support; justify it or revise selection.")
        outside = sorted(set(strings(selected_option.get("claim_ids"))) - lineage)
        if outside:
            issue("selected_lineage_conflict", "answer_claim_ids",
                  "The selected choice cites these claims outside the answer's dependency "
                  "lineage. If they are the supported conclusions, use them as answer_claim_ids, "
                  "replacing premise-only roots already inherited through depends_on "
                  "(at most three roots). Keep justified claims and citations; do not make "
                  "premises depend on later conclusions or invent edges.",
                  outside)
    option_lineage = closure([
        name for item in comparisons for name in strings(item.get("claim_ids"))
    ])
    return {
        "issues": issues, "covered_need_ids": [need for need in need_ids if need in covered],
        "unaddressed_need_ids": unaddressed, "advisories": coverage_advisories(unaddressed),
        "context_claim_ids": [name for name in known_claims
                              if name not in lineage | option_lineage],
        "available_observation_ids_by_need": {
            need: [name for name, item in observed.items() if need in strings(item.get("need_ids"))]
            for need in need_ids
        },
    }
