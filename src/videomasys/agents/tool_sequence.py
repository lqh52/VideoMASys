from __future__ import annotations

import logging

from videomasys.domain.schemas import (
    InspectCall,
    LocateCall,
    ReasoningPlan,
    RelateCall,
    ToolProgram,
)
from videomasys.model_gateway.vlm import VLMGateway

logger = logging.getLogger(__name__)

TOOL_DATAFLOW_RULES = """\
Every locate/relate output must reach inspect, directly or through a later relate;
do not emit unused calls. The evidence IDs on the actual inspect.sources outputs
must cover every need: listing an ID in inspect.evidence_need_ids is not enough.
For EACH depends_on edge, include a relate whose source output carries the dependency
ID and whose evidence_need_ids includes the dependent need. Include that relate
output itself in inspect.sources so its temporal candidates are actually considered.
Use before for a before need, and after for an after/outcome need. Inspect may also
include independent locate probes. For chains, include the intermediate relations
as well as the final relation when they satisfy separate evidence needs.
"""


class InvalidToolProgram(ValueError):
    pass


class ToolProgramValidator:
    """Static gate for typed locate/relate programs."""

    def validate(self, program: ToolProgram, plan: ReasoningPlan) -> ToolProgram:
        if program.max_intervals > plan.max_intervals:
            raise InvalidToolProgram("tool program exceeds the plan's interval budget")
        if program.max_inspected_seconds > plan.max_inspected_seconds:
            raise InvalidToolProgram("tool program exceeds the plan's video-seconds budget")

        need_ids = {need.id for need in plan.evidence_needs}
        known_outputs: set[str] = set()
        for index, call in enumerate(program.locate, start=1):
            if call.id != f"l{index}":
                raise InvalidToolProgram("locate ids must be sequential")
            self._validate_need_ids(call.evidence_need_ids, need_ids, call.id)
            known_outputs.add(call.id)
        for index, call in enumerate(program.relate, start=1):
            if call.id != f"r{index}":
                raise InvalidToolProgram("relate ids must be sequential")
            if call.source not in known_outputs:
                raise InvalidToolProgram(f"{call.id} references an unknown or later source")
            self._validate_need_ids(call.evidence_need_ids, need_ids, call.id)
            known_outputs.add(call.id)

        if any(source not in known_outputs for source in program.inspect.sources):
            raise InvalidToolProgram("inspect references an unknown retrieval output")
        if len(program.inspect.sources) != len(set(program.inspect.sources)):
            raise InvalidToolProgram("inspect sources must be unique")
        self._validate_need_ids(program.inspect.evidence_need_ids, need_ids, "inspect")

        # Only aggregate after IDs and references are valid, so dataflow traversal
        # is safe and one repair can address all independent coverage omissions.
        errors = []
        if getattr(plan, "scope", None) == "whole_video":
            errors.extend(self._whole_video_errors(program, plan))
        located_need_ids = {
            need_id for call in program.locate for need_id in call.evidence_need_ids
        }
        missing_locates = need_ids - located_need_ids
        if missing_locates:
            errors.append(
                "locate calls must cover every evidence need; missing direct locate IDs: "
                + ", ".join(sorted(missing_locates))
                + "; add/correct direct locate queries matching those needs, tag their exact "
                "IDs, and include the outputs in inspect.sources and the IDs in "
                "inspect.evidence_need_ids. Do not only relabel an unrelated query"
            )
        missing_labels = need_ids - set(program.inspect.evidence_need_ids)
        if missing_labels:
            errors.append(
                "inspect must cover every reasoning-plan evidence need; add missing "
                "inspect.evidence_need_ids: " + ", ".join(sorted(missing_labels))
            )
        errors.extend(self._dataflow_errors(program, plan))
        if sum(call.limit for call in program.locate) > 24:
            errors.append("locate candidate budget exceeds 24")
        if missing_locates:
            # Put repair actions and all missing IDs before potentially long
            # descriptions: the live controller bounds its repair-error excerpt.
            errors.append(
                "Missing direct-locate need descriptions: "
                + "; ".join(
                    f"{need.id}: {need.description!r}"
                    for need in plan.evidence_needs
                    if need.id in missing_locates
                )
            )
        if errors:
            raise InvalidToolProgram(". ".join(errors))
        return program

    @staticmethod
    def _whole_video_errors(program: ToolProgram, plan: ReasoningPlan) -> list[str]:
        """Independent representative probes must not become an event chain."""
        errors = []
        needs = plan.evidence_needs
        if len(needs) != 3 or any(
            need.depends_on or need.temporal_role != "context" for need in needs
        ):
            errors.append("whole_video requires three independent context needs")
        if program.relate:
            errors.append(
                "whole_video forbids relate calls: set relate=[] and inspect only l1,l2,l3; "
                "early/middle/late are independent coverage probes, not event dependencies"
            )
        if len(program.locate) != 3:
            errors.append("whole_video requires exactly three direct locates: l1,l2,l3")
        for call, need, position in zip(
            program.locate, needs, ("beginning", "middle", "end"), strict=False
        ):
            if call.evidence_need_ids != [need.id] or call.temporal_position != position:
                errors.append(
                    f"whole_video {call.id} must carry only evidence_need_ids=[{need.id!r}] "
                    f"and temporal_position={position!r}; preserve plan order"
                )
        if set(program.inspect.sources) != {"l1", "l2", "l3"}:
            errors.append("whole_video inspect.sources must contain exactly l1,l2,l3")
        if set(program.inspect.evidence_need_ids) != {need.id for need in needs}:
            errors.append("whole_video inspect.evidence_need_ids must contain exactly all plan IDs")
        return errors

    @staticmethod
    def _dataflow_errors(program: ToolProgram, plan: ReasoningPlan) -> list[str]:
        errors = []
        outputs = {call.id: call for call in [*program.locate, *program.relate]}
        inspected = set(program.inspect.sources)
        reachable = set(inspected)
        pending = list(inspected)
        while pending:
            call = outputs[pending.pop()]
            if isinstance(call, RelateCall) and call.source not in reachable:
                reachable.add(call.source)
                pending.append(call.source)
        unused = sorted(outputs.keys() - reachable)
        if unused:
            errors.append(
                f"unused retrieval outputs: {', '.join(unused)}; connect intended outputs "
                "to inspect.sources. Do not delete calls required by evidence dependencies"
            )

        # A retrieval anchor is not necessarily inspected. Coverage comes from the
        # outputs admitted to the inspection union, not ancestry or claimed labels.
        covered = {need_id for source in inspected for need_id in outputs[source].evidence_need_ids}
        missing = {need.id for need in plan.evidence_needs} - covered
        if missing:
            errors.append(
                "inspect.sources do not supply evidence for: "
                + ", ".join(sorted(missing))
                + "; include outputs carrying these exact IDs, not only ancestor tags "
                "or inspect labels"
            )

        for need in plan.evidence_needs:
            direction = {"before": "before", "after": "after", "outcome": "after"}.get(
                need.temporal_role
            )
            for dependency in need.depends_on:
                matches = [
                    call
                    for call in program.relate
                    if need.id in call.evidence_need_ids
                    and dependency in outputs[call.source].evidence_need_ids
                    and (direction is None or call.relation == direction)
                ]
                if matches and not any(call.id in inspected for call in matches):
                    names = ", ".join(call.id for call in matches)
                    quantifier = "at least one of " if len(matches) > 1 else ""
                    errors.append(
                        f"dependency {dependency} -> {need.id}: add {quantifier}{names} "
                        "to inspect.sources; deleting these calls breaks the dependency"
                    )
                elif not matches:
                    relation = f"relate({direction})" if direction else "relate"
                    errors.append(
                        f"dependency {dependency} -> {need.id} requires an inspected "
                        f"{relation} output carrying {need.id}, with a source carrying "
                        f"{dependency}; include it in inspect.sources"
                    )
        return errors

    @staticmethod
    def _validate_need_ids(values: list[str], known: set[str], owner: str) -> None:
        if any(value not in known for value in values):
            raise InvalidToolProgram(f"{owner} references an unknown evidence need")
        if len(values) != len(set(values)):
            raise InvalidToolProgram(f"{owner} evidence need ids must be unique")


class ToolSequenceAgent:
    def __init__(self, gateway: VLMGateway, validator: ToolProgramValidator | None = None) -> None:
        self.gateway = gateway
        self.validator = validator or ToolProgramValidator()

    async def compile(
        self,
        *,
        question: str,
        options: list[str] | None,
        plan: ReasoningPlan,
    ) -> ToolProgram:
        prompt = f"""
Compile this tool-agnostic evidence plan into typed locate/relate calls.

Question: {question}
Options: {options or "open-ended"}
Reasoning plan:
{plan.model_dump_json(indent=2)}

Functions:
- locate(id, query, limit, evidence_need_ids): semantic retrieval over the video's
  cached scene/event captions. Use it for independent anchor evidence.
- relate(id, source, relation, horizon_seconds, limit, evidence_need_ids): retrieve
  scenes before, after, during, or around one earlier locate/relate output. Use it for
  evidence needs with temporal dependencies.
- inspect(sources, evidence_need_ids): select the union to inspect visually. It must
  cover every evidence need and reference only locate/relate outputs.

Use one locate call for every distinct evidence need, then add relate calls to enforce
the plan's temporal dependencies. If there are more than four needs, combine closely
related needs in one locate query. Use l1..l4 and r1..r6 sequentially. Because cached
intervals are coarse, use a locate/relate limit of at least 2; never collapse an anchor
to one hypothesis. Vector retrieval may be redundant; the final inspection budget is
what prunes it. Set max_intervals to the smaller of the plan limit and the number of
evidence needs. Respect both evidence budgets. Return only the typed program. Do not
emit Python, compose-answer, or verify-answer calls. A relation follows the best
candidate in its source output. For a dependency chain, point each later relation to
the preceding relation output (for example, l1 -> r1 -> r2).
{TOOL_DATAFLOW_RULES}
""".strip()
        try:
            program = await self.gateway.complete_structured(
                role="typed locate-relate program compiler",
                prompt=prompt,
                response_model=ToolProgram,
                temperature=0.1,
                max_tokens=750,
            )
            # Provenance is assigned by the controller, never trusted from model JSON.
            program.compiler = "model"
            program.max_intervals = min(plan.max_intervals, len(plan.evidence_needs))
            program = self._ensure_recall_coverage(program, plan)
            return self.validator.validate(program, plan)
        except InvalidToolProgram as exc:
            logger.info("requesting locate/relate repair after validation error: %s", exc)
            repair_prompt = f"""
Repair this typed locate/relate program.

Validation error: {exc}
Reasoning plan:
{plan.model_dump_json(indent=2)}
Invalid program:
{program.model_dump_json(indent=2)}

Use sequential locate ids l1.. and relate ids r1... Every relate source must name an
earlier locate/relate output. Inspect must reference valid outputs and cover every
evidence need exactly as named in the reasoning plan. Return only the corrected typed
program.
{TOOL_DATAFLOW_RULES}
""".strip()
            try:
                repaired = await self.gateway.complete_structured(
                    role="typed locate-relate program repairer",
                    prompt=repair_prompt,
                    response_model=ToolProgram,
                    temperature=0.0,
                    max_tokens=750,
                )
                repaired.compiler = "model_repaired"
                repaired.max_intervals = min(plan.max_intervals, len(plan.evidence_needs))
                repaired = self._ensure_recall_coverage(repaired, plan)
                return self.validator.validate(repaired, plan)
            except Exception:
                logger.warning(
                    "model locate/relate repair failed; using safe fallback",
                    exc_info=True,
                )
                return self.validator.validate(self._fallback(plan), plan)
        except Exception:
            logger.warning(
                "model locate/relate compilation failed; using safe fallback",
                exc_info=True,
            )
            return self.validator.validate(self._fallback(plan), plan)

    @staticmethod
    def _ensure_recall_coverage(
        program: ToolProgram,
        plan: ReasoningPlan,
    ) -> ToolProgram:
        """Add cheap direct probes when the model over-compresses a dependency chain."""
        located = {need_id for call in program.locate for need_id in call.evidence_need_ids}
        missing = [need for need in plan.evidence_needs if need.id not in located]
        if not missing:
            return program

        added_ids: list[str] = []
        for need in missing:
            if len(program.locate) < 4:
                call_id = f"l{len(program.locate) + 1}"
                program.locate.append(
                    LocateCall(
                        id=call_id,
                        query=need.description,
                        limit=max(2, program.max_intervals),
                        evidence_need_ids=[need.id],
                    )
                )
                added_ids.append(call_id)
            else:
                call = program.locate[-1]
                call.query = f"{call.query}; {need.description}"
                call.evidence_need_ids.append(need.id)

        per_call_budget = max(2, 24 // len(program.locate))
        for call in program.locate:
            call.limit = min(call.limit, per_call_budget)
        program.inspect.sources = list(dict.fromkeys([*program.inspect.sources, *added_ids]))
        program.inspect.evidence_need_ids = [need.id for need in plan.evidence_needs]
        program.rationale = (
            f"{program.rationale} Controller added {len(missing)} direct recall "
            "probe(s) for uncovered evidence needs."
        )
        return program

    @staticmethod
    def _fallback(plan: ReasoningPlan) -> ToolProgram:
        locate: list[LocateCall] = []
        relate: list[RelateCall] = []
        output_by_need: dict[str, str] = {}
        need_groups = [[need] for need in plan.evidence_needs]
        if len(need_groups) > 4:
            need_groups = need_groups[:3] + [[need for group in need_groups[3:] for need in group]]
        per_locate_limit = min(
            plan.max_intervals * 2,
            max(2, 24 // len(need_groups)),
        )

        for group in need_groups:
            call_id = f"l{len(locate) + 1}"
            locate.append(
                LocateCall(
                    id=call_id,
                    query="; ".join(need.description for need in group),
                    limit=per_locate_limit,
                    evidence_need_ids=[need.id for need in group],
                )
            )
            for need in group:
                output_by_need[need.id] = call_id

        effective_output_by_need = dict(output_by_need)
        for need in plan.evidence_needs:
            sources = list(
                dict.fromkeys(
                    effective_output_by_need[item]
                    for item in need.depends_on
                    if item in effective_output_by_need
                )
            )
            for source in sources:
                if len(relate) >= 6:
                    raise InvalidToolProgram(
                        "the fallback requires more than six temporal relations; "
                        "simplify the reasoning plan"
                    )
                call_id = f"r{len(relate) + 1}"
                relation = {"before": "before", "after": "after", "outcome": "after"}.get(
                    need.temporal_role, "around"
                )
                relate.append(
                    RelateCall(
                        id=call_id,
                        source=source,
                        relation=relation,
                        horizon_seconds=max(20.0, min(plan.max_inspected_seconds, 120.0)),
                        limit=max(2, min(plan.max_intervals, len(plan.evidence_needs))),
                        evidence_need_ids=[need.id],
                    )
                )
                effective_output_by_need[need.id] = call_id

        sources = [call.id for call in locate] + [call.id for call in relate]
        return ToolProgram(
            compiler="fallback",
            rationale="Deterministic locate/relate fallback from evidence dependencies.",
            max_intervals=min(plan.max_intervals, len(plan.evidence_needs)),
            max_inspected_seconds=plan.max_inspected_seconds,
            locate=locate,
            relate=relate,
            inspect=InspectCall(
                sources=sources,
                evidence_need_ids=[need.id for need in plan.evidence_needs],
            ),
        )
