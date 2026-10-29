from __future__ import annotations

from pathlib import Path
from uuid import UUID

from videomasys.domain.schemas import (
    EvidenceAssessment,
    MultipleChoiceDecision,
    QAAnswer,
    QAPlan,
    ReasoningPlan,
    Verification,
)
from videomasys.model_gateway.vlm import VLMGateway


class ReasoningPlanner:
    """Decides what must be established without selecting concrete tools."""

    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def plan(
        self, question: str, duration_seconds: float, options: list[str] | None = None
    ) -> ReasoningPlan:
        prompt = f"""
Create an evidence plan for this question about a {duration_seconds:.1f}s video:
{question}
{format_options(options)}

Describe the objective, independent evidence needs, temporal dependencies, option
comparison strategy, and a concrete stopping condition. Stay tool-agnostic: do not
name APIs, searches, databases, clips, frames, or implementation operations. Evidence
needs must describe observable facts, not assume the correct answer. For causal or
purpose questions, separate the initiating anchor from the later observable outcome.
Use depends_on to express an ordered evidence graph. Keep the inspected-video budget
small; normally at most four intervals and 80 seconds.
""".strip()
        try:
            return await self.gateway.complete_structured(
                role="tool-agnostic video reasoning planner",
                prompt=prompt,
                response_model=ReasoningPlan,
                temperature=0.2,
                max_tokens=750,
            )
        except Exception:
            return ReasoningPlan(
                objective=question,
                evidence_needs=[
                    {
                        "id": "primary_evidence",
                        "description": question,
                        "temporal_role": "context",
                    }
                ],
                option_strategy="Compare every option against directly observed evidence."
                if options
                else None,
                max_intervals=4,
                max_inspected_seconds=min(max(duration_seconds, 20.0), 80.0),
                stop_condition="Enough direct evidence exists to answer or report insufficiency.",
                reasoning="Fallback evidence plan after a transient planner failure.",
            )


class QAPlanner:
    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def plan(
        self, question: str, duration_seconds: float, options: list[str] | None = None
    ) -> QAPlan:
        option_text = format_options(options)
        prompt = f"""
Create a compact retrieval plan for this question about a {duration_seconds:.1f}s video:
{question}
{option_text}

Produce 1–4 standalone semantic search queries. Expand pronouns into likely concepts
when possible, but do not assume an answer. target_entities should include things that
must be visually grounded. temporal_hint may describe ordering such as before/after.
For why, intent, cause, or purpose questions, cover both the initiating action and the
observable outcome afterward; at least one query must ask what happens next without
assuming which option is correct. For before/after questions, retrieve both anchor and
target events. max_intervals is the evidence budget, normally 3–5.
""".strip()
        try:
            return await self.gateway.complete_structured(
                role="QA planner",
                prompt=prompt,
            response_model=QAPlan,
            temperature=0.2,
            max_tokens=500,
            )
        except Exception:
            # A safe deterministic plan keeps transient planner failures recoverable.
            return QAPlan(
                search_queries=[question],
                target_entities=[],
                inspect_visual=True,
                max_intervals=4,
                reasoning="Fallback: retrieve directly with the normalized question.",
            )


class EvidenceInspector:
    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def inspect(
        self,
        *,
        question: str,
        options: list[str] | None,
        caption: str,
        start_seconds: float,
        end_seconds: float,
        frame_paths: list[Path],
        video_path: Path | None = None,
    ) -> EvidenceAssessment:
        option_text = format_options(options)
        prompt = f"""
Question: {question}
{option_text}
Candidate interval: {start_seconds:.3f}s–{end_seconds:.3f}s
Cached caption (retrieval hint only; it may be wrong): {caption}

Inspect the ordered frames or bounded proxy clip and record only observations relevant
to the question.
Do not defer to the cached caption when the frames disagree. For multiple-choice
questions, compare every option against discriminating visual details such as hand
placement, object mechanisms, and changes across frames. Mention uncertainty caused
by sparse frames. If irrelevant, set relevant=false and briefly state why. Do not
answer beyond this interval. Keep the observation under 120 words and never repeat a
sentence or phrase to add emphasis. Describe literal object transformations before
assigning labels: opening packaging, removing contents, assembling parts, and similar
steps may express an option's broader action even when its exact verb is not visible.
""".strip()
        return await self.gateway.complete_structured(
            role="visual evidence inspector",
            prompt=prompt,
            response_model=EvidenceAssessment,
            image_paths=[] if video_path else frame_paths,
            video_paths=[video_path] if video_path else [],
            temperature=0.2,
            max_tokens=350,
        )


class AnswerComposer:
    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def compose(
        self,
        question: str,
        evidence_text: str,
        options: list[str] | None = None,
        revision_feedback: list[str] | None = None,
        required_selected_option: int | None = None,
    ) -> QAAnswer:
        option_text = format_options(options)
        selection_instruction = (
            f"Use selected_option={required_selected_option}; it was fixed by the option "
            "judge. The answer text must name that exact option."
            if options and required_selected_option is not None
            else "Select exactly one option and return its zero-based index in selected_option."
            if options
            else "Set selected_option to null because this is open-ended QA."
        )
        revision_text = (
            "A verifier rejected a previous draft for these reasons:\n- "
            + "\n- ".join(revision_feedback)
            + "\nProduce a new answer that directly resolves every issue."
            if revision_feedback
            else ""
        )
        prompt = f"""
Question: {question}
{option_text}

Available evidence (node UUID and exact interval included):
{evidence_text}

Answer using only the evidence. Cite supporting node UUIDs and copy their interval
bounds exactly. If evidence is insufficient, say so directly and list the limitation.
Do not cite a node that is absent from the evidence. For multiple choice, compare
every option against every evidence interval. Eliminate any option explicitly negated
by an observation. For why, intent, cause, or purpose questions, later observable
outcomes are stronger evidence than the action at the anchor moment. The `answer` text
and `selected_option` index must refer to the same option. {selection_instruction}
{revision_text}
""".strip()
        return await self.gateway.complete_structured(
            role="evidence-grounded answer composer",
            prompt=prompt,
            response_model=QAAnswer,
            temperature=0.0,
            max_tokens=600,
        )


class MultipleChoiceJudge:
    """Forces an explicit evidence verdict for every choice before selection."""

    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def judge(
        self,
        question: str,
        evidence_text: str,
        options: list[str],
    ) -> MultipleChoiceDecision:
        prompt = f"""
Question: {question}
{format_options(options)}

Evidence in temporal order:
{evidence_text}

Assess every option exactly once in zero-based order as supported, contradicted, or
unknown. An option explicitly negated by evidence is contradicted. For why, intent,
cause, or purpose questions, use later observable outcomes to determine purpose.
An outcome can support the purpose even when it occurs later rather than during the
anchor interval. Select the single best-supported zero-based option and mark that
selected option supported, never contradicted. Do not infer an interaction with a
person or object that is absent from the evidence. Judge over the full temporal
sequence: an action beginning in a later interval supports that action as the purpose.
A statement that the action did not happen in an earlier interval does not contradict
direct evidence that it starts later. Beginning to open wrapping supports "unwrap";
completion is not required.
""".strip()
        decision = await self.gateway.complete_structured(
            role="multiple-choice evidence judge",
            prompt=prompt,
            response_model=MultipleChoiceDecision,
            temperature=0.0,
            max_tokens=700,
        )
        expected = set(range(len(options)))
        actual = {item.option_index for item in decision.assessments}
        if actual != expected or decision.selected_option not in expected:
            raise ValueError("option judge did not assess exactly the supplied choices")
        return decision


class AnswerVerifier:
    def __init__(self, gateway: VLMGateway) -> None:
        self.gateway = gateway

    async def verify(
        self,
        question: str,
        evidence_text: str,
        answer: QAAnswer,
        options: list[str] | None = None,
        choice_decision: MultipleChoiceDecision | None = None,
    ) -> Verification:
        option_text = format_options(options)
        selected_binding = (
            f"Index binding: selected_option={answer.selected_option} means exactly "
            f"{options[answer.selected_option]!r}. Never relabel this index."
            if options
            and answer.selected_option is not None
            and answer.selected_option < len(options)
            else ""
        )
        choice_context = (
            "Option-by-option evidence judgment:\n"
            + choice_decision.model_dump_json(indent=2)
            if choice_decision
            else ""
        )
        prompt = f"""
Question: {question}
{option_text}
Evidence:
{evidence_text}

Candidate answer:
{answer.model_dump_json(indent=2)}
{selected_binding}
{choice_context}

Check entailment, unsupported detail, temporal ordering, and citation validity. Accept
only if every material claim is supported. If rejection can be fixed without adding
facts, provide both a conservative corrected_answer and its zero-based
corrected_selected_option for multiple-choice questions. Never provide only one of
those fields. If another option is directly supported by the supplied evidence, a
rejection MUST include that option as the correction. For open-ended questions
corrected_selected_option must be null. Leave both correction fields null only when no
listed option is grounded. If accepted is false, issues must contain at least one
concrete reason; if there is no concrete issue, accepted must be true. The
option-by-option judgment is a separate agent's analysis, not ground truth. Check it
against the evidence, but account for its explicit supported/contradicted findings.
When that judgment marks the selected option supported and its reason is directly
grounded by cited observations, accept unless the citations or stated facts actually
conflict with it. Do not reject merely because another unlisted motivation is possible.

For a why, purpose, cause, or intent question, an observable action later in the event
sequence can support the purpose of an earlier setup action. Do not reject an answer
only because the outcome did not occur during the anchor interval itself. Evaluate the
full cited temporal sequence. Keep all issues together under 100 words.
""".strip()
        return await self.gateway.complete_structured(
            role="QA verifier",
            prompt=prompt,
            response_model=Verification,
            temperature=0.0,
            max_tokens=800,
        )


def valid_citation_ids(answer: QAAnswer, allowed_node_ids: set[UUID]) -> bool:
    return all(citation.node_id in allowed_node_ids for citation in answer.citations)


def format_options(options: list[str] | None) -> str:
    if not options:
        return ""
    return "Options (zero-based):\n" + "\n".join(
        f"[{index}] {option}" for index, option in enumerate(options)
    )
