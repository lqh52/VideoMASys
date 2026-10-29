from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from videomasys.agents.qa import (
    AnswerComposer,
    AnswerVerifier,
    EvidenceInspector,
    MultipleChoiceJudge,
    ReasoningPlanner,
    valid_citation_ids,
)
from videomasys.agents.tool_sequence import ToolSequenceAgent
from videomasys.domain.schemas import (
    EvidenceAssessment,
    QAAnswer,
    TaskStatus,
    ToolProgram,
)
from videomasys.media.ffmpeg import Interval, extract_proxy_clip, extract_representative_frames
from videomasys.retrieval.search import RetrievalHit, VideoRetriever
from videomasys.settings import Settings
from videomasys.storage.models import Artifact, Evidence, Task, TaskStep, Video


class TaskCancelled(RuntimeError):
    pass


class QAPipeline:
    def __init__(
        self,
        settings: Settings,
        planner: ReasoningPlanner,
        tool_sequencer: ToolSequenceAgent,
        retriever: VideoRetriever,
        inspector: EvidenceInspector,
        choice_judge: MultipleChoiceJudge,
        composer: AnswerComposer,
        verifier: AnswerVerifier,
    ) -> None:
        self.settings = settings
        self.planner = planner
        self.tool_sequencer = tool_sequencer
        self.retriever = retriever
        self.inspector = inspector
        self.choice_judge = choice_judge
        self.composer = composer
        self.verifier = verifier

    async def run(self, session: AsyncSession, task: Task, video: Video) -> None:
        question = str(task.request["question"])
        options = task.request.get("options")
        if options is not None:
            options = [str(option) for option in options]
        await self._check_cancelled(session, task)

        step = await self._start_step(session, task, "plan")
        reasoning_plan = await self.planner.plan(
            question, video.duration_seconds or 0.0, options
        )
        tool_program = await self.tool_sequencer.compile(
            question=question,
            options=options,
            plan=reasoning_plan,
        )
        task.plan = {
            "reasoning_plan": reasoning_plan.model_dump(mode="json"),
            "tool_program": tool_program.model_dump(mode="json"),
        }
        await self._finish_step(
            session,
            step,
            {
                "evidence_need_count": len(reasoning_plan.evidence_needs),
                "tool_call_count": len(tool_program.locate) + len(tool_program.relate) + 1,
            },
        )

        await self._check_cancelled(session, task)
        step = await self._start_step(session, task, "retrieve")
        candidate_count, hits, tool_trace = await self._execute_retrieval_program(
            session=session,
            video=video,
            program=tool_program,
        )
        await self._finish_step(
            session,
            step,
            {
                "candidate_count": candidate_count,
                "selected_interval_count": len(hits),
                "tool_trace": tool_trace,
            },
        )

        await self._check_cancelled(session, task)
        step = await self._start_step(session, task, "inspect_evidence")
        evidence_rows = await self._inspect_hits(session, task, video, question, options, hits)
        await self._finish_step(session, step, {"relevant_evidence_count": len(evidence_rows)})

        evidence_text = self._format_evidence(evidence_rows)
        await self._check_cancelled(session, task)
        step = await self._start_step(session, task, "compose_answer")
        choice_decision = (
            await self.choice_judge.judge(question, evidence_text, options)
            if options
            else None
        )
        required_option = choice_decision.selected_option if choice_decision else None
        answer = await self.composer.compose(
            question,
            evidence_text,
            options,
            required_selected_option=required_option,
        )
        if required_option is not None:
            answer.selected_option = required_option
        answer = self._normalize_citations(answer, evidence_rows)
        if options and (answer.selected_option is None or answer.selected_option >= len(options)):
            answer.selected_option = None
            answer.confidence = min(answer.confidence, 0.5)
            answer.limitations.append("The model did not return a valid option index.")
        elif options and answer.selected_option is not None:
            # The typed index is authoritative. Canonicalize before verification so a
            # model cannot pair a valid zero-based index with another option's label.
            answer.answer = canonical_multiple_choice_answer(answer.selected_option, options)
        await self._finish_step(
            session,
            step,
            {
                "citation_count": len(answer.citations),
                "option_decision": choice_decision.model_dump(mode="json")
                if choice_decision
                else None,
            },
        )

        await self._check_cancelled(session, task)
        step = await self._start_step(session, task, "verify_answer")
        verification = await self.verifier.verify(
            question,
            evidence_text,
            answer,
            options,
            choice_decision,
        )
        if not verification.accepted and not verification.issues:
            verification.issues.append(
                "Verifier rejected the candidate without an actionable reason; "
                "the grounded candidate was retained."
            )
        correction_is_complete = self._correction_is_complete(verification, options)
        if not verification.accepted and not correction_is_complete:
            revised_answer = await self.composer.compose(
                question,
                evidence_text,
                options,
                revision_feedback=verification.issues,
                required_selected_option=required_option,
            )
            if required_option is not None:
                revised_answer.selected_option = required_option
            revised_answer = self._normalize_citations(revised_answer, evidence_rows)
            if options and revised_answer.selected_option is not None:
                if revised_answer.selected_option < len(options):
                    revised_answer.answer = canonical_multiple_choice_answer(
                        revised_answer.selected_option,
                        options,
                    )
                else:
                    revised_answer.selected_option = None
            revised_verification = await self.verifier.verify(
                question,
                evidence_text,
                revised_answer,
                options,
                choice_decision,
            )
            if revised_verification.accepted or self._correction_is_complete(
                revised_verification, options
            ):
                answer = revised_answer
                verification = revised_verification
                correction_is_complete = self._correction_is_complete(
                    verification, options
                )
        final_answer = answer.answer
        final_selected_option = answer.selected_option
        correction_applied = False
        if not verification.accepted and correction_is_complete:
            final_answer = verification.corrected_answer
            if options:
                final_selected_option = verification.corrected_selected_option
            correction_applied = True
        if options and final_selected_option is not None:
            final_answer = canonical_multiple_choice_answer(final_selected_option, options)
        task.result = {
            "answer": final_answer,
            "selected_option": final_selected_option,
            "confidence": min(answer.confidence, verification.confidence),
            "citations": [citation.model_dump(mode="json") for citation in answer.citations],
            "limitations": answer.limitations,
            "verified": verification.accepted,
            "corrected_by_verifier": correction_applied,
            "verification_issues": verification.issues,
        }
        task.status = TaskStatus.SUCCEEDED.value
        task.error = None
        await self._finish_step(session, step, {"accepted": verification.accepted})
        await session.commit()

    @staticmethod
    def _correction_is_complete(
        verification: object,
        options: list[str] | None,
    ) -> bool:
        corrected_answer = getattr(verification, "corrected_answer", None)
        corrected_option = getattr(verification, "corrected_selected_option", None)
        return corrected_answer is not None and (
            not options
            or (corrected_option is not None and corrected_option < len(options))
        )

    async def _execute_retrieval_program(
        self,
        *,
        session: AsyncSession,
        video: Video,
        program: ToolProgram,
    ) -> tuple[int, list[RetrievalHit], list[dict[str, object]]]:
        outputs: dict[str, list[RetrievalHit]] = {}
        tool_trace: list[dict[str, object]] = []
        candidate_count = 0
        selected: list[RetrievalHit] = []
        for call in program.locate:
            raw_hits = await self.retriever.search(
                session,
                video_id=video.id,
                queries=[call.query],
                limit=call.limit,
            )
            candidate_count += len(raw_hits)
            outputs[call.id] = await self.retriever.expand_to_scenes(
                session,
                hits=raw_hits,
                question=call.query,
                limit=call.limit,
            )
            tool_trace.append(
                {
                    "call_id": call.id,
                    "function": "locate",
                    "output_count": len(outputs[call.id]),
                }
            )
        for call in program.relate:
            outputs[call.id] = await self.retriever.temporal_neighbors(
                session,
                hits=outputs.get(call.source, []),
                direction=call.relation,
                horizon_seconds=call.horizon_seconds,
                limit=call.limit,
            )
            candidate_count += len(outputs[call.id])
            tool_trace.append(
                {
                    "call_id": call.id,
                    "function": "relate",
                    "source": call.source,
                    "relation": call.relation,
                    "output_count": len(outputs[call.id]),
                }
            )
        candidates = self._merge_hit_outputs(
            outputs.get(source, []) for source in program.inspect.sources
        )
        selected = self._apply_evidence_budget(candidates, program)
        tool_trace.append(
            {
                "function": "inspect",
                "sources": program.inspect.sources,
                "output_count": len(selected),
            }
        )
        return candidate_count, selected, tool_trace

    @staticmethod
    def _merge_hit_outputs(
        groups: Iterable[list[RetrievalHit]],
    ) -> list[RetrievalHit]:
        by_interval: dict[tuple[float, float], RetrievalHit] = {}
        for group in groups:
            for hit in group:
                key = (hit.node.start_seconds, hit.node.end_seconds)
                previous = by_interval.get(key)
                if previous is None or hit.score > previous.score:
                    by_interval[key] = hit
        return sorted(by_interval.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _apply_evidence_budget(
        candidates: list[RetrievalHit], program: ToolProgram
    ) -> list[RetrievalHit]:
        selected: list[RetrievalHit] = []
        inspected_seconds = 0.0
        for hit in candidates:
            duration = max(0.0, hit.node.end_seconds - hit.node.start_seconds)
            if selected and inspected_seconds + duration > program.max_inspected_seconds:
                continue
            selected.append(hit)
            inspected_seconds += duration
            if len(selected) >= program.max_intervals:
                break
        return sorted(selected, key=lambda item: item.node.start_seconds)

    async def _inspect_hits(
        self,
        session: AsyncSession,
        task: Task,
        video: Video,
        question: str,
        options: list[str] | None,
        hits: list[RetrievalHit],
    ) -> list[Evidence]:
        # AsyncSession is not safe for concurrent queries, so paths are read serially;
        # only the expensive, stateless VLM calls are batched concurrently.
        prepared = []
        for hit in hits:
            paths = await self.retriever.frame_paths(session, hit.node.id)
            evidence_root = (
                self.settings.artifact_root
                / "cache"
                / str(video.id)
                / "qa_evidence"
                / str(hit.node.id)
            )
            if len(paths) < self.settings.qa_frames_per_interval:
                paths = await extract_representative_frames(
                    Path(video.source_path),
                    Interval(hit.node.start_seconds, hit.node.end_seconds),
                    evidence_root,
                    self.settings.qa_frames_per_interval,
                )
                for index, path in enumerate(paths):
                    session.add(
                        Artifact(
                            video_id=video.id,
                            node_id=hit.node.id,
                            kind="qa_evidence_frame",
                            path=str(path.resolve()),
                            artifact_metadata={"ordinal": index},
                        )
                    )
                await session.commit()
            proxy_path: Path | None = None
            if self.settings.qa_use_video_clip:
                proxy_path = evidence_root / "motion_proxy.mp4"
                if not proxy_path.is_file():
                    await extract_proxy_clip(
                        Path(video.source_path),
                        Interval(hit.node.start_seconds, hit.node.end_seconds),
                        proxy_path,
                        self.settings.qa_clip_fps,
                    )
            prepared.append((hit, paths, proxy_path))

        async def inspect_one(
            hit: RetrievalHit, paths: list[Path], proxy_path: Path | None
        ) -> tuple[RetrievalHit, list[str], EvidenceAssessment]:
            assessment = await self.inspector.inspect(
                question=question,
                options=options,
                caption=hit.node.caption,
                start_seconds=hit.node.start_seconds,
                end_seconds=hit.node.end_seconds,
                frame_paths=paths,
                video_path=proxy_path,
            )
            assessment.observation = compact_observation(assessment.observation)
            return hit, [str(path.resolve()) for path in paths], assessment

        results = await asyncio.gather(
            *(inspect_one(hit, paths, proxy_path) for hit, paths, proxy_path in prepared)
        )
        evidence_rows: list[Evidence] = []
        for hit, paths, assessment in results:
            if not assessment.relevant:
                continue
            evidence = Evidence(
                task_id=task.id,
                node_id=hit.node.id,
                start_seconds=hit.node.start_seconds,
                end_seconds=hit.node.end_seconds,
                observation=assessment.observation,
                confidence=assessment.confidence,
                frame_paths=paths,
            )
            session.add(evidence)
            evidence_rows.append(evidence)
        await session.commit()
        return evidence_rows

    @staticmethod
    def _format_evidence(rows: list[Evidence]) -> str:
        if not rows:
            return "No relevant visual evidence was retrieved."
        return "\n".join(
            f"- node={row.node_id} interval={row.start_seconds:.3f}–{row.end_seconds:.3f}s "
            f"confidence={row.confidence:.2f}: {row.observation}"
            for row in rows
        )

    @staticmethod
    def _normalize_citations(answer: QAAnswer, rows: list[Evidence]) -> QAAnswer:
        by_id = {row.node_id: row for row in rows}
        if not valid_citation_ids(answer, set(by_id)):
            answer.citations = [item for item in answer.citations if item.node_id in by_id]
            answer.confidence = min(answer.confidence, 0.5)
            answer.limitations.append("One or more invalid model citations were removed.")
        for citation in answer.citations:
            source = by_id[citation.node_id]
            citation.start_seconds = source.start_seconds
            citation.end_seconds = source.end_seconds
        return answer
    @staticmethod
    async def _start_step(session: AsyncSession, task: Task, name: str) -> TaskStep:
        step = TaskStep(
            task_id=task.id,
            name=name,
            status="running",
            details={"monotonic_start": time.monotonic()},
        )
        session.add(step)
        await session.commit()
        return step

    @staticmethod
    async def _finish_step(
        session: AsyncSession, step: TaskStep, details: dict[str, object]
    ) -> None:
        start = float(step.details.pop("monotonic_start", time.monotonic()))
        step.status = "succeeded"
        step.finished_at = datetime.now(UTC)
        step.latency_ms = round((time.monotonic() - start) * 1000)
        step.details = details
        await session.commit()

    @staticmethod
    async def _check_cancelled(session: AsyncSession, task: Task) -> None:
        await session.refresh(task, attribute_names=["status"])
        if task.status == TaskStatus.CANCEL_REQUESTED.value:
            task.status = TaskStatus.CANCELLED.value
            await session.commit()
            raise TaskCancelled("task cancellation requested")


def compact_observation(value: str, max_characters: int = 800) -> str:
    """Remove decoder repetition and bound evidence before later agent calls."""
    normalized = " ".join(value.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        key = sentence.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(sentence.strip())
    compacted = " ".join(unique)
    if len(compacted) <= max_characters:
        return compacted
    truncated = compacted[: max_characters - 1].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{truncated}…"


def canonical_multiple_choice_answer(selected: int, options: list[str]) -> str:
    """Render one unambiguous answer from the validated zero-based option index."""
    return f"[{selected}] {options[selected]}"
