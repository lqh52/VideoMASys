from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VideoStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class NodeKind(StrEnum):
    VIDEO = "video"
    EVENT = "event"
    SCENE = "scene"


class VideoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_hash: str
    status: VideoStatus
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    node_count: int = 0


class QATaskCreate(BaseModel):
    video_id: UUID
    question: str = Field(min_length=2, max_length=2000)
    options: list[str] | None = Field(default=None, min_length=2, max_length=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("options")
    @classmethod
    def normalize_options(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value for value in cleaned):
            raise ValueError("options cannot be empty")
        return cleaned


class EvidenceRead(BaseModel):
    node_id: UUID
    start_seconds: float
    end_seconds: float
    observation: str
    confidence: float
    frame_paths: list[str] = Field(default_factory=list)


class TaskStepRead(BaseModel):
    name: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TaskRead(BaseModel):
    id: UUID
    video_id: UUID
    task_type: str
    status: TaskStatus
    request: dict[str, Any]
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    evidence: list[EvidenceRead] = Field(default_factory=list)
    steps: list[TaskStepRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class QAPlan(BaseModel):
    """Declarative plan produced by the planner and consumed by fixed operators."""

    search_queries: list[str] = Field(min_length=1, max_length=4)
    target_entities: list[str] = Field(default_factory=list, max_length=12)
    temporal_hint: str | None = Field(default=None, max_length=300)
    inspect_visual: bool = True
    max_intervals: int = Field(default=4, ge=1, le=8)
    reasoning: str = Field(max_length=800)

    @field_validator("search_queries")
    @classmethod
    def clean_queries(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty search query is required")
        return list(dict.fromkeys(cleaned))


class EvidenceNeed(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    description: str = Field(min_length=3, max_length=500)
    temporal_role: Literal["anchor", "before", "after", "outcome", "context"]
    depends_on: list[str] = Field(default_factory=list, max_length=4)


class ReasoningPlan(BaseModel):
    """Tool-agnostic statement of which facts are needed and how they relate."""

    objective: str = Field(min_length=3, max_length=500)
    evidence_needs: list[EvidenceNeed] = Field(min_length=1, max_length=6)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=6)
    option_strategy: str | None = Field(default=None, max_length=500)
    max_intervals: int = Field(default=4, ge=1, le=8)
    max_inspected_seconds: float = Field(default=80.0, ge=5.0, le=240.0)
    stop_condition: str = Field(min_length=3, max_length=500)
    reasoning: str = Field(min_length=3, max_length=1000)

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> ReasoningPlan:
        ids = [need.id for need in self.evidence_needs]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence need ids must be unique")
        known: set[str] = set()
        for need in self.evidence_needs:
            if any(dependency not in known for dependency in need.depends_on):
                raise ValueError("evidence needs may depend only on earlier needs")
            known.add(need.id)
        return self


class LocateCall(BaseModel):
    """Typed equivalent of segment_retrieval.locate(query=..., limit=...)."""

    id: str = Field(pattern=r"^l[1-4]$")
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=4, ge=2, le=8)
    evidence_need_ids: list[str] = Field(min_length=1, max_length=6)
    temporal_position: Literal["any", "beginning", "middle", "end"] = "any"


class RelateCall(BaseModel):
    """Typed temporal traversal from a previous locate/relate result."""

    id: str = Field(pattern=r"^r[1-6]$")
    source: str = Field(pattern=r"^[lr][1-9]$")
    relation: Literal["before", "after", "during", "around"]
    horizon_seconds: float = Field(default=60.0, ge=20.0, le=180.0)
    limit: int = Field(default=4, ge=2, le=8)
    evidence_need_ids: list[str] = Field(min_length=1, max_length=6)


class InspectCall(BaseModel):
    """Materialize a bounded union of retrieval results for visual inspection."""

    sources: list[str] = Field(min_length=1, max_length=10)
    evidence_need_ids: list[str] = Field(min_length=1, max_length=6)


class ToolProgram(BaseModel):
    version: Literal["2.0"] = "2.0"
    compiler: Literal["model", "model_repaired", "fallback"] = "model"
    rationale: str = Field(min_length=3, max_length=800)
    max_intervals: int = Field(default=4, ge=1, le=8)
    max_inspected_seconds: float = Field(default=80.0, ge=5.0, le=240.0)
    locate: list[LocateCall] = Field(min_length=1, max_length=4)
    relate: list[RelateCall] = Field(default_factory=list, max_length=6)
    inspect: InspectCall


class SceneAnalysis(BaseModel):
    caption: str = Field(min_length=1, max_length=1200)
    visible_entities: list[str] = Field(default_factory=list, max_length=20)
    actions: list[str] = Field(default_factory=list, max_length=20)
    searchable_tags: list[str] = Field(default_factory=list, max_length=20)
    boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceAssessment(BaseModel):
    relevant: bool
    observation: str = Field(min_length=1, max_length=800)
    confidence: float = Field(ge=0.0, le=1.0)


class OptionAssessment(BaseModel):
    option_index: int = Field(ge=0, le=9)
    verdict: Literal["supported", "contradicted", "unknown"]
    reason: str = Field(min_length=1, max_length=500)


class MultipleChoiceDecision(BaseModel):
    assessments: list[OptionAssessment] = Field(min_length=2, max_length=10)
    selected_option: int = Field(ge=0, le=9)
    reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_assessments(self) -> MultipleChoiceDecision:
        indices = [assessment.option_index for assessment in self.assessments]
        if len(indices) != len(set(indices)):
            raise ValueError("each option may be assessed only once")
        if self.selected_option not in indices:
            raise ValueError("selected option must have an assessment")
        selected = next(
            item for item in self.assessments if item.option_index == self.selected_option
        )
        if selected.verdict == "contradicted":
            raise ValueError("selected option cannot be contradicted")
        return self


class Citation(BaseModel):
    node_id: UUID
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)


class QAAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    selected_option: int | None = Field(default=None, ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=8)


class Verification(BaseModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list, max_length=12)
    corrected_answer: str | None = Field(default=None, max_length=4000)
    corrected_selected_option: int | None = Field(default=None, ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_decision(self) -> Verification:
        if not self.accepted and not self.issues and self.corrected_answer is None:
            raise ValueError("a rejected answer requires a specific issue or correction")
        if self.corrected_selected_option is not None and self.corrected_answer is None:
            raise ValueError("an option correction requires a corrected answer")
        return self
