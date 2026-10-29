"""Real, bounded QA stages; no GPU, Gradio, database server, or network at import."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

from videomasys.agents.tool_sequence import ToolProgramValidator
from videomasys.domain.schemas import ReasoningPlan, ToolProgram
from videomasys.zerogpu import clean_prompts, structured_prompts, svp_prompts, tuned_prompts
from videomasys.zerogpu.media_sampling import refine_scene_frames, sample_scene_frames
from videomasys.zerogpu.prompts import (
    EVIDENCE_CALIBRATION_RULES,
    OPEN_ANSWER_RULES,
    OPEN_VERIFIER_RULES,
    planner_prompt,
    tool_prompt,
)
from videomasys.zerogpu.prompts import (
    PROMPT_VERSION as LEGACY_PROMPT_VERSION,
)
from videomasys.zerogpu.retrieval import execute_program
from videomasys.zerogpu.svp_retrieval import (
    SVPPlan,
    SVPProgram,
    execute_svp_program,
    validate_svp_task_program,
)

CACHE_VERSION = "zerogpu-scenes-v3-actions-endpoints-rp108"
MAX_VIDEO_SECONDS = 180
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_SCENES = 12


class Scene(BaseModel):
    index: int
    start: float
    end: float
    frames: list[str]
    frame_times: list[float] = Field(default_factory=list)
    caption: str = ""


class Caption(BaseModel):
    caption: str = Field(min_length=3, max_length=600)


class Observation(BaseModel):
    scene: int = Field(ge=0)
    observation: str = Field(min_length=3, max_length=600)
    relevant: bool
    uncertainty: str = Field(default="", max_length=240)
    refine_at: float | None = Field(
        default=None,
        description="Time of a brief/unclear key interaction worth closer sampling; otherwise null",
    )


class Inspection(BaseModel):
    evidence: list[Observation] = Field(min_length=1, max_length=3)


class QAPlan(ReasoningPlan):
    scope: Literal["local_event", "whole_video"] = Field(
        description="Whole-video objectives need representative temporal coverage, not one event."
    )

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope == "whole_video" and (
            len(self.evidence_needs) != 3
            or any(n.depends_on or n.temporal_role != "context" for n in self.evidence_needs)
        ):
            raise ValueError(
                "whole_video requires three independent context needs, in early/middle/late "
                "order; no dependencies or predicted action sequence."
            )
        return self


class ChoiceComparison(BaseModel):
    choice: str
    status: Literal["best_supported", "compatible", "contradicted", "unclear"]
    reason: str = Field(min_length=3, max_length=220)


class EvidenceAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=1200)
    supporting_scenes: list[int] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=4)
    inference: str = Field(default="", max_length=500)


class AnswerText(EvidenceAnswer):
    option_comparisons: list[ChoiceComparison] = Field(default_factory=list, max_length=5)


class Answer(AnswerText):
    selected_option: int | None = Field(default=None, ge=0)
    selected_choice: str | None = None


class Verdict(BaseModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list, max_length=5)
    needs_revision: bool = Field(
        default=False,
        description="Request one correction using existing evidence; false for accepted drafts",
    )


def checked_command(args: list[str], timeout: int = 40) -> bytes:
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            "Video processing timed out. Try a shorter, lower-resolution clip."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Cannot decode this video. Please use a standard MP4 or WebM file."
        ) from exc
    return result.stdout


def prepare_video(path: Path, cache_root: Path) -> dict:
    if path.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        raise ValueError("Upload an MP4, WebM, MOV, MKV, or AVI video.")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise ValueError("The demo accepts nonempty videos up to 100 MiB.")
    metadata = json.loads(
        checked_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), {})
    duration = float(metadata.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or not 0 < duration <= MAX_VIDEO_SECONDS or not stream:
        raise ValueError("The demo accepts videos up to 3 minutes with a valid video stream.")
    if max(stream.get("width", 0), stream.get("height", 0)) > 3840:
        raise ValueError("Please resize the video to 4K or smaller before uploading.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    key = digest.hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    directory = cache_root / key
    if not directory.exists() and sum(p.is_dir() for p in cache_root.iterdir()) >= 64:
        raise ValueError(
            "The demo cache is full. The owner needs to clear it or restart the Space."
        )
    directory.mkdir(parents=True, exist_ok=True)
    # Keep >=1s windows for very short clips; otherwise prepare at least three
    # representatives for later whole-video questions. Old 1/2-window indexes
    # miss Cache.load's length check; unchanged longer-video caches are reusable.
    # Sub-3s clips still support local questions; global coverage fails explicitly.
    count = min(MAX_SCENES, max(3 if duration >= 3 else 1, math.ceil(duration / 20)))
    scenes = []
    for i in range(count):
        start, end = duration * i / count, duration * (i + 1) / count
        scene = Scene(index=i, start=start, end=end, frames=[]).model_dump()
        scenes.append(sample_scene_frames(path, directory, scene, count=4))
    return {
        "key": key,
        "duration": duration,
        "scenes": scenes,
        "database": str(directory / "index.sqlite"),
        "video_path": str(path.resolve()),
        "frame_directory": str(directory.resolve()),
        "trace": [],
        "cached": False,
    }


class SceneCache:
    """Small on-disk vector index for a demo, not a replacement for server pgvector."""

    def __init__(self, path: str, signature: str):
        self.path, self.signature = path, signature

    def load(self, scenes: list[dict]) -> tuple[list[dict], np.ndarray] | None:
        if not Path(self.path).is_file():
            return None
        try:
            with sqlite3.connect(self.path) as db:
                row = db.execute("SELECT signature, captions, vectors FROM scene_cache").fetchone()
            if not row or row[0] != self.signature:
                return None
            captions = json.loads(row[1])
            vectors = np.asarray(json.loads(row[2]), dtype=np.float32)
            if len(captions) != len(scenes) or vectors.ndim != 2 or len(vectors) != len(scenes):
                return None
            if not np.isfinite(vectors).all():
                return None
            return [dict(s, caption=c) for s, c in zip(scenes, captions, strict=True)], vectors
        except (sqlite3.Error, ValueError, TypeError):
            return None

    def save(self, scenes: list[dict], vectors: np.ndarray) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS scene_cache "
                "(signature TEXT, captions TEXT, vectors TEXT)"
            )
            db.execute("DELETE FROM scene_cache")
            db.execute(
                "INSERT INTO scene_cache VALUES (?, ?, ?)",
                (
                    self.signature,
                    json.dumps([s["caption"] for s in scenes]),
                    json.dumps(vectors.tolist()),
                ),
            )


def tool_search_context(
    context: str, scenes: list[dict], scope: str, *, caption_limit: int = 240,
) -> str:
    """Expose fallible retrieval hints to the compiler, never to the blind planner."""
    overview = [
        {
            "scene": scene["index"],
            "span": [round(scene["start"], 1), round(scene["end"], 1)],
            "search_hint": scene["caption"][:caption_limit],
        }
        for scene in scenes
    ]
    if scope == "whole_video":
        duration = max((scene["end"] for scene in scenes), default=0.0)
        positions = ("beginning", "middle", "end")
        groups = {
            position: {"bounds_seconds": [duration * i / 3, duration * (i + 1) / 3], "scenes": []}
            for i, position in enumerate(positions)
        }
        for scene, hint in zip(scenes, overview, strict=True):
            midpoint = (scene["start"] + scene["end"]) / 2
            position = (
                "beginning"
                if midpoint < duration / 3
                else "middle"
                if midpoint < 2 * duration / 3
                else "end"
            )
            groups[position]["scenes"].append(hint)
        timeline = {"duration_seconds": duration, "groups_by_window_midpoint": groups}
    else:
        timeline = overview
    return context + "\nFallible cached timeline, not verified evidence: " + json.dumps(timeline)


class LivePipeline:
    def __init__(
        self,
        generate,
        embed,
        model_id: str,
        embedding_id: str,
        *,
        generate_structured=None,
        prompt_profile="clean",
    ):
        if not isinstance(prompt_profile, str) or prompt_profile not in (
            "clean", "tuned", "legacy", "svp", "structured"
        ):
            raise ValueError("prompt_profile must be clean, tuned, legacy, svp or structured.")
        self.prompt_profile = prompt_profile
        self.role_prompts = {
            "clean": clean_prompts, "legacy": clean_prompts,
            "tuned": tuned_prompts, "svp": svp_prompts,
            "structured": structured_prompts,
        }[prompt_profile]
        self.prompt_version = (
            self.role_prompts.PROMPT_VERSION
            if prompt_profile != "legacy" else LEGACY_PROMPT_VERSION
        )
        self.generate, self.embed = generate, embed
        self.generate_structured = generate_structured
        self.signature = f"{CACHE_VERSION}:{model_id}:{embedding_id}"
        if prompt_profile != "legacy":
            self.signature += f":prompts={self.prompt_version}"
        self.model_id, self.embedding_id = model_id, embedding_id

    def structured(
        self,
        trace,
        stage,
        prompt,
        schema,
        images=(),
        tokens=500,
        validate=None,
        prompt_examples=None,
        response_schema=None,
        repair_diagnostics=None,
    ):
        """At most one explicit repair; raw responses and errors are never concealed."""
        response_schema = deepcopy(response_schema or schema.model_json_schema())
        contract = json.dumps(response_schema, separators=(",", ":"))
        structured_format = (
            "\nReturn one JSON object matching the supplied response schema. "
            "No markdown; keep text fields concise."
        )
        if self.generate_structured is not None:
            request = prompt + structured_format
        else:
            request = (
                prompt + "\nReturn one JSON object only, no markdown. Keep text fields concise."
                "\nThe required JSON schema is:\n" + contract
            )
        entry = {"stage": stage, "status": "running", "model": self.model_id, "attempts": []}
        entry.update(prompt_profile=self.prompt_profile, prompt_version=self.prompt_version)
        if prompt_examples is not None:
            entry["prompt_examples"] = prompt_examples
        trace.append(entry)
        start = time.monotonic()
        for _attempt in range(2):
            record = {
                "prompt": request,
                "images": [Path(p).name for p in images],
                "max_new_tokens": tokens,
            }
            if self.generate_structured is not None:
                record["response_schema"] = deepcopy(response_schema)
            entry["attempts"].append(record)
            try:
                if self.generate_structured is not None:
                    # Providers normalize their own supported schema subset. Keep
                    # the source contract and trace independent of adapter mutations.
                    raw = self.generate_structured(
                        request, images, tokens, deepcopy(response_schema)
                    )
                else:
                    raw = self.generate(request, images, tokens)
                record["raw_response"] = raw
                candidate = raw.strip()
                if candidate.startswith("```"):
                    candidate = "\n".join(candidate.splitlines()[1:-1])
                result = schema.model_validate_json(candidate)
                if validate:
                    validated = validate(result)
                    if isinstance(validated, schema):
                        result = validated
                if self.prompt_profile == "structured":
                    parsed = json.loads(candidate)
                    canonical = result.model_dump(mode="json")
                    if parsed != canonical:
                        record["canonicalization"] = {
                            "operation": "schema_validation_and_exact_reference_normalization",
                            "changed_fields": sorted(
                                key for key in set(parsed) | set(canonical)
                                if parsed.get(key) != canonical.get(key)
                            ),
                            "raw_response_preserved": True,
                        }
                entry.update(
                    status="complete",
                    output=result.model_dump(),
                    latency_ms=round((time.monotonic() - start) * 1000),
                )
                return result
            except (ValueError, TypeError) as exc:
                record["validation_error"] = str(exc)
                repair_error = str(exc)
                if self.generate_structured is not None and isinstance(exc, ValidationError):
                    # The full diagnostic remains in the trace. Do not echo a
                    # malformed model response through Pydantic's input excerpt.
                    repair_error = json.dumps(
                        exc.errors(include_input=False, include_url=False, include_context=False),
                        separators=(",", ":"),
                    )
                previous = record.get("raw_response", "")
                try:
                    parsed_previous = json.loads(previous)
                    if repair_diagnostics is not None:
                        # Context-dependent checks (e.g. a locked answer choice)
                        # may not be reproducible from the JSON alone. Preserve
                        # their actual failure alongside generic graph diagnostics.
                        diagnostics = {**repair_diagnostics(parsed_previous),
                                       "validation_error": repair_error}
                        record["repair_diagnostics"] = diagnostics
                        repair_error = json.dumps(diagnostics, separators=(",", ":"))
                    if self.prompt_profile == "structured":
                        # Preserve valid JSON for the Tool agent as well as Answer.
                        # Slicing a complete program can hide its final steps and
                        # inspect sources from the one allowed repair.
                        compact_previous = json.dumps(parsed_previous, separators=(",", ":"))
                        feedback = (
                            "\nYour previous invalid response: " + compact_previous
                            if len(compact_previous) <= 16000 else
                            "\nThe previous JSON exceeded the repair-text allowance and is "
                            "retained in the trace. Produce a shorter complete object from "
                            "the original context and validation feedback; no partial draft "
                            "has been supplied."
                        )
                    else:
                        feedback = "\nYour previous invalid response: " + previous[:3000]
                except (ValueError, TypeError):
                    # Keep the entire raw failure in the trace, but do not prime
                    # the repair with a truncated/repetitive generation loop.
                    feedback = (
                        "\nThe previous output was not valid JSON. Start fresh; describe each "
                        "action only once and use at most two sentences per text field."
                    )
                request = (
                    prompt
                    + (
                        structured_format
                        if self.generate_structured is not None
                        else "\nRequired schema: " + contract
                    )
                    + feedback
                    + "\nValidation error to fix: "
                    + repair_error[:8000 if repair_diagnostics is not None else 1000]
                    + (
                        "\nRepair only the reported defects; preserve valid claims, citations "
                        "and selection unless the evidence requires changing them. Update "
                        "answer_claim_ids if the wrong roots caused the error: conclusions "
                        "can replace their inherited premises, not just be added to full slots. "
                        "Planner coverage notes are advisory, not repair obligations. "
                        "Do not invent references or attach unrelated claims to satisfy coverage. "
                        "Return the complete corrected JSON object, with no extra explanation."
                        if repair_diagnostics is not None else
                        "\nReturn a complete corrected JSON object. Correct the field types; "
                        "do not repeat the invalid structure. Return no explanation."
                    )
                )
            except Exception as exc:
                record["error"] = str(exc)
                entry.update(status="failed", latency_ms=round((time.monotonic() - start) * 1000))
                raise
        entry.update(status="failed", latency_ms=round((time.monotonic() - start) * 1000))
        raise ValueError(f"{stage} failed validation after two attempts. See the real trace.")

    def index(self, bundle):
        cache = SceneCache(bundle["database"], self.signature)
        previous = cache.load(bundle["scenes"])
        if previous:
            bundle["scenes"], _ = previous
            bundle["cached"] = True
            return bundle
        for scene in bundle["scenes"]:
            result = self.structured(
                bundle["trace"],
                f"Caption scene {scene['index']}",
                self.role_prompts.caption_prompt(scene)
                if self.prompt_profile != "legacy"
                else (
                    "Create a searchable action caption for these chronological frames. Prioritize "
                    "who does what to which object: hand-object contact, movement between places, "
                    "and object state changes from first to last frame. "
                    "Mention a brief interaction "
                    "even if visible in only one frame. Describe actions, not room decoration. "
                    "Do not invent intent or actions in unseen gaps. "
                    "Distinguish holding, carrying, "
                    "manipulating, and releasing an object. Use at most 60 words. "
                    f"Window: {scene['start']:.1f}–{scene['end']:.1f}s; "
                    f"sample times: {scene.get('frame_times', [])}."
                ),
                Caption,
                scene["frames"],
                tokens=200,
            )
            scene["caption"] = result.caption
        return bundle

    def save_index(self, bundle):
        vectors = self.embed([s["caption"] for s in bundle["scenes"]])
        SceneCache(bundle["database"], self.signature).save(bundle["scenes"], vectors)
        return bundle

    def inspect_windows(
        self, trace, selected, question, needs, *, video_path=None, directory=None,
        needs_by_scene=None,
    ):
        """Isolate visual attention per window; retain each real call in one UI stage."""
        entry = {
            "stage": "Evidence inspection",
            "prompt_profile": self.prompt_profile,
            "prompt_version": self.prompt_version,
            "status": "running",
            "model": self.model_id,
            "attempts": [],
            "calls": [],
            "repair_count": 0,
            "execution": "One model call per selected window; controller aggregates observations.",
        }
        trace.append(entry)
        started = time.monotonic()
        observations = []
        try:
            for scene in selected:
                scene_needs = (
                    needs_by_scene.get(scene["index"], needs)
                    if needs_by_scene is not None else needs
                )
                schema = create_model(
                    f"ObservationScene{scene['index']}",
                    __base__=Observation,
                    scene=(
                        Literal[scene["index"]],
                        Field(description="Exact scene ID, not image number"),
                    ),
                )
                local_trace = []
                try:
                    value = self.structured(
                        local_trace,
                        f"Inspect scene {scene['index']}",
                        self.role_prompts.inspect_prompt(scene, question, scene_needs)
                        if self.prompt_profile != "legacy"
                        else (
                            f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
                            f"These {len(scene['frames'])} chronological images are ALL from scene "
                            f"{scene['index']}, spanning {scene['start']:.2f}–{scene['end']:.2f}s. "
                            "Approximate sample times: "
                            f"{[round(t, 2) for t in scene.get('frame_times', [])]}. "
                            "Describe the visible action sequence in this window only. "
                            "Follow hands, tools and objects from the first image to the last. "
                            "Distinguish contact with one object from merely being near it. "
                            "Report visible state changes "
                            "and movements; do not complete an action from common knowledge. "
                            "Seeing a tool near a container does not by itself show "
                            "dipping it inside. "
                            "Seeing packaging near someone does not by itself show opening it. "
                            "The question may ask about an event outside this window: never invent "
                            "that event to fit the question. Sparse gaps cannot prove absence. "
                            "Do not answer WHY or choose an answer. "
                            "State unclear details in uncertainty. "
                            "If the answer depends on a brief hand/tool/object interaction seen in "
                            "only one sample, set refine_at to that sample's time "
                            "for a closer look. "
                            "Otherwise use null. Do not request refinement for irrelevant setup. "
                            "Observation: at most 80 words; uncertainty: at most 20 words."
                        ),
                        schema,
                        scene["frames"],
                        tokens=300,
                        validate=lambda v, scene=scene: self.validate_focus(v, scene),
                    )
                    observations.append(Observation.model_validate(value.model_dump()))
                finally:
                    for call in local_trace:
                        call["scene"] = scene["index"]
                        if needs_by_scene is not None:
                            call["evidence_sought"] = scene_needs
                        entry["calls"].append(call)
                        entry["repair_count"] += max(0, len(call["attempts"]) - 1)
                        entry["attempts"].extend(
                            {**attempt, "scene": scene["index"], "model_call": len(entry["calls"])}
                            for attempt in call["attempts"]
                        )
            requested = next(
                (
                    (i, value)
                    for i, value in enumerate(observations)
                    if value.relevant and value.refine_at is not None
                ),
                None,
            )
            if requested and video_path and directory:
                position, previous = requested
                original = selected[position]
                detail = refine_scene_frames(
                    Path(video_path), Path(directory), original, previous.refine_at, count=12
                )
                detail_trace = []
                try:
                    refined = self.structured(
                        detail_trace,
                        f"Refine scene {original['index']}",
                        self.role_prompts.refine_prompt(
                            detail, question,
                            needs_by_scene.get(original["index"], needs)
                            if needs_by_scene is not None else needs,
                        )
                        if self.prompt_profile != "legacy"
                        else (
                            f"Question: {question}\nEvidence sought: {needs}\n"
                            f"These twelve chronological frames cover scene {original['index']}, "
                            f"within {detail['detail_span']}. Approximate sample times: "
                            f"{[round(t, 2) for t in detail['frame_times']]}. "
                            "Follow the visible contact/motion across the sequence, including what "
                            "the hand/tool was doing immediately BEFORE contact and immediately "
                            "AFTER. Name the observable action when the sequence supports it, not "
                            "just 'handling'. Distinguish actual changes from plausible "
                            "everyday use. Describe each action once, in at most 80 words. "
                            "Explain any unresolved physical distinction in uncertainty. "
                            "No further sampling is available; "
                            "set refine_at=null. Do not select an answer."
                        ),
                        create_model(
                            f"RefinedScene{original['index']}",
                            __base__=Observation,
                            scene=(Literal[original["index"]], ...),
                        ),
                        detail["frames"],
                        tokens=300,
                    )
                    observations[position] = Observation.model_validate(refined.model_dump())
                    # Keep final calls at 18 images without losing broad context:
                    # four close-look samples plus the original two endpoints.
                    selected[position] = refine_scene_frames(
                        Path(video_path),
                        Path(directory),
                        original,
                        previous.refine_at,
                        count=6,
                        retain_context=True,
                    )
                finally:
                    for call in detail_trace:
                        call.update(scene=original["index"], detail_span=detail["detail_span"])
                        entry["calls"].append(call)
                        entry["repair_count"] += max(0, len(call["attempts"]) - 1)
                        entry["attempts"].extend(
                            {
                                **attempt,
                                "scene": original["index"],
                                "model_call": len(entry["calls"]),
                                "detail_span": detail["detail_span"],
                            }
                            for attempt in call["attempts"]
                        )
                    entry["refinement"] = {
                        "scene": original["index"],
                        "requested_time": previous.refine_at,
                        "detail_span": detail["detail_span"],
                        "sample_count": 12,
                        "max_refinements": 1,
                        "retained_context_times": selected[position].get(
                            "retained_context_times", []
                        ),
                    }
            inspection = Inspection(evidence=observations)
            entry.update(status="complete", output=inspection.model_dump())
            return inspection
        except Exception:
            entry["status"] = "failed"
            raise
        finally:
            entry["latency_ms"] = round((time.monotonic() - started) * 1000)

    @staticmethod
    def validate_focus(value, scene):
        if value.refine_at is not None and not (
            math.isfinite(value.refine_at) and scene["start"] <= value.refine_at <= scene["end"]
        ):
            raise ValueError("refine_at must be a time inside this scene, or null.")

    def choose_option(self, trace, question, options, images):
        """SVP-style concise visual decision, separate from explanation formatting."""
        prompt = (
            self.role_prompts.choice_prompt(question, options)
            if self.prompt_profile != "legacy"
            else (
                f"Question: {question}\nChoices: {json.dumps(options)}\n"
                "Choose the best answer based on the chronological frames. "
                "Prefer the most specific choice directly supported by the visible actions; "
                "do not invent a narrower cause or outcome. "
                "Return the exact choice only."
            )
        )
        if self.prompt_profile == "structured" and self.generate_structured is not None:
            from videomasys.zerogpu.structured_schema import ChoiceSelection, bounded_schema

            def valid_choice(value):
                if value.selected_choice is not None and value.selected_choice not in options:
                    raise ValueError("selected_choice must be one supplied option or null")

            value = self.structured(
                trace, "Choice selection",
                prompt + "\nReturn selected_choice in the response object. Use null if the "
                "sampled evidence does not allow a choice; this is a provisional decision.",
                ChoiceSelection, images, tokens=120, validate=valid_choice,
                response_schema=bounded_schema(ChoiceSelection, options=options),
            )
            return value.selected_choice
        if self.prompt_profile == "structured":
            prompt += "\nReturn the exact choice only."
        entry = {
            "stage": "Choice selection",
            "prompt_profile": self.prompt_profile,
            "prompt_version": self.prompt_version,
            "status": "running",
            "model": self.model_id,
            "attempts": [],
        }
        trace.append(entry)
        started = time.monotonic()
        try:
            for _ in range(2):
                record = {
                    "prompt": prompt,
                    "images": [Path(p).name for p in images],
                    "max_new_tokens": 120,
                }
                entry["attempts"].append(record)
                try:
                    raw = self.generate(prompt, images, 120)
                    record["raw_response"] = raw
                    choice = raw.strip()
                    if choice.startswith('"'):
                        decoded = json.loads(choice)
                        if isinstance(decoded, str):
                            choice = decoded
                            record["normalization"] = "decode JSON string"
                    if choice not in options:
                        raise ValueError(
                            "Return exactly one supplied choice, not an index or explanation."
                        )
                    entry.update(status="complete", output={"selected_choice": choice})
                    return choice
                except (ValueError, TypeError) as exc:
                    record["validation_error"] = str(exc)
                    prompt += (
                        "\nYour output must match one choice exactly. No explanation or index."
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                    raise
            raise ValueError(
                "Choice selection failed validation after two attempts. See the real trace."
            )
        except Exception:
            entry["status"] = "failed"
            raise
        finally:
            entry["latency_ms"] = round((time.monotonic() - started) * 1000)

    def _retrieve_svp(self, bundle, context, plan):
        """Compile ordered dataflow; allow one execution-feedback retry, never a fallback.

        Structural JSON repair stays in ``structured``. This separate controller
        retry sees actual candidate/coverage failures, not an invented observation
        or an answer. Both attempts share the enclosing request allowance.
        """
        trace = bundle["trace"]
        # Unlike v2, a global question is not rewritten as three mandatory thirds.
        search_context = tool_search_context(
            context, bundle["scenes"], "local_event",
            # Caption schema already caps text at 600 characters. Keep its later
            # actions visible to the structured compiler; preserve other profiles.
            caption_limit=600 if self.prompt_profile == "structured" else 240,
        )
        cached = SceneCache(bundle["database"], self.signature).load(bundle["scenes"])
        if cached is None:
            raise ValueError("The video cache is unavailable; index the video again.")
        feedback = ""
        for iteration in range(2):
            prompt, examples = self.role_prompts.tool_prompt(search_context, plan.model_dump())
            if feedback:
                prompt += (
                    "\nEXECUTION FEEDBACK (actual previous results, not visual confirmation):\n"
                    + feedback
                    + "\nRevise the program to address these gaps within the SAME plan and "
                    "budgets. Do not invent successful results. No further execution retry is "
                    "available. A video-wide locate must be explicit, never an implicit fallback."
                )
            program = self.structured(
                trace, "Tool agent" if not iteration else "Tool agent execution repair",
                prompt, SVPProgram, tokens=1200,
                validate=lambda value: validate_svp_task_program(value, plan, bundle["scenes"]),
                prompt_examples=examples,
            )
            program.compiler = "model_repaired" if iteration else "model"
            trace[-1].update(output=program.model_dump(), execution_iteration=iteration + 1)
            yield bundle
            started = time.monotonic()
            queries = self.embed([call.query for call in program.locate])
            result = execute_svp_program(
                program, bundle["scenes"], cached[1], queries, scope=plan.scope,
            )
            gaps = {
                "uncovered_evidence_needs": result.get("uncovered_evidence_needs", []),
                # A broad search can supply target semantics without contributing
                # a final window. Do not retry just to view redundant candidates;
                # required temporal branches must still reach selected evidence.
                "uncovered_temporal_outputs": result.get("uncovered_temporal_outputs", []),
                "global_coverage_incomplete": not result.get("coverage", {}).get(
                    "global_coverage_complete", True,
                ),
            }
            failed = not result["selected"] or any(gaps.values())
            trace.append({
                "stage": "Retrieval" if not iteration else "Retrieval execution repair",
                "status": "failed" if failed else "complete", "output": result,
                "execution_iteration": iteration + 1, "embedding_model": self.embedding_id,
                "index": "SQLite + exact cosine; ordered scoped dataflow",
                "latency_ms": round((time.monotonic() - started) * 1000),
            })
            bundle["selected"] = result["selected"]
            yield bundle
            if not failed:
                return result
            # The full nested lineage stays in the trace. Feedback uses exact
            # compact observations so a long chain cannot inflate the next prompt.
            compact_calls = [{
                "call": item["call"],
                "source_scene_ids": item.get("source_scene_ids", []),
                "candidate_scope_scene_ids": item.get("candidate_scope_scene_ids", []),
                "anchor_scene": item.get("anchor_scene"),
                "hit_count": len(item.get("hits", [])),
                "first_four_hits": [
                    {key: hit[key] for key in (
                        "scene", "semantic_cosine", "temporal_proximity", "window_relation",
                    ) if key in hit}
                    for hit in item.get("hits", [])[:4]
                ],
            } for item in result["calls"] if isinstance(item.get("call"), dict)]
            feedback = json.dumps({
                "program": program.model_dump(), "gaps": gaps,
                "selected_scenes": [scene["index"] for scene in result["selected"]],
                "coverage": result.get("coverage", {}),
                "calls": compact_calls,
            }, separators=(",", ":"))
        raise ValueError(
            "SVP retrieval still has uncovered evidence after one execution-feedback repair. "
            "No answer was generated. See both actual retrieval attempts."
        )

    @staticmethod
    def _inspection_targets(plan, retrieval):
        """Carry selected evidence roles into inspection, without injecting cached claims."""
        descriptions = {need.id: need.description for need in plan.evidence_needs}
        targets = {}
        for item in retrieval.get("selection_reasons", []):
            ids = list(dict.fromkeys(
                need_id for reason in item.get("reasons", [])
                for need_id in (
                    reason.get("evidence_need_ids") or reason.get("lineage_evidence_need_ids", [])
                )
                if need_id in descriptions
            ))
            if ids:
                targets[item["scene"]] = [descriptions[need_id] for need_id in ids]
        return targets

    def qa(self, bundle, question: str, options: list[str]):
        if not 2 <= len(question.strip()) <= 2000:
            raise ValueError("Enter a question between 2 and 2000 characters.")
        if options and (not 2 <= len(options) <= 5 or any(len(o) > 300 for o in options)):
            raise ValueError("Use 2–5 answer choices, each at most 300 characters.")
        if options and (any(not o.strip() for o in options) or len(set(options)) != len(options)):
            raise ValueError("Answer choices must be nonempty and distinct.")
        if self.prompt_profile == "structured":
            for field in ("answer", "verification", "evidence_graph", "resolution",
                          "completion_issue", "reviewed_pair"):
                bundle.pop(field, None)
            bundle.update(status="unverified", final_answer_complete=False,
                          review_complete=False, retained_previous_draft=False,
                          retained_reviewed_answer=False, explanation_repair_failed=False)
        trace = bundle["trace"]
        context = f"Question: {question}\n"
        if options:
            context += f"Answer choices (hypotheses): {json.dumps(options)}\n"
        else:
            context += "Response mode: open-ended.\n"

        def validate_plan(plan):
            if len(plan.evidence_needs) > 3 or plan.max_intervals > 3:
                raise ValueError("Use at most three evidence needs and intervals.")
            if plan.max_inspected_seconds > 60:
                raise ValueError("The inspection budget must not exceed 60 seconds.")
            if (self.prompt_profile == "structured" and plan.scope == "whole_video"
                    and any(need.depends_on for need in plan.evidence_needs)):
                raise ValueError(
                    "Whole-video needs must be independent. Reconsider the question's target: "
                    "if it asks about a particular object use or event and needs evidence "
                    "relative to that interaction, use scope='local_event' and preserve the "
                    "genuine event dependencies and need IDs. Words such as 'overall process' "
                    "do not alone require whole_video. For a true overall-activity question, "
                    "keep whole_video and use empty depends_on for independent representative "
                    "actions or context; a conceptual connection is not a temporal dependency. "
                    "Do not erase a genuine before/after dependency just to retain the scope."
                )

        plan_prompt, plan_examples = (
            self.role_prompts.planner_prompt(context)
            if self.prompt_profile != "legacy"
            else planner_prompt(context)
        )
        plan = self.structured(
            trace,
            "Planner",
            plan_prompt,
            SVPPlan if self.prompt_profile in {"svp", "structured"} else QAPlan,
            tokens=600,
            validate=validate_plan,
            prompt_examples=plan_examples,
        )
        yield bundle

        if self.prompt_profile in {"svp", "structured"}:
            retrieval = yield from self._retrieve_svp(bundle, context, plan)
        else:
            retrieval = yield from self._retrieve_v2(bundle, context, plan)
        selected = bundle["selected"]
        if not selected:
            raise ValueError("No evidence fits the plan's budget. No answer was generated.")
        yield from self._answer_retrieved(bundle, question, options, plan, retrieval)

    def _retrieve_v2(self, bundle, context, plan):
        """Preserved v2 baseline; no new SVP semantics are applied implicitly."""
        trace = bundle["trace"]
        search_context = tool_search_context(context, bundle["scenes"], plan.scope)
        program_prompt, program_examples = (
            self.role_prompts.tool_prompt(search_context, plan.model_dump())
            if self.prompt_profile != "legacy"
            else tool_prompt(search_context, plan.model_dump())
        )
        program = self.structured(
            trace,
            "Tool agent",
            program_prompt,
            ToolProgram,
            tokens=800,
            validate=lambda value: ToolProgramValidator().validate(value, plan),
            prompt_examples=program_examples,
        )
        program.compiler = "model"  # provenance belongs to the controller
        trace[-1]["output"] = program.model_dump()
        yield bundle
        start = time.monotonic()
        cached = SceneCache(bundle["database"], self.signature).load(bundle["scenes"])
        if cached is None:
            raise ValueError("The video cache is unavailable; index the video again.")
        vectors = cached[1]
        queries = self.embed([c.query for c in program.locate])
        retrieval = execute_program(program, bundle["scenes"], vectors, queries, scope=plan.scope)
        coverage_failed = (
            plan.scope == "whole_video" and not retrieval["temporal_coverage"]["complete"]
        )
        trace.append(
            {
                "stage": "Retrieval",
                "status": "failed" if coverage_failed else "complete",
                "output": retrieval,
                "embedding_model": self.embedding_id,
                "index": "SQLite + exact cosine",
                "latency_ms": round((time.monotonic() - start) * 1000),
            }
        )
        bundle["selected"] = retrieval["selected"]
        yield bundle
        if coverage_failed:
            missing = ", ".join(retrieval["temporal_coverage"]["missing_positions"])
            raise ValueError(
                "Whole-video evidence coverage is incomplete: no distinct representative "
                f"fits the retrieval and inspection budgets for {missing}. "
                "No answer was generated. See Retrieval for the actual candidates and coverage."
            )
        return retrieval

    def _answer_retrieved(self, bundle, question, options, plan, retrieval):
        if self.prompt_profile == "structured":
            from videomasys.zerogpu.structured_pipeline import answer_retrieved

            yield from answer_retrieved(self, bundle, question, options, plan, retrieval)
            return
        trace = bundle["trace"]
        selected = bundle["selected"]
        context = f"Question: {question}\n"
        if options:
            context += f"Answer choices (hypotheses): {json.dumps(options)}\n"
        else:
            context += "Response mode: open-ended.\n"
        # Cached captions find windows; fresh, denser images decide the answer.
        # Legacy in-memory test bundles may already supply their own frames.
        if bundle.get("video_path") and bundle.get("frame_directory"):
            selected = [
                sample_scene_frames(
                    Path(bundle["video_path"]), Path(bundle["frame_directory"]), scene, count=6
                )
                for scene in selected
            ]
            bundle["selected"] = selected
        inspection = self.inspect_windows(
            trace,
            selected,
            question,
            [need.description for need in plan.evidence_needs],
            video_path=bundle.get("video_path"),
            directory=bundle.get("frame_directory"),
            **({"needs_by_scene": self._inspection_targets(plan, retrieval)}
               if self.prompt_profile == "svp" else {}),
        )
        mapping, image_number = [], 1
        for scene in selected:
            mapping.append(
                {
                    "scene": scene["index"],
                    "span": [round(scene["start"], 2), round(scene["end"], 2)],
                    "image_numbers": list(range(image_number, image_number + len(scene["frames"]))),
                    "sample_times": [round(t, 2) for t in scene.get("frame_times", [])],
                }
            )
            image_number += len(scene["frames"])
        images = [p for s in selected for p in s["frames"]]

        # Relevance is a fallible model judgment, not permission to hide frames
        # from the image-backed answerer and verifier.
        relevant_ids = {scene["index"] for scene in selected}
        evidence = inspection.model_dump_json()
        yield bundle
        selected_choice = self.choose_option(trace, question, options, images) if options else None
        if options:
            yield bundle

        def validate_answer(value):
            if not value.answer.strip():
                raise ValueError("answer must contain a substantive response or visual limitation.")
            if not options and value.answer.strip().rstrip(".").casefold() in {
                "none",
                "null",
                "n/a",
            }:
                raise ValueError(
                    "Open-ended QA needs a substantive answer or a specific visual limitation, "
                    "not a placeholder. Answer choices are not required."
                )
            if not value.supporting_scenes or not set(value.supporting_scenes) <= relevant_ids:
                raise ValueError(
                    f"supporting_scenes must use only scene IDs {sorted(relevant_ids)}; "
                    f"you used {value.supporting_scenes}. Do not cite image numbers."
                )
            if options:
                compared = [item.choice for item in value.option_comparisons]
                if len(set(compared)) != len(options) or set(compared) != set(options):
                    raise ValueError(
                        "Compare every supplied option exactly once using its exact text."
                    )
                best = [c.choice for c in value.option_comparisons if c.status == "best_supported"]
                if len(best) > 1 or (best and best[0] != value.selected_choice):
                    raise ValueError("Only the selected_choice may be marked best_supported.")
                if not best and not value.limitations:
                    raise ValueError(
                        "With no best-supported choice, state the unresolved ambiguity."
                    )

        comparison_schema = create_model(
            "SuppliedChoiceComparison",
            __base__=ChoiceComparison,
            choice=(Literal[tuple(options)] if options else str, ...),
        )
        answer_schema = create_model(
            "GroundedAnswer",
            __base__=AnswerText,
            selected_choice=(
                Literal[selected_choice] if options else type(None),
                Field(
                    description="The exact candidate from Choice selection; null for open-ended QA"
                ),
            ),
            option_comparisons=(
                list[comparison_schema],
                Field(min_length=len(options), max_length=len(options)),
            ),
            supporting_scenes=(
                list[Literal[tuple(sorted(relevant_ids))]],
                Field(min_length=1, max_length=3, description="Scene IDs, never image numbers"),
            ),
        )
        if not options:
            answer_schema = create_model(
                "OpenEndedAnswer",
                __base__=EvidenceAnswer,
                supporting_scenes=(
                    list[Literal[tuple(sorted(relevant_ids))]],
                    Field(min_length=1, max_length=3, description="Scene IDs, never image numbers"),
                ),
            )
        answer_prompt = (
            context
            + "Answer from the selected images and inspected observations below. Images are "
            "authoritative if an observation omits or misreads a detail. Never use unsampled "
            "video or treat the question's assumptions as new visual evidence. "
            f"The separate visual choice reader selected: {selected_choice!r}. "
            "Explain and audit this candidate; it is not proof. Do not silently change it. "
            "If it is unsupported, mark it unclear with specific limitations instead of "
            "inventing justification. The verifier can reject it. "
            "Image mapping: "
            + json.dumps(mapping)
            + "\nObservations: "
            + evidence
            + "\nFirst compare EVERY supplied choice in option_comparisons, with one short "
            f"visual reason each: exactly {len(options)} entries, INCLUDING the selected choice. "
            "Mark the most specific supported answer best_supported; "
            "a merely compatible property/context is compatible, not equally explanatory. "
            "An object's contents or location alone do not explain its FUNCTION: prefer a "
            "supported action explaining its use over merely restating what it contains. "
            "Distinguish competing actions using the interaction before, during, and after "
            "contact. If none is established, mark unclear and state the missing observation. "
            "Compare ALL choices against concrete actions, not merely the setting. Choose "
            "the most specific supported explanation; a broad compatible description may "
            "be inferior. Copy the chosen option text EXACTLY into selected_choice; do not "
            "output a numeric option index. The answer prose must explain that same choice. "
            "For why/purpose questions, a same-object action sequence can support "
            "a reasonable purpose inference; intent need not be literally visible. Separate "
            "visible facts in answer from the reasoning bridge in inference. Do not claim "
            "completion of an action beyond the shown frames. "
            f"supporting_scenes may ONLY contain IDs {sorted(relevant_ids)}, never image numbers. "
            "Cite only scenes supporting the explanation, not every retrieved setup scene. "
            "If evidence "
            "is ambiguous, state the particular missing fact; do not invent certainty. "
            "Answer in at most 80 words and inference in at most 45 words."
        )
        if not options:
            answer_prompt = (
                context
                + OPEN_ANSWER_RULES
                + "\nImage mapping: "
                + json.dumps(mapping)
                + "\nObservations: "
                + evidence
                + f"\nsupporting_scenes may only contain IDs {sorted(relevant_ids)}."
            )
        calibration_rules = EVIDENCE_CALIBRATION_RULES
        if self.prompt_profile != "legacy":
            answer_prompt = self.role_prompts.answer_prompt(
                context, mapping, evidence, relevant_ids, options, selected_choice
            )
            calibration_rules = ""
        answer = self.structured(
            trace,
            "Answer",
            answer_prompt + "\n" + calibration_rules,
            answer_schema,
            images,
            tokens=750,
            validate=validate_answer,
        )
        selected_option = options.index(answer.selected_choice) if options else None
        answer = Answer(**answer.model_dump(), selected_option=selected_option)
        trace[-1]["qa_mode"] = "multiple_choice" if options else "open_ended"
        if options:
            trace[-1]["controller_transform"] = {
                "operation": "exact_choice_text_to_zero_based_index",
                "selected_choice": answer.selected_choice,
                "selected_option": selected_option,
            }
        trace[-1]["output"] = answer.model_dump()
        bundle["answer"] = answer.model_dump()
        yield bundle

        def validate_verdict(value):
            if not value.accepted and not value.issues:
                raise ValueError("A rejection needs a specific issue.")
            if value.accepted and value.issues:
                raise ValueError("An accepted draft must have issues=[].")
            if value.accepted and value.needs_revision:
                raise ValueError("An accepted draft must have needs_revision=false.")
            if (
                value.accepted
                and options
                and not any(
                    c.choice == answer.selected_choice and c.status == "best_supported"
                    for c in answer.option_comparisons
                )
            ):
                raise ValueError("Cannot accept a choice the answerer itself marks as unresolved.")

        verifier_prompt = (
            context
            + "Visually audit the draft using the selected images, not just the observations. "
            "An inspector can be wrong: check its claimed object interactions against the "
            "actual images. Never rubber-stamp a plausible everyday use without seeing its "
            "visual basis. Check observations, temporal ordering, and exact option text. "
            "Check the same actor/object links the anchor and outcome. Reject invented visual "
            "facts, contradicted actions, mismatched indices, or equally supported alternatives. "
            "Also audit WHY plausible alternatives were downgraded: saying 'preparation rather "
            "than another function' needs a visible discriminator, not an assumed purpose. "
            "Do not accept a broad property as a function when a competing specific use remains "
            "unresolved by the actual contact sequence. Report that ambiguity instead. "
            "Purpose/why questions allow a grounded inference from visible actions: do NOT "
            "reject merely because intent is not directly visible, an action spans frames, or "
            "the clip stops before the inferred next action completes. Distinguish 'not "
            "observed in these samples' from 'did not happen'. Prefer the specific option "
            "that explains the observed interaction over a vague compatible alternative. "
            "The inference field must connect visible facts, not substitute for missing facts. "
            "Accepted drafts must have issues=[]; otherwise give specific evidence failures. "
            f"The draft selects EXACTLY this option: {answer.selected_choice!r}. "
            "Reject if the prose explains a different choice or never supports this choice. "
            "Image mapping: "
            + json.dumps(mapping)
            + "\nUncovered retrieval needs (not proof of absent events): "
            + json.dumps(retrieval.get("uncovered_evidence_needs", []))
            + "\nUncovered temporal outputs: "
            + json.dumps(retrieval.get("uncovered_temporal_outputs", []))
            + "\nEvidence: "
            + evidence
            + "\nDraft: "
        )
        if not options:
            verifier_prompt = (
                context
                + OPEN_VERIFIER_RULES
                + "\nImage mapping: "
                + json.dumps(mapping)
                + "\nUncovered retrieval needs: "
                + json.dumps(retrieval.get("uncovered_evidence_needs", []))
                + "\nUncovered temporal outputs: "
                + json.dumps(retrieval.get("uncovered_temporal_outputs", []))
                + "\nObservations: "
                + evidence
                + "\nDraft: "
            )
        review_rules = (
            "\n"
            + EVIDENCE_CALIBRATION_RULES
            + "\nSet needs_revision=true only when the issues can be addressed by correcting "
            "or qualifying the answer using these SAME images. A missing visual cause can "
            "be handled by explicitly explaining uncertainty. Set false if accepted or if "
            "a wording correction cannot resolve the evidence problem. No new retrieval "
            "or invented facts are permitted."
        )

        def review_payload(value):
            if not options:
                value = EvidenceAnswer.model_validate(value.model_dump())
            return value.model_dump_json()

        if self.prompt_profile != "legacy":
            verifier_prompt = self.role_prompts.verifier_prompt(
                context, mapping, evidence, retrieval, options, answer.selected_choice
            )
            review_rules = self.role_prompts.review_rules()

        verdict = self.structured(
            trace,
            "Verifier",
            verifier_prompt + review_payload(answer) + review_rules,
            Verdict,
            images,
            tokens=250,
            validate=validate_verdict,
        )
        bundle["verification"] = verdict.model_dump()
        bundle["status"] = "verified" if verdict.accepted else "unverified"
        yield bundle

        # One feedback-driven correction, never an unbounded accept-until-pass loop.
        # Both initial outputs remain in the trace, and the final reviewer may reject.
        if not verdict.accepted and verdict.needs_revision:
            revised = self.structured(
                trace,
                "Answer",
                self.role_prompts.revision_prompt(
                    answer_prompt, review_payload(answer), verdict.issues
                )
                if self.prompt_profile != "legacy"
                else (
                    answer_prompt
                    + "\n"
                    + EVIDENCE_CALIBRATION_RULES
                    + "\nRevise your previous draft to address this visual review. Use only the "
                    "same images: remove unsupported details, correct citations, and qualify "
                    "unproven causes. Do not just restate rejected claims. If support is missing, "
                    "explain that specific limitation. Do not claim the review itself is evidence."
                    + "\nPrevious draft: "
                    + review_payload(answer)
                    + "\nReview issues: "
                    + json.dumps(verdict.issues)
                ),
                answer_schema,
                images,
                tokens=750,
                validate=validate_answer,
            )
            selected_option = options.index(revised.selected_choice) if options else None
            answer = Answer(**revised.model_dump(), selected_option=selected_option)
            trace[-1].update(
                revision=1,
                revision_reason="verifier_feedback",
                qa_mode="multiple_choice" if options else "open_ended",
                output=answer.model_dump(),
            )
            if options:
                trace[-1]["controller_transform"] = {
                    "operation": "exact_choice_text_to_zero_based_index",
                    "selected_choice": answer.selected_choice,
                    "selected_option": selected_option,
                }
            bundle["answer"] = answer.model_dump()
            bundle.pop("verification", None)
            bundle["status"] = "unverified"
            yield bundle
            verdict = self.structured(
                trace,
                "Verifier",
                verifier_prompt
                + review_payload(answer)
                + review_rules
                + (
                    self.role_prompts.FINAL_REVIEW_RULE
                    if self.prompt_profile != "legacy"
                    else (
                        "\nThis is the final review after one correction. No further revision "
                        "is available. Set needs_revision=false and report any remaining issues; "
                        "do not accept merely because the revision budget is exhausted."
                    )
                ),
                Verdict,
                images,
                tokens=250,
                validate=validate_verdict,
            )
            trace[-1].update(revision=1, max_answer_revisions=1)
            bundle["verification"] = verdict.model_dump()
            bundle["status"] = "verified" if verdict.accepted else "unverified"
            yield bundle
