"""Owner-triggered fixed-frame prompt replay and bounded end-to-end validation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from evaluate_openai_prompts import (  # noqa: E402
    EvaluationError,
    read_owner_key,
    safe_failure,
    selected_model,
    source_hashes,
    write_json,
)
from pydantic import Field, create_model  # noqa: E402

from videomasys.zerogpu import clean_prompts  # noqa: E402
from videomasys.zerogpu.core import (  # noqa: E402
    Answer,
    AnswerText,
    ChoiceComparison,
    EvidenceAnswer,
    Inspection,
    Observation,
    Verdict,
    prepare_video,
)
from videomasys.zerogpu.openai import OpenAI41MiniVLM  # noqa: E402
from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    GPT41_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.usage import UsageGuard  # noqa: E402

ROOT = REPO / "artifacts/prompt_tuning_2026-09-07"
BASELINE = REPO / "artifacts/local_clean_v2_2026-09-07"
LIMITS = {"max_runs": 8, "max_requests": 60, "max_requests_per_run": 32}
FORMAT = (
    "\nReturn one JSON object matching the supplied response schema. "
    "No markdown; keep text fields concise."
)
REPLAY_CASES = ("sofa_mc", "kitchen_open")
FULL_CASES = ("painting_open", "ball_mc")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


class ProvenanceVLM(OpenAI41MiniVLM):
    """Hash the actual credential-free wire payload before normal admitted transport."""

    def _perform(self, payload, deadline, record):
        body = json.loads(payload)
        content = body["messages"][1]["content"]
        record["request_sha256"] = sha(payload)
        record["wire_image_sha256"] = [
            sha(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
            for item in content if item["type"] == "image_url"
        ]
        record["request_schema_sha256"] = json_sha(body.get("response_format"))
        return super()._perform(payload, deadline, record)


def prompt_module(profile):
    return importlib.import_module(f"videomasys.zerogpu.{profile}_prompts")


def evaluation_directory(profile, version, mode, case):
    allowed = {
        "clean": {"clean-roles-v2"},
        "tuned": {"grounded-roles-v1", "grounded-roles-v2"},
    }
    if version not in allowed.get(profile, set()):
        raise EvaluationError("This prompt version is not authorized for the bounded batch.")
    prefix = f"{profile}__{version}" if version == "grounded-roles-v2" else profile
    return ROOT / f"{prefix}__{mode}__{case}"


def schemas(options, candidate, scene_ids, suspect):
    comparison = create_model(
        "SuppliedChoiceComparison", __base__=ChoiceComparison,
        choice=(Literal[tuple(options)] if options else str, ...),
    )
    citations = (
        list[Literal[tuple(sorted(scene_ids))]],
        Field(min_length=1, max_length=3, description="Scene IDs, never image numbers"),
    )
    if options:
        answer = create_model(
            "GroundedAnswer", __base__=AnswerText,
            selected_choice=(Literal[candidate], Field(
                description="The exact candidate from Choice selection; null for open-ended QA"
            )),
            option_comparisons=(list[comparison], Field(
                min_length=len(options), max_length=len(options)
            )),
            supporting_scenes=citations,
        )
    else:
        answer = create_model(
            "OpenEndedAnswer", __base__=EvidenceAnswer, supporting_scenes=citations,
        )
    inspection = create_model(
        f"ObservationScene{suspect}", __base__=Observation,
        scene=(Literal[suspect], Field(description="Exact scene ID, not image number")),
    )
    return answer, inspection


def review_payload(answer, options):
    value = Answer.model_validate(answer)
    if not options:
        value = EvidenceAnswer.model_validate(value.model_dump())
    return value.model_dump_json()


def replay_inputs(case):
    """Reconstruct and prove equality to original clean requests before admission."""
    directory = BASELINE / f"{GPT41_MINI_MODEL}__{case}"
    source = json.loads((directory / "bundle.json").read_text())
    trace = json.loads((directory / "raw-trace.json").read_text())
    def first(stage):
        return next(item for item in trace if item["stage"] == stage)

    selected = source["selected"]
    options = source["options"]
    candidate = first("Choice selection")["output"]["selected_choice"] if options else None
    suspect = 3 if case == "sofa_mc" else 6
    scene = next(item for item in selected if item["index"] == suspect)
    inspect_call = next(
        item for item in first("Evidence inspection")["calls"]
        if item["stage"] == f"Inspect scene {suspect}"
    )
    original_answer = first("Answer")
    original_review = first("Verifier")
    original_prompt = original_answer["attempts"][0]["prompt"]
    context = "Question: " + original_prompt.split("\nQuestion: ", 1)[1].split(
        "Image mapping: ", 1
    )[0]
    mapping = json.JSONDecoder().raw_decode(original_prompt.split("Image mapping: ", 1)[1])[0]
    evidence = original_prompt.split("\nObservations: ", 1)[1].split(
        "\nsupporting_scenes", 1
    )[0]
    Inspection.model_validate_json(evidence)
    needs = [need["description"] for need in first("Planner")["output"]["evidence_needs"]]
    scene_ids = {item["index"] for item in selected}
    answer_schema, inspect_schema = schemas(options, candidate, scene_ids, suspect)
    retrieval = first("Retrieval")["output"]
    data = {
        "context": context, "mapping": mapping, "evidence": evidence,
        "scene_ids": sorted(scene_ids), "options": options, "candidate": candidate,
        "retrieval": retrieval, "scene": scene, "question": source["question"],
        "needs": needs, "original_draft": original_answer["output"],
        "images": [path for item in selected for path in item["frames"]],
        "selected": selected,
    }
    clean = replay_prompts(clean_prompts, data)
    checks = (
        (clean["answer"] + FORMAT, original_answer["attempts"][0]["prompt"]),
        (clean["inspection"] + FORMAT, inspect_call["attempts"][0]["prompt"]),
        (clean["original_review"] + FORMAT, original_review["attempts"][0]["prompt"]),
        (answer_schema.model_json_schema(), original_answer["attempts"][0]["response_schema"]),
        (inspect_schema.model_json_schema(), inspect_call["attempts"][0]["response_schema"]),
        (Verdict.model_json_schema(), original_review["attempts"][0]["response_schema"]),
        ([Path(path).name for path in data["images"]], original_answer["attempts"][0]["images"]),
        ([Path(path).name for path in data["images"]], original_review["attempts"][0]["images"]),
        ([Path(path).name for path in scene["frames"]], inspect_call["attempts"][0]["images"]),
    )
    if not all(left == right for left, right in checks):
        raise EvaluationError("Recorded baseline inputs no longer match exact replay contracts.")
    data["provenance"] = {
        "baseline_files": {
            str(directory / name): sha((directory / name).read_bytes())
            for name in ("bundle.json", "raw-trace.json", "manifest.json")
        },
        "baseline_source_sha256": json.loads((directory / "manifest.json").read_text())[
            "source_sha256"
        ],
        "frame_files": {path: sha(Path(path).read_bytes()) for path in data["images"]},
        "fixed_inputs_sha256": json_sha(data),
        "original_request_sha256": {
            name: sha(prompt.encode()) for name, prompt in clean.items()
        },
        "exact_clean_request_and_schema_match": True,
        "independent_stage_replay": True,
        "inspection_output_not_fed_to_answer": True,
    }
    return data, answer_schema, inspect_schema


def replay_prompts(module, data):
    review = module.verifier_prompt(
        data["context"], data["mapping"], data["evidence"], data["retrieval"],
        data["options"], data["candidate"],
    )
    return {
        "inspection": module.inspect_prompt(data["scene"], data["question"], data["needs"]),
        "answer": module.answer_prompt(
            data["context"], data["mapping"], data["evidence"], data["scene_ids"],
            data["options"], data["candidate"],
        ) + "\n",
        "original_review": review + review_payload(data["original_draft"], data["options"])
        + module.review_rules(),
        "review_prefix": review,
    }


def validate_answer(value, data):
    if not value.answer.strip() or value.answer.strip().rstrip(".").casefold() in {
        "none", "null", "n/a",
    }:
        raise ValueError("A substantive answer or specific visual limitation is required.")
    if data["options"]:
        choices = [item.choice for item in value.option_comparisons]
        if len(set(choices)) != len(data["options"]) or set(choices) != set(data["options"]):
            raise ValueError("Compare every supplied choice exactly once.")
        best = [item.choice for item in value.option_comparisons if item.status == "best_supported"]
        if len(best) > 1 or (best and best[0] != data["candidate"]):
            raise ValueError("Only the fixed candidate may be best-supported.")
        if not best and not value.limitations:
            raise ValueError("Unresolved choices need a specific limitation.")


def validate_verdict(value, draft, options):
    if value.accepted and (value.issues or value.needs_revision):
        raise ValueError("Accepted drafts cannot have issues or request revision.")
    if not value.accepted and not value.issues:
        raise ValueError("Rejected drafts need a specific issue.")
    if value.accepted and options and not any(
        item["choice"] == draft["selected_choice"] and item["status"] == "best_supported"
        for item in draft.get("option_comparisons", [])
    ):
        raise ValueError("Cannot accept an unresolved answer choice.")


def run_replay(active, bundle, data, answer_schema, inspect_schema, module):
    prompts = replay_prompts(module, data)
    trace = bundle["trace"]
    print("Suspect-window inspection started", flush=True)
    active.structured(
        trace, "Replay inspection", prompts["inspection"], inspect_schema,
        data["scene"]["frames"], tokens=300,
        validate=lambda value: active.validate_focus(value, data["scene"]),
    )
    print("Frozen-observation answer started", flush=True)
    answer = active.structured(
        trace, "Replay answer", prompts["answer"], answer_schema, data["images"], tokens=750,
        validate=lambda value: validate_answer(value, data),
    )
    fresh = answer.model_dump()
    if data["options"]:
        fresh["selected_option"] = data["options"].index(fresh["selected_choice"])
    fresh = Answer(**fresh).model_dump()
    bundle["answer"] = fresh
    print("Original-draft review started", flush=True)
    old_review = active.structured(
        trace, "Replay original verifier", prompts["original_review"], Verdict,
        data["images"], tokens=250,
        validate=lambda value: validate_verdict(value, data["original_draft"], data["options"]),
    )
    bundle["original_draft_verification"] = old_review.model_dump()
    print("Fresh-draft review started", flush=True)
    fresh_review = active.structured(
        trace, "Replay fresh verifier", prompts["review_prefix"]
        + review_payload(fresh, data["options"]) + module.review_rules(),
        Verdict, data["images"], tokens=250,
        validate=lambda value: validate_verdict(value, fresh, data["options"]),
    )
    bundle["verification"] = fresh_review.model_dump()


def full_inputs(case):
    if case == "ball_mc":
        return {
            "video": Path("/scratch/DataSets/NExtQA/NExTVideo/0052/2535384528.mp4"),
            "question": "what does the man do after finishing observing the ball",
            "options": ["walks away", "caps the bottle", "hug and dance", "laugh",
                        "pass the microphone to lady"],
            "validation_scope": "new video excluded from prompt-development cases",
            "selection_seed": "mini-tune-holdout-2026-09-07-v1",
            "dataset_question_id": 7,
        }
    entries = json.loads((REPO / "space_library/library.json").read_text())
    filename = "egoschema-00b9a0de.mp4"
    entry = entries[3]
    if entry["file"] != filename:
        raise EvaluationError("The fixed validation library entry has changed.")
    return {
        "video": REPO / "space_library" / filename,
        "question": entry["question"],
        "options": [],
        "validation_scope": "different video from replay, not a guaranteed unseen benchmark",
    }


def evaluate(mode, profile, case):
    if profile not in ("clean", "tuned") or (
        mode == "replay" and case not in REPLAY_CASES
    ) or (mode == "full" and (case not in FULL_CASES or profile != "tuned")) or mode not in (
        "replay", "full"
    ):
        raise EvaluationError("Choose an allowed replay pair or tuned full-pipeline case.")
    module = prompt_module(profile)
    output = evaluation_directory(profile, module.PROMPT_VERSION, mode, case)
    fixed = replay_inputs(case) if mode == "replay" else None
    inputs = fixed[0] if fixed else full_inputs(case)
    key = read_owner_key()
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise EvaluationError(
            "This profile/mode/case already exists; no rerun or overwrite."
        ) from None
    hashes = source_hashes()
    hashes[str(Path(__file__).relative_to(REPO))] = sha(Path(__file__).read_bytes())
    manifest = {
        "model": GPT41_MINI_MODEL, "provider": "OpenAI", "mode": mode, "case": case,
        "prompt_profile": profile, "prompt_version": module.PROMPT_VERSION,
        "source_sha256": hashes, "limits": LIMITS, "ledger": str(ROOT / "usage.sqlite3"),
        "model_fallback": False, "automatic_reruns": False,
        "validation_scope": inputs.get("validation_scope", "fixed-frame stage replay"),
        "input": {name: str(value) if isinstance(value, Path) else value
                  for name, value in inputs.items() if name in (
                      "question", "options", "video", "selection_seed", "dataset_question_id"
                  )},
        "provenance": inputs.get("provenance"),
    }
    write_json(output / "manifest.json", manifest, key)
    write_json(output / "prompt-sources.json", {
        name: (REPO / "src/videomasys/zerogpu" / name).read_text()
        for name in ("clean_prompts.py", "tuned_prompts.py")
    }, key)
    if fixed:
        write_json(output / "fixed-inputs.json", fixed[0], key)
    bundle = {"trace": [], "question": inputs["question"], "options": inputs["options"]}
    completed, failure, guard = False, None, None
    started = time.monotonic()
    try:
        if fixed:
            bundle["selected"] = deepcopy(inputs["selected"])
        else:
            print("Video preparation started", flush=True)
            bundle = prepare_video(inputs["video"], ROOT / "cache" / GPT41_MINI_MODEL / profile)
            bundle.update(question=inputs["question"], options=inputs["options"],
                          video_name=inputs["video"].name)
        guard = UsageGuard(ROOT / "usage.sqlite3", **LIMITS)
        with selected_model(GPT41_MINI_MODEL):
            pipeline = RemotePipeline(
                provider="openai", token=key, guard=guard, prompt_profile=profile,
                embed=CPUEmbedder(), vlm_factory=ProvenanceVLM,
            )
        with pipeline.analysis_session(bundle) as active:
            if fixed:
                run_replay(active, bundle, *fixed, module)
            else:
                print("Scene captioning started", flush=True)
                active.index(bundle)
                if not bundle["cached"]:
                    active.save_index(bundle)
                bundle["trace"].append({
                    "stage": "Video cache", "status": "complete", "cache_hit": bundle["cached"],
                    "cache_version": active.signature, "scenes": bundle["scenes"],
                })
                iterator = active.qa(bundle, inputs["question"], inputs["options"])
                try:
                    for bundle in iterator:
                        print(f"{bundle['trace'][-1]['stage']} finished", flush=True)
                finally:
                    iterator.close()
            completed = "answer" in bundle and "verification" in bundle
    except (Exception, KeyboardInterrupt) as exc:
        completed = False
        failure = safe_failure(exc)
        bundle["error"] = failure[1]
    elapsed = round(time.monotonic() - started, 3)
    stage = bundle["trace"][-1]["stage"] if bundle["trace"] else "Preparation/admission"
    if "inference" in bundle:
        bundle["trace"].append({"stage": "Hosted inference usage", **bundle["inference"]})
    report = {
        "model": GPT41_MINI_MODEL, "mode": mode, "case": case, "prompt_profile": profile,
        "qa_completed": completed, "result": "passed" if completed else "failed",
        "result_is_execution_not_correctness": True, "elapsed_seconds": elapsed,
        "last_pipeline_stage": stage, "answer": bundle.get("answer"),
        "verifier_accepted": bundle.get("verification", {}).get("accepted"),
        "original_draft_verification": bundle.get("original_draft_verification"),
        "error_code": failure[0] if failure else None, "error": failure[1] if failure else None,
        "raw_trace_file": "raw-trace.json", "bundle_file": "bundle.json",
        "allowance": None,
        "selected_frame_paths": [p for scene in bundle.get("selected", [])
                                 for p in scene["frames"]],
        "inference_invoked": any(
            item.get("request_attempted") is True
            for call in bundle.get("inference", {}).get("calls", [])
            for item in call.get("attempts", [])
        ),
    }
    if guard is not None:
        try:
            report["allowance"] = guard.snapshot()
        except Exception:
            report["allowance"] = {"status": "Usage summary unavailable"}
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{'Complete' if completed else 'Stopped'}: {elapsed}s; artifacts: {output}", flush=True)
    return 0 if completed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--mode", choices=("replay", "full"), required=True)
    parser.add_argument("--profile", choices=("clean", "tuned"), required=True)
    parser.add_argument("--case", choices=REPLAY_CASES + FULL_CASES, required=True)
    args = parser.parse_args(argv)
    try:
        return evaluate(args.mode, args.profile, args.case)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Evaluation setup/output failed; no automatic rerun was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
