"""Evaluate the four curated examples against real /analyze outputs.

Reference answers are read locally and never included in inference requests. Each
run gets a fresh directory, and generator outputs are checkpointed as they arrive.
This script consumes GPU quota unless --dry-run is supplied. A local Gradio app is
also supported with --space http://127.0.0.1:PORT; no model is loaded by this script.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import types
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_NEXTQA = Path("/scratch/DataSets/NExtQA")
DEFAULT_EGOSCHEMA = Path("/scratch/Projects/3DInfant/SVP_data/EgoSchema")
EXAMPLE_IDS = ("nextqa-sofa", "nextqa-tree", "egoschema-kitchen", "egoschema-painting")
SOURCE_FILES = {
    "schemas": "src/videomasys/domain/schemas.py",
    "tools": "src/videomasys/agents/tool_sequence.py",
    "core": "src/videomasys/zerogpu/core.py",
    "inference": "src/videomasys/zerogpu/inference.py",
}
OPTIONAL_SOURCE_FILES = {
    "retrieval": "src/videomasys/zerogpu/retrieval.py",
    "media_sampling": "src/videomasys/zerogpu/media_sampling.py",
    "prompts": "src/videomasys/zerogpu/prompts.py",
}
# The UI uses light edits of NExT-QA's original wording. Explicitly reviewed
# aliases prevent a loose semantic/fuzzy match from attaching the wrong label.
NEXTQA_ALIASES = {
    "nextqa-sofa": {
        "qid": "1",
        "source_question": (
            "why did the boy pick up one present from the group of them and move to the sofa"
        ),
        "question": "Why did the boy pick up one present and move to the sofa?",
        "source_options": [
            "share with the girl",
            "approach lady sitting there",
            "unwrap it",
            "playing with toy train",
            "gesture something",
        ],
        "options": [
            "Share with the girl",
            "Approach the lady sitting there",
            "Unwrap it",
            "Play with the toy train",
            "Gesture something",
        ],
    },
    "nextqa-tree": {
        "qid": "3",
        "source_question": (
            "why did the boy return to the christmas tree after unwrapping one present"
        ),
        "question": ("Why did the boy return to the Christmas tree after unwrapping one present?"),
        "source_options": [
            "blow the candles",
            "decorate the tree",
            "wants to continue unwrapping",
            "look at the tree",
            "put the wrapper aside",
        ],
        "options": [
            "Blow the candles",
            "Decorate the tree",
            "Continue unwrapping presents",
            "Look at the tree",
            "Put the wrapper aside",
        ],
    },
}


def normalized(text: str) -> str:
    text = re.sub(r"^\s*O\d+\s*:\s*", "", text, flags=re.I)
    return " ".join(re.findall(r"\w+", text.casefold()))


def assert_same(actual, expected, description: str) -> None:
    if isinstance(actual, list) and isinstance(expected, list):
        equal = [normalized(s) for s in actual] == [normalized(s) for s in expected]
    else:
        equal = normalized(actual) == normalized(expected)
    if not equal:
        raise ValueError(f"Reference cross-check failed for {description}: {actual!r}")


def load_examples(library_root: Path, nextqa_root: Path, egoschema_root: Path) -> list[dict]:
    """Fail closed if the curated questions/options no longer match their labels."""
    library = json.loads((library_root / "library.json").read_text())
    if len(library) != len(EXAMPLE_IDS):
        raise ValueError("This evaluation expects the four reviewed curated library entries.")
    nextqa_path = nextqa_root / "nextqa" / "val.csv"
    with nextqa_path.open(newline="", encoding="utf-8") as source:
        nextqa_rows = list(csv.DictReader(source))
    ego_path = egoschema_root / "subset_first10_with_gt.json"
    ego_rows = json.loads(ego_path.read_text())
    examples = []
    for number, (example_id, item) in enumerate(zip(EXAMPLE_IDS, library, strict=True), 1):
        video_path = (library_root / item["file"]).resolve()
        if not video_path.is_relative_to(library_root.resolve()) or not video_path.is_file():
            raise ValueError(f"Invalid or unavailable example media: {item['file']}")
        if example_id.startswith("nextqa-"):
            alias = NEXTQA_ALIASES[example_id]
            matches = [
                r
                for r in nextqa_rows
                if r["video"] == item["video_id"] and r["qid"] == alias["qid"]
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one NExT-QA annotation for {example_id}.")
            row = matches[0]
            source_options = [row[f"a{i}"] for i in range(5)]
            assert_same(row["question"], alias["source_question"], f"{example_id} source question")
            assert_same(source_options, alias["source_options"], f"{example_id} source choices")
            assert_same(item["question"], alias["question"], f"{example_id} UI question")
            assert_same(item["options"], alias["options"], f"{example_id} UI choices")
            reference = {
                "annotation_file": str(nextqa_path),
                "annotation_id": f"{row['video']}:{row['qid']}",
                "question": row["question"],
                "options": source_options,
                "correct_option": int(row["answer"]),
                "wording_check": "explicitly reviewed UI paraphrases; option order unchanged",
            }
        else:
            matches = [r for r in ego_rows if r.get("video_uid") == item["video_id"]]
            if len(matches) != 1:
                raise ValueError(f"Expected one EgoSchema annotation for {example_id}.")
            row = matches[0]
            assert_same(item["question"], row["question"], f"{example_id} question")
            assert_same(item["options"], row["options"], f"{example_id} choices")
            answers = [
                i
                for i, option in enumerate(row["options"])
                if normalized(option) == normalized(row["ground_truth"])
            ]
            if len(answers) != 1:
                raise ValueError(f"Ambiguous EgoSchema answer for {example_id}.")
            reference = {
                "annotation_file": str(ego_path),
                "annotation_id": row["video_uid"],
                "question": row["question"],
                "options": row["options"],
                "correct_option": answers[0],
                "wording_check": "exact normalized match; O1–O5 prefixes removed in UI",
            }
        correct = reference["correct_option"]
        if not 0 <= correct < len(item["options"]):
            raise ValueError(f"Invalid ground-truth option for {example_id}.")
        reference["correct_choice"] = item["options"][correct]
        examples.append(
            {
                "id": example_id,
                "number": number,
                "title": item["title"],
                "video": str(video_path),
                "video_id": item["video_id"],
                "question": item["question"],
                "options": item["options"],
                "reference": reference,
            }
        )
    return examples


def inference_inputs(example: dict, file_handler) -> tuple:
    """The entire inference payload: video, question, choices. Never ground truth."""
    return (file_handler(example["video"]), example["question"], "\n".join(example["options"]))


def reverse_options(example: dict) -> dict:
    """Reverse presented choices, keeping source references intact and scoring exact."""
    reordered = copy.deepcopy(example)
    original_options = list(example["options"])
    permutation = list(reversed(range(len(original_options))))
    reordered["options"] = [original_options[index] for index in permutation]
    reordered["original_reference"] = copy.deepcopy(example["reference"])
    original_correct = example["reference"]["correct_option"]
    reordered["reference"]["correct_option"] = permutation.index(original_correct)
    # Keep the remapped scoring index and its option list consistent. The exact
    # dataset annotation, including original wording/order, remains above.
    reordered["reference"]["options"] = list(reordered["options"])
    reordered["reference"]["wording_check"] = (
        "Source checked before reversal; original_reference preserves the dataset annotation."
    )
    reordered["option_order"] = {
        "transformation": "reversed",
        "presented_index_to_original_index": permutation,
        "original_options": original_options,
        "original_correct_option": original_correct,
        "scoring_correct_option": reordered["reference"]["correct_option"],
    }
    return reordered


def summarize_output(output, example: dict) -> dict:
    """Read structured trace outputs, not rendered HTML or guessed answer text."""
    if not isinstance(output, (list, tuple)) or len(output) < 4:
        return {"completed": False, "correct": None, "error": "No usable API output yet."}
    status, answer_html, _gallery, trace = output[:4]
    if not isinstance(trace, list):
        return {"completed": False, "correct": None, "error": "API trace is not a list."}
    answers = [s.get("output", {}) for s in trace if s.get("stage") == "Answer"]
    verdicts = [s.get("output", {}) for s in trace if s.get("stage") == "Verifier"]
    answer = answers[-1] if answers else {}
    verdict = verdicts[-1] if verdicts else {}
    selected_option = answer.get("selected_option")
    valid_option = type(selected_option) is int and 0 <= selected_option < len(example["options"])
    failures = [s for s in trace if s.get("status") == "failed"]
    completed = bool(
        answers
        and verdicts
        and not failures
        and trace[-1].get("stage") == "Verifier"
        and trace[-1].get("status") == "complete"
    )
    retrievals = [
        {"stage": s["stage"], **s.get("output", {})}
        for s in trace
        if isinstance(s.get("output"), dict) and "selected" in s["output"]
    ]
    cache = next((s for s in trace if s.get("stage") == "Video cache"), {})
    return {
        "completed": completed,
        "status_html": status,
        "answer_html": answer_html,
        "answer": answer,
        "selected_option": selected_option,
        "selected_choice": example["options"][selected_option] if valid_option else None,
        "correct": (selected_option == example["reference"]["correct_option"])
        if valid_option
        else None,
        "verdict": verdict,
        "accepted": verdict.get("accepted"),
        "retrieved_scene_ids": [s["index"] for r in retrievals for s in r.get("selected", [])],
        "retrievals": retrievals,
        "cache_hit": cache.get("cache_hit"),
        "cached_scenes": cache.get("scenes", []),
        "cache_version": cache.get("cache_version"),
        "stage_metrics": [
            {
                "stage": s.get("stage"),
                "status": s.get("status"),
                "latency_ms": s.get("latency_ms"),
                "attempts": len(s.get("attempts", [])),
                "repairs": s.get("repair_count", max(0, len(s.get("attempts", [])) - 1)),
            }
            for s in trace
        ],
        "repair_count": sum(
            s.get("repair_count", max(0, len(s.get("attempts", [])) - 1)) for s in trace
        ),
        "failures": failures,
    }


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def load_local_source(archive: Path | None):
    """Load frozen classes without leaving canonical imports changed for later runs."""
    if archive:
        with tarfile.open(archive) as source:
            contents = {}
            optional = {
                key: path
                for key, path in OPTIONAL_SOURCE_FILES.items()
                if path in source.getnames()
            }
            for key, path in {**SOURCE_FILES, **optional}.items():
                member = source.getmember(path)
                if not member.isfile() or member.size > 2_000_000:
                    raise ValueError(f"Invalid evaluation source archive member: {path}")
                contents[key] = source.extractfile(member).read().decode("utf-8")
    else:
        contents = {
            key: Path(path).read_text()
            for key, path in {**SOURCE_FILES, **OPTIONAL_SOURCE_FILES}.items()
            if key in SOURCE_FILES or Path(path).is_file()
        }
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in contents.items()}
    if archive:
        # Modules retain their own class globals after these temporary import bindings
        # are restored. Import/exec is intentional: archives are owner-trusted code.
        canonical_names = {
            "schemas": "videomasys.domain.schemas",
            "tools": "videomasys.agents.tool_sequence",
            **({"retrieval": "videomasys.zerogpu.retrieval"} if "retrieval" in contents else {}),
            **(
                {"media_sampling": "videomasys.zerogpu.media_sampling"}
                if "media_sampling" in contents
                else {}
            ),
            **({"prompts": "videomasys.zerogpu.prompts"} if "prompts" in contents else {}),
            "core": "videomasys.zerogpu.core",
        }
        previous = {name: sys.modules.get(name) for name in canonical_names.values()}
        modules = {}
        try:
            for key, name in canonical_names.items():
                unique = f"_videomasys_eval_{key}_{hashes[key][:12]}"
                module = types.ModuleType(unique)
                source_path = {**SOURCE_FILES, **OPTIONAL_SOURCE_FILES}[key]
                module.__file__ = f"{archive}!/{source_path}"
                sys.modules[unique] = module
                sys.modules[name] = module
                exec(compile(contents[key], module.__file__, "exec"), module.__dict__)
                modules[key] = module
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        core = modules["core"]
    else:
        from videomasys.zerogpu import core
    tree = ast.parse(contents["inference"])
    systems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            fields = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant)
            }
            role, content = fields.get("role"), fields.get("content")
            if isinstance(role, ast.Constant) and role.value == "system":
                if not isinstance(content, ast.Constant) or not isinstance(content.value, str):
                    raise ValueError("Runtime system prompt is no longer a string literal.")
                systems.append(content.value)
    if len(systems) != 1:
        raise ValueError("Expected exactly one system prompt in inference.py.")
    return core, systems[0], hashes


def local_pipeline(core, system_prompt: str, args):
    """Real HTTP model/embedding inference, with the hosted message ordering/settings."""
    import httpx
    import numpy as np
    from PIL import Image

    archive = getattr(args, "source_archive", None)
    if archive:
        with tarfile.open(archive) as source:
            inference_source = source.extractfile(SOURCE_FILES["inference"]).read().decode("utf-8")
    else:
        inference_source = Path(SOURCE_FILES["inference"]).read_text()
    thumbnail_expressions = [
        node.args[0]
        for node in ast.walk(ast.parse(inference_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "thumbnail"
        and len(node.args) == 1
    ]
    if len(thumbnail_expressions) != 1:
        raise ValueError("Expected one frame thumbnail policy in inference.py.")
    # The source archive is explicitly owner-trusted, as when loading its pipeline.
    # Evaluate the exact resize expression rather than duplicating a changing policy.
    thumbnail_policy = compile(ast.Expression(thumbnail_expressions[0]), "thumbnail_policy", "eval")
    generation_settings = {"temperature": 0}
    for node in ast.walk(ast.parse(inference_source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr == "generate"
        ):
            for keyword in node.keywords:
                if keyword.arg == "repetition_penalty":
                    value = ast.literal_eval(keyword.value)
                    if not isinstance(value, (int, float)) or not value > 0:
                        raise ValueError("Expected a positive literal repetition_penalty.")
                    generation_settings["repetition_penalty"] = value
    http = httpx.Client(timeout=args.timeout)

    def generate(prompt, paths, tokens):
        images = []
        size = eval(thumbnail_policy, {"__builtins__": {}, "len": len}, {"paths": paths})
        for path in paths:
            with Image.open(path) as source:
                picture = source.convert("RGB")
                picture.thumbnail(size)
                with io.BytesIO() as buffer:
                    picture.save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                picture.close()
            images.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        response = http.post(
            args.vlm_url.rstrip("/") + "/chat/completions",
            json={
                "model": args.model_id,
                **generation_settings,
                "max_tokens": tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": images + [{"type": "text", "text": prompt}]},
                ],
            },
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        if not isinstance(result, str):
            raise ValueError("VLM returned no text content.")
        return result

    def embed(texts):
        response = http.post(
            args.embedding_url.rstrip("/") + "/embeddings",
            json={"model": args.embedding_id, "input": texts},
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if len(rows) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Embedding endpoint returned invalid vectors.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Embedding endpoint returned a zero vector.")
        return vectors / norms

    pipeline = core.LivePipeline(generate, embed, args.model_id, args.embedding_id)
    pipeline.generation_settings = generation_settings
    return pipeline, http


def run_local_example(core, pipeline, example, directory, cache_root):
    started = time.monotonic()
    record = {
        "example": example,
        "started_at": datetime.now(UTC).isoformat(),
        "events": [],
        "raw_output": None,
        "error": None,
        "summary": summarize_output(None, example),
    }
    path = directory / f"{example['number']:02d}-{example['id']}.json"
    bundle = {"trace": []}

    def checkpoint():
        # The public API exposes the trace as output four; preserve the complete
        # local bundle too, including every real frame path/caption/selected scene.
        output = ["local evaluation", "", [], bundle["trace"]]
        record["raw_output"] = json.loads(json.dumps(output))
        record["bundle"] = json.loads(json.dumps(bundle))
        record["summary"] = summarize_output(output, example)
        record["wall_seconds"] = round(time.monotonic() - started, 3)
        record["events"].append(
            {
                "wall_seconds": record["wall_seconds"],
                "stage": bundle["trace"][-1].get("stage") if bundle["trace"] else "Prepare video",
            }
        )
        save_json(path, record)

    checkpoint()
    try:
        bundle = core.prepare_video(Path(example["video"]), cache_root)
        bundle.update(options=example["options"], question=example["question"])
        cached = core.SceneCache(bundle["database"], pipeline.signature).load(bundle["scenes"])
        if cached:
            bundle["scenes"], _ = cached
            bundle["cached"] = True
        else:
            bundle = pipeline.index(bundle)
            bundle = pipeline.save_index(bundle)
        bundle["trace"].append(
            {
                "stage": "Video cache",
                "status": "complete",
                "cache_hit": bundle["cached"],
                "segmentation": "uniform temporal windows; not learned scene boundaries",
                "scene_count": len(bundle["scenes"]),
                "scenes": [
                    {k: s[k] for k in ("index", "start", "end", "caption")}
                    for s in bundle["scenes"]
                ],
                "cache_version": pipeline.signature,
            }
        )
        checkpoint()
        for update in pipeline.qa(bundle, example["question"], example["options"]):
            bundle = update
            checkpoint()
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        bundle["trace"].append({"stage": "Run", "status": "failed", "error": str(exc)})
        checkpoint()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    record["finished_at"] = datetime.now(UTC).isoformat()
    checkpoint()
    return record


def run_example(client, example: dict, directory: Path, timeout: float, file_handler) -> dict:
    started = time.monotonic()
    record = {
        "example": example,
        "started_at": datetime.now(UTC).isoformat(),
        "events": [],
        "raw_output": None,
        "error": None,
        "summary": summarize_output(None, example),
    }
    path = directory / f"{example['number']:02d}-{example['id']}.json"
    seen = 0
    job = None

    def checkpoint(output=None):
        if output is not None:
            # Make a detached copy: Gradio generator results can reference mutable values.
            record["raw_output"] = json.loads(json.dumps(output, default=str))
            record["summary"] = summarize_output(record["raw_output"], example)
        record["wall_seconds"] = round(time.monotonic() - started, 3)
        save_json(path, record)

    def collect():
        nonlocal seen
        outputs = list(job.outputs())
        for output in outputs[seen:]:
            record["events"].append(
                {
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "output": output,
                }
            )
            checkpoint(output)
        seen = len(outputs)

    checkpoint()
    try:
        job = client.submit(*inference_inputs(example, file_handler), api_name="/analyze")
        while not job.done():
            collect()
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Analysis exceeded {timeout:g} seconds.")
            time.sleep(0.5)
        collect()
        checkpoint(job.result(timeout=5))
    except KeyboardInterrupt:
        record["error"] = (
            "Interrupted locally; server work may continue until the current stage ends."
        )
        if job is not None:
            collect()
            job.cancel()
        checkpoint()
        raise
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        if job is not None:
            collect()
            try:
                job.cancel()
            except Exception:
                pass
        checkpoint()
    record["finished_at"] = datetime.now(UTC).isoformat()
    checkpoint()
    return record


def aggregate(records: list[dict]) -> dict:
    completed = [r for r in records if r["summary"].get("completed") and not r.get("error")]
    correct = [r for r in completed if r["summary"].get("correct") is True]
    return {
        "attempted": len(records),
        "completed": len(completed),
        "correct": len(correct),
        "accuracy_over_attempted": len(correct) / len(records) if records else None,
        "accepted_wrong": sum(
            r["summary"].get("accepted") is True and r["summary"].get("correct") is False
            for r in completed
        ),
        "correct_rejected": sum(r["summary"].get("accepted") is False for r in correct),
        "wall_seconds": round(sum(r.get("wall_seconds", 0) for r in records), 3),
        "results": [
            {
                "id": r["example"]["id"],
                "selected_option": r["summary"].get("selected_option"),
                "correct_option": r["example"]["reference"]["correct_option"],
                "completed": r["summary"].get("completed"),
                "correct": r["summary"].get("correct"),
                "accepted": r["summary"].get("accepted"),
                "repairs": r["summary"].get("repair_count"),
                "wall_seconds": r.get("wall_seconds"),
                "error": r.get("error"),
            }
            for r in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("hosted", "local"), default="hosted")
    parser.add_argument("--space", default="teddieh52/videomasys-demo")
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8205/v1")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8206/v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--embedding-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--source-archive", type=Path, help="Owner-trusted frozen source tar.gz; local only."
    )
    parser.add_argument(
        "--cache-root", type=Path, help="Local scene cache; defaults to this run's cache/."
    )
    parser.add_argument(
        "--phase", required=True, help="baseline/improved, optionally with a run label."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/zerogpu-library")
    )
    parser.add_argument("--library-root", type=Path, default=Path("space_library"))
    parser.add_argument("--nextqa-root", type=Path, default=DEFAULT_NEXTQA)
    parser.add_argument("--egoschema-root", type=Path, default=DEFAULT_EGOSCHEMA)
    parser.add_argument("--examples", nargs="+", help="Subset by 1–4 or stable example ID.")
    parser.add_argument(
        "--reverse-options",
        action="store_true",
        help="Reverse presented choices and remap scoring labels; never send labels to inference.",
    )
    parser.add_argument("--timeout", type=float, default=600, help="Per-example wall-clock limit.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Check references/media; no network/GPU."
    )
    args = parser.parse_args()
    if not re.fullmatch(r"(?:baseline|improved)(?:-[a-z0-9][a-z0-9-]*)?", args.phase):
        parser.error("--phase must be baseline/improved or a label such as improved-v1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.source_archive and args.engine != "local":
        parser.error("--source-archive requires --engine local")
    examples = load_examples(args.library_root, args.nextqa_root, args.egoschema_root)
    if args.examples:
        selected = set(args.examples)
        known = {v for e in examples for v in (e["id"], str(e["number"]))}
        if selected - known:
            parser.error(f"Unknown examples: {', '.join(sorted(selected - known))}")
        examples = [e for e in examples if e["id"] in selected or str(e["number"]) in selected]
    if args.reverse_options:
        examples = [reverse_options(example) for example in examples]
    for example in examples:
        with Path(example["video"]).open("rb") as source:
            example["video_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output_dir / args.phase / stamp
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "space": args.space,
        "engine": args.engine,
        "reverse_options": args.reverse_options,
        "created_at": datetime.now(UTC).isoformat(),
        "dry_run": args.dry_run,
        "examples": examples,
        "reference_policy": "Local scoring only; never sent to the model.",
    }
    save_json(directory / "manifest.json", manifest)
    print(f"Artifacts: {directory}", flush=True)
    if args.engine == "local":
        if not args.source_archive:
            # Run frozen current code so edits made during a long evaluation cannot
            # change its imports, resize policy, or later reproducibility.
            snapshot = directory / "source_snapshot.tar.gz"
            with tarfile.open(snapshot, "w:gz") as target:
                for path in {**SOURCE_FILES, **OPTIONAL_SOURCE_FILES}.values():
                    if Path(path).is_file():
                        target.add(path, arcname=path)
            args.source_archive = snapshot
        core, system_prompt, hashes = load_local_source(args.source_archive)
        manifest.update(
            source_archive=str(args.source_archive) if args.source_archive else None,
            source_sha256=hashes,
            model_id=args.model_id,
            embedding_id=args.embedding_id,
            vlm_url=args.vlm_url,
            embedding_url=args.embedding_url,
            system_prompt=system_prompt,
            temperature=0,
            latency_note="Local vLLM inference: timings are not hosted ZeroGPU queue latency.",
        )
        save_json(directory / "manifest.json", manifest)
    if args.dry_run:
        for e in examples:
            print(
                f"Checked {e['id']}: reference option "
                f"{e['reference']['correct_option']} (zero-based)"
            )
        return
    if args.engine == "local":
        pipeline, http = local_pipeline(core, system_prompt, args)
        manifest["generation_settings"] = pipeline.generation_settings
        save_json(directory / "manifest.json", manifest)
        records = []
        try:
            for example in examples:
                print(f"Running {example['id']} …", flush=True)
                record = run_local_example(
                    core, pipeline, example, directory, args.cache_root or directory / "cache"
                )
                records.append(record)
                summary = aggregate(records)
                save_json(directory / "summary.json", summary)
                print(json.dumps(summary["results"][-1]), flush=True)
        finally:
            http.close()
        print(
            json.dumps({k: v for k, v in aggregate(records).items() if k != "results"}), flush=True
        )
        if any(r.get("error") or not r["summary"].get("completed") for r in records):
            raise SystemExit(1)
        return
    from gradio_client import Client, handle_file
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not args.space.startswith(("http://", "https://")):
        # A read-only revision stamp makes baseline/improved comparisons attributable.
        manifest["space_revision"] = HfApi(token=token).space_info(args.space).sha
        save_json(directory / "manifest.json", manifest)
    client = Client(args.space, token=token, download_files=False, verbose=False)
    records = []
    for example in examples:
        print(f"Running {example['id']} …", flush=True)
        record = run_example(client, example, directory, args.timeout, handle_file)
        # Do not log token-bearing exceptions, even if an upstream client includes one.
        if token and record.get("error"):
            record["error"] = record["error"].replace(token, "[REDACTED]")
            save_json(directory / f"{example['number']:02d}-{example['id']}.json", record)
        records.append(record)
        summary = aggregate(records)
        save_json(directory / "summary.json", summary)
        print(json.dumps(summary["results"][-1]), flush=True)
    print(json.dumps({k: v for k, v in aggregate(records).items() if k != "results"}), flush=True)
    if any(r.get("error") or not r["summary"].get("completed") for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
