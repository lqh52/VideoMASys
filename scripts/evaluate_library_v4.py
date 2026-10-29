"""Separate four-example development smoke test; never part of the fresh 40 gate.

--prepare freezes official scoring references and local media without inference.
Default is read-only preflight. --run permits at most four once-only cold-index
sessions, 160 HTTP attempts (40 each), and $1 via the audited conservative ledger.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_zerogpu_library import (  # noqa: E402
    DEFAULT_EGOSCHEMA,
    DEFAULT_NEXTQA,
    EXAMPLE_IDS,
    NEXTQA_ALIASES,
    assert_same,
)
from prepare_benchmark_40 import annotations  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_library_v4_shared", REPO / "scripts/evaluate_benchmark_40_v4.py",
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
base = shared.base
ROOT = REPO / "artifacts/library_v4_2026-09-08"
FRESH_MANIFEST = REPO / "artifacts/benchmark_40_v4_2026-09-08/manifest.json"
LIBRARY_AUDIT = REPO / "artifacts/library_release_2026-09-08/audit.json"
LIBRARY = REPO / "space_library"
VERSION = "structured-evidence-v4"
GATE = {"planned_cases": 4, "minimum_nonabstaining_reviewed_answers": 4, "minimum_correct": 4,
        "policy": "All four normally completed, nonabstaining, matching-review answers must "
                  "be correct. Retained drafts are excluded. Separate development smoke "
                  "test: never added to the held-out 40 denominator."}
base.ROOT, base.VERSION = ROOT, VERSION
base.LIMITS = {"max_runs": 4, "max_requests": 160, "max_requests_per_run": 40}
base.MAX_USD = 1.0
base.HELPERS += ("scripts/evaluate_library_v4.py", "scripts/evaluate_zerogpu_library.py",
                 "space_library/library.json")


def prepare_manifest():
    """Validate official annotations, explicit UI aliases, and exact local video bytes."""
    library_path = LIBRARY / "library.json"
    library = json.loads(library_path.read_text())
    if len(library) != 4:
        raise base.EvaluationError("Exactly the four reviewed library entries are required.")
    rows, sources = annotations(DEFAULT_NEXTQA, DEFAULT_EGOSCHEMA)
    official = {row["id"]: row for values in rows.values() for row in values}
    cases = []
    for identifier, item in zip(EXAMPLE_IDS, library, strict=True):
        if item.get("id", identifier) != identifier:
            raise base.EvaluationError("The library order or reviewed IDs changed.")
        dataset = "nextqa" if identifier.startswith("nextqa-") else "egoschema"
        if dataset == "nextqa":
            alias = NEXTQA_ALIASES[identifier]
            annotation_id = f"nextqa-{item['video_id']}-q{alias['qid']}"
            row = official[annotation_id]
            for name in ("question", "options"):
                assert_same(row[name], alias["source_" + name], f"{identifier} source {name}")
                assert_same(item[name], alias[name], f"{identifier} UI {name}")
        else:
            annotation_id = f"egoschema-{item['video_id']}"
            row = official[annotation_id]
            for name in ("question", "options"):
                assert_same(item[name], row[name], f"{identifier} {name}")
        video = (LIBRARY / item["file"]).resolve()
        if not video.is_relative_to(LIBRARY.resolve()) or not video.is_file():
            raise base.EvaluationError("Library media is missing or outside its directory.")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height:format=duration", "-of", "json", str(video),
        ], text=True))
        duration = float(probe["format"]["duration"])
        if not 0 < duration <= 180.001 or video.stat().st_size > 100 * 1024 * 1024:
            raise base.EvaluationError("Library media exceeds the fixed prototype bounds.")
        choice = row["label"]
        if (type(choice) is not int or not 0 <= choice < 5 or len(item["options"]) != 5
            or len(set(item["options"])) != 5):
            raise base.EvaluationError("Invalid official library reference or option list.")
        cases.append({"id": identifier, "dataset": dataset,
            "split": "val" if dataset == "nextqa" else "official_labeled_subset",
            "video_id": item["video_id"], "video": str(video),
            "video_sha256": base.file_sha(video), "video_bytes": video.stat().st_size,
            "duration_seconds": duration, "width": probe["streams"][0]["width"],
            "height": probe["streams"][0]["height"],
            "question": item["question"], "options": item["options"],
            "reference": {"annotation_id": annotation_id, "correct_option": choice,
                "correct_choice": item["options"][choice],
                "source_question": row["question"], "source_options": row["options"]}})
    fresh = json.loads(FRESH_MANIFEST.read_text())
    controls = {**fresh["historical_files_sha256"],
                str(FRESH_MANIFEST): base.file_sha(FRESH_MANIFEST),
                str(LIBRARY_AUDIT): base.file_sha(LIBRARY_AUDIT)}
    return {"intended_prompt_profile": "structured", "intended_prompt_version": VERSION,
        "purpose": "Previously exposed library development smoke; not held-out evaluation.",
        "release_gate": GATE, "cold_cache_policy": "Separate new cache per question, "
        "including the two questions sharing the NExT-QA video.",
        "library_manifest_sha256": base.file_sha(library_path),
        "label_sources": [{"path": str(path), "sha256": base.file_sha(path)} for path in sources],
        "historical_files_sha256": controls, "cases": cases}


def load_manifest():
    path = base.ROOT / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise base.EvaluationError("Prepare the new four-example library manifest first.")
    manifest = json.loads(path.read_text())
    if manifest != prepare_manifest() or not shared.controls_unchanged(manifest):
        raise base.EvaluationError("Frozen library inputs or previous controls changed.")
    return manifest


def summarize(manifest, budget=None, guard=None):
    # Retain all audited usage/timing/error calculations; replace only fixed cohort sizes.
    result = shared._summarize(manifest, budget, guard)
    result.update(planned=4, finished=len(result["results"]) == 4)
    for dataset, values in result["datasets"].items():
        values["planned"] = 2
        values["accuracy_all_planned"] = values["correct"] / 2 if values["attempted"] == 2 else None
        values["nonabstaining_reviewed_answers"] = sum(
            row.get("release_eligible_answer", False) for row in result["results"]
            if row["dataset"] == dataset)
    eligible = sum(row.get("release_eligible_answer", False) for row in result["results"])
    correct = sum(row["correct"] for row in result["results"])
    result["frozen_inputs_unchanged"] = (shared.frozen_inputs_unchanged()
        and all(row.get("sources_unchanged") is True for row in result["results"]))
    result["historical_controls_unchanged"] = shared.controls_unchanged(manifest)
    result["release_gate"] = {**GATE, "nonabstaining_reviewed_answers": eligible,
        "correct": correct, "evaluated": result["finished"],
        "passed": bool(result["finished"] and eligible == correct == 4
                       and result["frozen_inputs_unchanged"]
                       and result["historical_controls_unchanged"])}
    return result


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        value = {**value, "release_gate": GATE, "evaluation_kind": "library_development_smoke"}
    return shared._write_json(path, value, key)


base.load_manifest, base.summarize, base.write_json = load_manifest, summarize, write_run_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2,), default=2)
    args = parser.parse_args(argv)
    if base.PROMPT_VERSION != VERSION:
        raise base.EvaluationError("Library smoke is pinned to structured-evidence-v4.")
    if args.prepare:
        if (base.ROOT / "manifest.json").exists():
            raise base.EvaluationError("Library manifest already frozen; no overwrite allowed.")
        manifest = prepare_manifest()
        base.ROOT.mkdir(mode=0o700, exist_ok=True)
        # Preparation never reads credentials. The inference writer requires a
        # nonempty secret for redaction, so do not pass it an empty string here.
        with (base.ROOT / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"prepared": True, "cases": 4, "model_called": False,
                          "manifest_sha256": base.file_sha(base.ROOT / "manifest.json")}))
        return 0
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": 4, "unique_videos": 3,
            "model": base.GPT41_MINI_MODEL, "prompt_version": VERSION,
            "source_sha256": base.sources(), "limits": base.LIMITS, "max_usd": base.MAX_USD,
            "release_gate": GATE, "credential_read": False, "model_called": False}, indent=2))
        return 0
    with (base.ROOT / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = base.run(manifest, workers=args.workers, resume=args.resume)
        result = json.loads((base.ROOT / "summary.json").read_text())
        return code if code else (0 if result["release_gate"]["passed"] else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
