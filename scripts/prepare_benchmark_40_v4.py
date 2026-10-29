"""Freeze a new 20+20 labeled cohort; never call a model or alter earlier batches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from prepare_benchmark_40 import (
    MAX_BYTES,
    annotations,
    historical_exclusions,
    select_cases,
    sha256,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts/benchmark_40_v4_2026-09-08"
PREVIOUS = REPO / "artifacts/benchmark_40_2026-09-08"
SEED = 20260909
VERSION = "structured-evidence-v4"
RELEASE_GATE = {
    "planned_cases": 40,
    "minimum_nonabstaining_reviewed_answers": 36,
    "minimum_correct": 28,
    "policy": "Both thresholds must pass on this fixed 40-case cohort. Failures, abstentions "
              "and retained previous drafts are ineligible. A final complete review is required; "
              "review disagreement alone does not disqualify a normally completed final answer.",
}


def prior_controls(previous=PREVIOUS):
    previous = Path(previous)
    manifest = json.loads((previous / "manifest.json").read_text())
    if Counter(case["dataset"] for case in manifest["cases"]) != {"nextqa": 20, "egoschema": 20}:
        raise ValueError("The prior benchmark must contain exactly 20+20 cases.")
    paths = [previous / name for name in (
        "manifest.json", "run-config.json", "summary.json", "source-snapshots.json",
        "usage.sqlite3", "budget.sqlite3", "audit.json",
    )]
    paths += [previous / "cases" / case["id"] / name for case in manifest["cases"]
              for name in ("bundle.json", "report.json")]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Prior benchmark controls must be preserved regular files.")
    return manifest, {str(path.resolve()): sha256(path) for path in paths}


def build_manifest(repo, nextqa, egoschema, output, previous=PREVIOUS):
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("The v4 manifest already exists; no resampling or overwrite.")
    prior, controls = prior_controls(previous)
    rows, label_paths = annotations(Path(nextqa), Path(egoschema))
    exclusions, exposure_report = historical_exclusions(Path(repo).resolve(), output.parent, rows)
    for dataset in ("nextqa", "egoschema"):
        exclusions[dataset] = sorted(set(exclusions[dataset]) | {
            case["video_id"] for case in prior["cases"] if case["dataset"] == dataset
        })
    videos = {
        "nextqa": {path.stem: path for path in sorted((Path(nextqa) / "NExTVideo").rglob("*.mp4"))},
        "egoschema": {path.stem: path for path in sorted((Path(egoschema) /
            "competitions/egoschema-public/videos/videos").glob("*.mp4"))},
    }
    cases, reports = [], {}
    for dataset in ("nextqa", "egoschema"):
        selected, reports[dataset] = select_cases(
            dataset, rows[dataset], videos[dataset], exclusions[dataset], 20, SEED,
        )
        cases.extend(selected)
    if len({case["video_sha256"] for case in cases}) != 40:
        raise ValueError("The new cohort must contain 40 unique video byte hashes.")
    if {case["video_sha256"] for case in cases} & {
        case["video_sha256"] for case in prior["cases"]
    }:
        raise ValueError("A new video duplicates bytes from the prior benchmark.")
    return {
        "created_at": datetime.now(UTC).isoformat(), "seed": SEED,
        "selection_method": "Sort official question IDs per dataset, shuffle independently with "
                            "random.Random(20260909), take the first 20 eligible unique videos. "
                            "No answer/type stratification; no replacement after inference.",
        "intended_prompt_profile": "structured", "intended_prompt_version": VERSION,
        "release_gate": RELEASE_GATE,
        "label_policy": "Official NExT-QA val.csv and EgoSchema questions.json joined to "
                        "subset_answers.json. Original questions/options preserved; reference "
                        "is scoring-only and never passed to inference.",
        "media_policy": {"max_duration_seconds": 180, "max_video_bytes": MAX_BYTES,
                         "max_dimension": 3840, "validation": "ffprobe plus source SHA-256; "
                         "full decode deferred to fresh pipeline frame extraction."},
        "label_sources": [{"path": str(path.resolve()), "sha256": sha256(path)}
                          for path in label_paths],
        "excluded_video_ids": exclusions, "exclusion_report": exposure_report,
        "previous_benchmark_manifest": str((Path(previous) / "manifest.json").resolve()),
        "historical_files_sha256": controls,
        "selection_report": reports, "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nextqa", type=Path, default=Path("/scratch/DataSets/NExtQA"))
    parser.add_argument("--egoschema", type=Path,
                        default=Path("/scratch/Projects/3DInfant/SVP_data/EgoSchema"))
    args = parser.parse_args(argv)
    output = ROOT / "manifest.json"
    manifest = build_manifest(REPO, args.nextqa, args.egoschema, output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": sha256(output),
                      "seed": SEED, "cases": len(manifest["cases"]),
                      "selection_report": manifest["selection_report"]}, indent=2))


if __name__ == "__main__":
    main()
