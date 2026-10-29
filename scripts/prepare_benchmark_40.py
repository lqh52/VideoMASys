"""Select 20 unseen, labeled examples per dataset without calling any model.

NExT-QA uses the official validation CSV. EgoSchema uses questions.json joined to
the public 500-question subset_answers.json, never generated descriptions or
submission predictions. Ground truth is stored only under each case's reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

SEED = 20260908
MAX_BYTES = 100 * 1024 * 1024
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".py", ".md", ".txt", ".yaml", ".yml"}
ID_PATTERN = (
    r"(?<![0-9])[0-9]{8,12}(?![0-9])|"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|egoschema-[0-9a-f]{8}"
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def annotations(nextqa: Path, egoschema: Path) -> tuple[dict, list[Path]]:
    next_labels = nextqa / "nextqa" / "val.csv"
    ego_questions = egoschema / "questions.json"
    ego_labels = egoschema / "subset_answers.json"
    with next_labels.open(newline="", encoding="utf-8") as source:
        next_rows = [
            {
                "id": f"nextqa-{row['video']}-q{row['qid']}",
                "video_id": row["video"],
                "question": row["question"],
                "options": [row[f"a{i}"] for i in range(5)],
                "question_type": row["type"],
                "label": int(row["answer"]),
            }
            for row in csv.DictReader(source)
        ]
    labels = json.loads(ego_labels.read_text())
    ego_rows = [
        {
            "id": f"egoschema-{row['q_uid']}",
            "video_id": row["q_uid"],
            "question": row["question"],
            "options": [row[f"option {i}"] for i in range(5)],
            "label": labels[row["q_uid"]],
        }
        for row in json.loads(ego_questions.read_text())
        if row["q_uid"] in labels
    ]
    if len(ego_rows) != len(labels) or len({row["id"] for row in ego_rows}) != len(labels):
        raise ValueError("EgoSchema public labels do not join one-to-one with questions.")
    return {"nextqa": next_rows, "egoschema": ego_rows}, [next_labels, ego_questions, ego_labels]


def historical_exclusions(repo: Path, output: Path, rows: dict) -> tuple[dict, dict]:
    """Conservatively exclude dataset IDs mentioned in prior project text/artifacts."""
    known = {row["video_id"] for items in rows.values() for row in items}
    prefixes = {
        f"egoschema-{identifier[:8]}": identifier for identifier in known if "-" in identifier
    }
    roots = [
        repo / name for name in ("artifacts", "docs", "scripts", "src", "tests", "space_library")
    ]
    superseded = repo / "artifacts/benchmark_200_2026-09-08"
    files = [repo / "README.md"]
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and not path.is_relative_to(output)
            and not path.is_relative_to(superseded)
        )
    sources = defaultdict(set)
    # rg outputs only the matched ID, not potentially large trace lines or credentials.
    for offset in range(0, len(files), 150):
        batch = [str(path) for path in files[offset : offset + 150] if path.is_file()]
        result = subprocess.run(
            [
                "rg",
                "--pcre2",
                "--no-heading",
                "--with-filename",
                "--only-matching",
                ID_PATTERN,
                *batch,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        for line in result.stdout.splitlines():
            filename, match = line.rsplit(":", 1)
            identifier = prefixes.get(match, match)
            if identifier in known:
                sources[identifier].add(str(Path(filename).relative_to(repo)))
    by_dataset = {
        dataset: sorted({row["video_id"] for row in items} & sources.keys())
        for dataset, items in rows.items()
    }
    report = {
        "method": (
            "Conservative full video-ID or EgoSchema filename-prefix mentions in project text, "
            "including prior artifacts, tests, demos, and evaluation scripts. Excludes this "
            "output root and the superseded, never-executed benchmark_200 preparation."
        ),
        "scanned_text_files": len(files),
        "sources_by_video_id": {key: sorted(value) for key, value in sorted(sources.items())},
    }
    return by_dataset, report


def media_metadata(path: Path) -> tuple[float, int, int]:
    result = subprocess.run(
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
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    stream = next(item for item in data["streams"] if item.get("codec_type") == "video")
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Missing/invalid video duration")
    return duration, int(stream["width"]), int(stream["height"])


def select_cases(
    dataset: str, rows: list[dict], videos: dict, excluded: list[str], count: int, seed: int
) -> tuple[list[dict], dict]:
    candidates = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(candidates)
    skipped, examples = Counter(), defaultdict(list)
    seen, selected = set(), []
    examined = 0
    for row in candidates:
        examined += 1
        identifier = row["video_id"]
        options, label = row["options"], row["label"]
        path = videos.get(identifier)
        reason = None
        if identifier in excluded:
            reason = "prior_project_video"
        elif identifier in seen:
            reason = "duplicate_video_in_random_draw"
        elif (
            not isinstance(row["question"], str)
            or not 2 <= len(row["question"].strip()) <= 2000
            or len(options) != 5
            or any(
                not isinstance(item, str) or not 1 <= len(item.strip()) <= 300 for item in options
            )
            or len({item.strip().casefold() for item in options}) != 5
            or type(label) is not int
            or not 0 <= label < 5
        ):
            reason = "invalid_question_options_or_label"
        elif path is None or not path.is_file():
            reason = "missing_video"
        elif not 0 < path.stat().st_size <= MAX_BYTES:
            reason = "empty_or_over_100_mib"
        else:
            try:
                duration, width, height = media_metadata(path)
                if duration > 180:
                    reason = "over_180_seconds"
                elif max(width, height) > 3840:
                    reason = "over_3840_pixels"
            except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError):
                reason = "invalid_media_probe"
        if reason:
            skipped[reason] += 1
            examples[reason].append(row["id"])
            continue
        seen.add(identifier)
        case = {key: value for key, value in row.items() if key != "label"}
        case.update(
            dataset=dataset,
            split="val" if dataset == "nextqa" else "public_labeled_subset_500",
            video=str(path.resolve()),
            video_sha256=sha256(path),
            video_bytes=path.stat().st_size,
            duration_seconds=duration,
            width=width,
            height=height,
            reference={"correct_option": label, "correct_choice": options[label]},
        )
        selected.append(case)
        if len(selected) % 25 == 0:
            print(f"{dataset}: validated {len(selected)}/{count}", flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)}/{count} eligible {dataset} cases; skips={dict(skipped)}"
        )
    return selected, {
        "official_labeled_questions": len(rows),
        "official_unique_videos": len({row["video_id"] for row in rows}),
        "selected_questions": len(selected),
        "selected_unique_videos": len(seen),
        "examined_in_seeded_order": examined,
        "not_examined_after_quota": len(rows) - examined,
        "skipped_counts_in_examined_draw": dict(sorted(skipped.items())),
        "skipped_case_ids_in_examined_draw": dict(sorted(examples.items())),
        "counts_note": (
            f"Media checks stop at {count} eligible examples; skip counts describe "
            "the examined random draw, not a full dataset census."
        ),
        "duration_seconds": {
            "min": min(case["duration_seconds"] for case in selected),
            "max": max(case["duration_seconds"] for case in selected),
        },
        "label_counts": dict(
            sorted(Counter(case["reference"]["correct_option"] for case in selected).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nextqa", type=Path, default=Path("/scratch/DataSets/NExtQA"))
    parser.add_argument(
        "--egoschema", type=Path, default=Path("/scratch/Projects/3DInfant/SVP_data/EgoSchema")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmark_40_2026-09-08/manifest.json")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    if args.count <= 0:
        raise ValueError("count must be positive")
    rows, label_paths = annotations(args.nextqa, args.egoschema)
    exclusions, exclusion_report = historical_exclusions(args.repo.resolve(), output.parent, rows)
    print(
        f"Prior-video exclusions: { {key: len(value) for key, value in exclusions.items()} }",
        flush=True,
    )
    videos = {
        "nextqa": {path.stem: path for path in sorted((args.nextqa / "NExTVideo").rglob("*.mp4"))},
        "egoschema": {
            path.stem: path
            for path in sorted(
                (args.egoschema / "competitions/egoschema-public/videos/videos").glob("*.mp4")
            )
        },
    }
    cases, reports = [], {}
    for dataset in ("nextqa", "egoschema"):
        selected, reports[dataset] = select_cases(
            dataset, rows[dataset], videos[dataset], exclusions[dataset], args.count, args.seed
        )
        cases.extend(selected)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "selection_method": (
            "For each dataset independently: sort official question IDs, shuffle with "
            f"Python random.Random(seed), take the first {args.count} eligible questions "
            "with unique videos. No answer- or question-type stratification."
        ),
        "intended_prompt_profile": "structured",
        "intended_prompt_version": "structured-evidence-v3",
        "label_policy": (
            "NExT-QA official val.csv answer indices; EgoSchema questions.json joined only "
            "to public subset_answers.json. Original question and five-option order preserved. "
            "reference is scoring-only and MUST NOT be passed to inference."
        ),
        "media_policy": {
            "max_duration_seconds": 180,
            "max_video_bytes": MAX_BYTES,
            "max_dimension": 3840,
            "validation": (
                "ffprobe video stream/duration/size plus SHA-256; full decode "
                "is deferred to pipeline frame extraction."
            ),
        },
        "label_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in label_paths
        ],
        "excluded_video_ids": exclusions,
        "exclusion_report": exclusion_report,
        "selection_report": reports,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": sha256(output),
                "cases": len(cases),
                "selection_report": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
