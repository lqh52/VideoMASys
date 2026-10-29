from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkExample:
    dataset: str
    example_id: str
    video_path: Path
    question: str
    options: list[str]
    correct_option: int


def load_nextqa(root: Path, split: str = "val") -> list[BenchmarkExample]:
    annotation_path = root / "nextqa" / f"{split}.csv"
    video_root = root / "NExTVideo"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    video_paths = {path.stem: path for path in video_root.rglob("*.mp4")}
    examples: list[BenchmarkExample] = []
    with annotation_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            video_id = row["video"]
            path = video_paths.get(video_id)
            if path is None:
                continue
            examples.append(
                BenchmarkExample(
                    dataset="nextqa",
                    example_id=f"{video_id}:{row['qid']}",
                    video_path=path,
                    question=row["question"],
                    options=[row[f"a{index}"] for index in range(5)],
                    correct_option=int(row["answer"]),
                )
            )
    return examples


def load_egoschema(
    root: Path, annotation_file: str = "subset_first10_with_gt.json"
) -> list[BenchmarkExample]:
    annotation_path = root / annotation_file
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    video_root = root / "competitions" / "egoschema-public" / "videos" / "videos"
    examples: list[BenchmarkExample] = []
    for row in rows:
        video_id = row.get("video_uid") or row.get("q_uid")
        path = video_root / f"{video_id}.mp4"
        if not path.is_file():
            continue
        options = row.get("options") or [row[f"option {index}"] for index in range(5)]
        answer = row.get("ground_truth")
        correct_option = _egoschema_answer_index(answer, options)
        examples.append(
            BenchmarkExample(
                dataset="egoschema",
                example_id=video_id,
                video_path=path,
                question=row["question"],
                options=options,
                correct_option=correct_option,
            )
        )
    return examples


def _egoschema_answer_index(answer: str, options: list[str]) -> int:
    if answer in options:
        return options.index(answer)
    match = re.match(r"\s*O([1-9])\s*:", answer or "", flags=re.IGNORECASE)
    if match:
        index = int(match.group(1)) - 1
        if index < len(options):
            return index
    raise ValueError(f"cannot map EgoSchema answer to an option: {answer!r}")

