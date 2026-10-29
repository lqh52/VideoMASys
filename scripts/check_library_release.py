"""Read-only library/media/label and optional public Space audit; never runs inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from deploy_zerogpu import deployment_files  # noqa: E402
from evaluate_zerogpu_library import load_examples, normalized  # noqa: E402
from prepare_benchmark_40 import annotations, media_metadata, sha256  # noqa: E402


def decoded_hash(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-threads", "1", "-i", str(path),
         "-map", "0:v:0", "-an", "-vsync", "0", "-threads", "1",
         "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, check=True, timeout=120,
    )
    if result.stderr.strip() or not result.stdout.strip().startswith("SHA256="):
        raise ValueError("Video decoding did not complete cleanly.")
    return result.stdout.strip().split("=", 1)[1]


def audit_local(nextqa, egoschema):
    from PIL import Image

    from videomasys.zerogpu.ui import read_library

    root = ROOT / "space_library"
    manifest = json.loads((root / "library.json").read_text())
    deployed = deployment_files(root)
    loaded = read_library(root)
    examples = load_examples(root, nextqa, egoschema)
    rows, sources = annotations(nextqa, egoschema)
    official = {row["id"]: row for values in rows.values() for row in values}
    media, records = {}, []
    for example, item, ui_item in zip(examples, manifest, loaded, strict=True):
        reference = example["reference"]
        if example["id"].startswith("nextqa-"):
            video_id, qid = reference["annotation_id"].split(":")
            official_id = f"nextqa-{video_id}-q{qid}"
            source_matches = list((nextqa / "NExTVideo").glob(f"*/{video_id}.mp4"))
            if len(source_matches) != 1:
                raise ValueError("NExT-QA source media must resolve uniquely.")
            original = source_matches[0]
        else:
            official_id = f"egoschema-{item['video_id']}"
            original = (egoschema / "competitions/egoschema-public/videos/videos"
                        / f"{item['video_id']}.mp4")
        row = official[official_id]
        if row["label"] != reference["correct_option"]:
            raise ValueError("Curated reference differs from the official label.")
        # NExT-QA UI paraphrases are checked against explicit aliases by load_examples.
        if example["id"].startswith("egoschema-") and (
            normalized(row["question"]) != normalized(item["question"])
            or [normalized(v) for v in row["options"]]
            != [normalized(v) for v in item["options"]]
        ):
            raise ValueError("EgoSchema UI question/choices differ from official annotations.")
        video = Path(example["video"])
        if ui_item["path"] != str(video) or not ui_item["thumbnail_data"]:
            raise ValueError("UI media or thumbnail does not load as expected.")
        if item["file"] not in media:
            duration, width, height = media_metadata(video)
            if not (0 < duration <= 180 and max(width, height) <= 3840
                    and video.stat().st_size <= 100 * 1024 * 1024):
                raise ValueError("Library video exceeds app limits.")
            local_hash, original_hash = sha256(video), sha256(original)
            pixels = decoded_hash(video)
            original_pixels = pixels if local_hash == original_hash else decoded_hash(original)
            if pixels != original_pixels:
                raise ValueError("Library video pixels differ from the official source video.")
            media[item["file"]] = {
                "local_path": str(video), "bytes": video.stat().st_size,
                "duration_seconds": duration, "width": width, "height": height,
                "sha256": local_hash, "source_path": str(original.resolve()),
                "source_sha256": original_hash,
                "source_file_identical": local_hash == original_hash,
                "decoded_video_sha256": pixels, "source_decoded_video_sha256": original_pixels,
                "full_video_decode_passed": True, "source_decoded_video_identical": True,
            }
        thumb = root / item["thumbnail"]
        with Image.open(thumb) as picture:
            thumbnail = {"path": str(thumb), "bytes": thumb.stat().st_size,
                         "width": picture.width, "height": picture.height,
                         "sha256": sha256(thumb)}
        records.append({
            "id": example["id"], "ui_index": example["number"] - 1,
            "video_id": item["video_id"], "file": item["file"],
            "ui_load_path": ui_item["path"], "question": item["question"],
            "options": item["options"], "official_question_id": official_id,
            "correct_option_zero_based": row["label"],
            "correct_choice": item["options"][row["label"]],
            "official_source_options": row["options"],
            "wording_check": reference["wording_check"], "thumbnail": thumbnail,
        })
    keys = {(r["video_id"], normalized(r["question"])) for r in records}
    if len(keys) != len(records):
        raise ValueError("Duplicate library question.")
    return {
        "checked_at": datetime.now(UTC).isoformat(), "library_static_valid": True,
        "library_manifest_sha256": sha256(root / "library.json"),
        "question_count": len(records), "unique_videos": len(media), "duplicate_questions": 0,
        "shared_video_note": "The sofa and tree questions intentionally use one Christmas clip.",
        "reference_sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "examples": records, "media": media,
        "deployment_library_paths": sorted(k for k in deployed if k.startswith("space_library/")),
        "reference_answers_in_deployment": False,
        "visual_review": {"status": "pending",
                          "note": "Separate sampled-frame inspection required."},
        "model_release_gate": {"status": "pending", "required_new_holdout_cases": 40,
                               "minimum_final_answers": 36, "minimum_correct_all_attempts": 28,
                               "library_live_inference": "not run by this audit"},
        "remote_writes": False, "paid_inference_calls": 0,
    }


def audit_space(repo_id, paths):
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    info = api.repo_info(repo_id, repo_type="space")
    runtime = api.get_space_runtime(repo_id)
    variables = api.get_space_variables(repo_id)
    allowed = {"VLM_BACKEND", "VLM_MODEL", "VLM_PROMPT_PROFILE", "DEMO_MAX_RUNS",
               "DEMO_MAX_REQUESTS", "DEMO_MAX_REQUESTS_PER_RUN", "DEMO_CACHE_DIR"}
    matches = []
    for remote in api.get_paths_info(repo_id, paths, revision=info.sha, repo_type="space"):
        local = ROOT / remote.path
        lfs_sha256 = getattr(getattr(remote, "lfs", None), "sha256", None)
        if lfs_sha256:
            digest, expected = sha256(local), lfs_sha256
        else:
            content = local.read_bytes()
            digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            expected = remote.blob_id
        matches.append({"path": remote.path, "local_bytes": local.stat().st_size,
                        "remote_bytes": remote.size, "content_hash_matches": digest == expected,
                        "remote_content_hash": expected,
                        "hash_algorithm": "sha256" if lfs_sha256 else "git_blob_sha1"})
    return {
        "authenticated_account": identity.get("name"), "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/spaces/{repo_id}",
        "app_url": "https://" + repo_id.replace("/", "-") + ".hf.space",
        "private": info.private, "source_revision": info.sha, "stage": str(runtime.stage),
        "hardware": str(runtime.hardware), "requested_hardware": str(runtime.requested_hardware),
        "sleep_seconds": runtime.sleep_time,
        "variables": {key: value.value for key, value in variables.items() if key in allowed},
        "all_library_files_match": len(matches) == len(paths)
        and all(item["content_hash_matches"] for item in matches),
        "library_files": matches, "app_domain_readiness_checked": False,
        "rollback_note": "Preserve this source revision and these variables before any release.",
        "deployment_guard": (
            "Uploader requires --allow-public; it preserves hardware/sleep if --hardware is "
            "omitted, never changes visibility or variables, and has no release-gate or rollback "
            "automation. Check current revision before a release; use a parent-commit guard "
            "for an atomic update. Rollback must restore source and prompt-profile together."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nextqa", type=Path, default=Path("/scratch/DataSets/NExtQA"))
    parser.add_argument("--egoschema", type=Path,
                        default=Path("/scratch/Projects/3DInfant/SVP_data/EgoSchema"))
    parser.add_argument("--space",
                        help="Optional read-only Hub management check, never app inference")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/library_release_2026-09-08/audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit.")
    result = audit_local(args.nextqa, args.egoschema)
    if args.space:
        result["space"] = audit_space(args.space, result["deployment_library_paths"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(json.dumps({"audit": str(args.output), "library_static_valid": True,
                      "examples": result["question_count"],
                      "unique_videos": result["unique_videos"],
                      "paid_inference_calls": 0, "remote_writes": False}))


if __name__ == "__main__":
    main()
