"""Publish allowlisted app files; preserve existing hardware unless explicitly requested."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
MAX_THUMBNAIL_BYTES = 64 * 1024


def thumbnail_file(root: Path, value: str) -> tuple[Path, Path]:
    """Only explicitly selected, small raster files may accompany the videos."""
    from PIL import Image, UnidentifiedImageError

    if not isinstance(value, str) or not value:
        raise ValueError("Library thumbnails must name a JPEG or PNG file.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Library thumbnails must stay inside the selected directory.")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Library thumbnail files and folders must not be symlinks.")
    source = candidate.resolve()
    if not source.is_relative_to(root):
        raise ValueError("Library thumbnails must stay inside the selected directory.")
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not source.is_file():
        raise ValueError("Each thumbnail must reference an existing JPEG or PNG.")
    if not 0 < source.stat().st_size <= MAX_THUMBNAIL_BYTES:
        raise ValueError("Library thumbnails must fit the 64 KiB limit.")
    try:
        with Image.open(source) as thumbnail:
            if thumbnail.format not in {"JPEG", "PNG"} or not (
                1 <= thumbnail.width <= 1024 and 1 <= thumbnail.height <= 1024
            ):
                raise ValueError("Library thumbnails must be JPEG/PNG images up to 1024×1024.")
            thumbnail.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Library thumbnails must contain a valid JPEG or PNG image.") from exc
    return relative, source


def deployment_files(library_root: Path | None = None):
    files = {
        "README.md": ROOT / "deploy/zerogpu/README.md",
        "app.py": ROOT / "deploy/zerogpu/app.py",
        "requirements.txt": ROOT / "deploy/zerogpu/requirements.txt",
        "packages.txt": ROOT / "deploy/zerogpu/packages.txt",
        "pyproject.toml": ROOT / "pyproject.toml",
    }
    for source in (ROOT / "src/videomasys").rglob("*.py"):
        if not source.is_symlink():
            files[source.relative_to(ROOT).as_posix()] = source
    if library_root is not None:
        root = library_root.resolve()
        manifest = root / "library.json"
        if manifest.is_symlink():
            raise ValueError("Library manifest must not be a symlink.")
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, list) or not 1 <= len(entries) <= 10:
            raise ValueError("Provide 1–10 explicitly selected library examples.")
        files["space_library/library.json"] = manifest
        for entry in entries:
            relative = Path(entry["file"])
            source = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(root):
                raise ValueError("Library files must stay inside the selected directory.")
            if source.suffix.lower() not in {".mp4", ".webm"} or not source.is_file():
                raise ValueError("Each example must reference an existing MP4 or WebM.")
            if not 0 < source.stat().st_size <= 100 * 1024 * 1024:
                raise ValueError("Library videos must fit the 100 MiB upload limit.")
            if set(entry) - {
                "title",
                "file",
                "dataset",
                "video_id",
                "question",
                "options",
                "thumbnail",
            }:
                raise ValueError(
                    "Library metadata must not include reference answers or extra fields."
                )
            files[f"space_library/{relative.as_posix()}"] = source
            if "thumbnail" in entry:
                thumbnail_relative, thumbnail_source = thumbnail_file(root, entry["thumbnail"])
                files[f"space_library/{thumbnail_relative.as_posix()}"] = thumbnail_source
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Exact owner/space-name, e.g. teddieh52/videomasys-demo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hardware",
        choices=(
            SpaceHardware.CPU_BASIC.value,
            SpaceHardware.ZERO_A10G.value,
            SpaceHardware.L4X1.value,
        ),
        help="Explicit hardware request; omitted preserves existing hardware and sleep settings "
        "(new Spaces default to CPU Basic with the hosted inference backend)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        help="Required positive inactivity timeout with --hardware l4x1; never disables sleep",
    )
    parser.add_argument(
        "--allow-public",
        action="store_true",
        help="Explicitly permit updating an existing public Space; never changes visibility "
        "and new Spaces are still created private",
    )
    parser.add_argument(
        "--include-library",
        action="store_true",
        help="Include only the videos and thumbnails listed in space_library/library.json",
    )
    args = parser.parse_args()
    if len(args.repo_id.split("/")) != 2:
        parser.error("Provide an explicit owner/space-name.")
    if args.sleep_seconds is not None and args.sleep_seconds <= 0:
        parser.error("--sleep-seconds must be positive; disabling sleep is not supported.")
    if args.hardware == SpaceHardware.L4X1.value and args.sleep_seconds is None:
        parser.error("Paid --hardware l4x1 requires explicit positive --sleep-seconds.")
    if args.sleep_seconds is not None and args.hardware != SpaceHardware.L4X1.value:
        parser.error("--sleep-seconds requires an explicit --hardware l4x1 request.")
    files = deployment_files(ROOT / "space_library" if args.include_library else None)
    if args.dry_run:
        print("\n".join(sorted(files)))
        return
    api = HfApi()
    created = False
    try:
        info = api.repo_info(args.repo_id, repo_type="space")
    except RepositoryNotFoundError as exc:
        # Never mistake an authentication failure for permission to create a repo.
        # Concurrent creation also fails safely rather than changing that target.
        if exc.response is None or exc.response.status_code != 404:
            raise
        creation = {
            "repo_type": "space",
            "private": True,
            "space_sdk": "gradio",
            "space_hardware": SpaceHardware(args.hardware)
            if args.hardware
            else SpaceHardware.CPU_BASIC,
            "exist_ok": False,
        }
        if args.sleep_seconds is not None:
            creation["space_sleep_time"] = args.sleep_seconds
        api.create_repo(args.repo_id, **creation)
        created = True
        info = api.repo_info(args.repo_id, repo_type="space")
    if not info.private and not args.allow_public:
        raise RuntimeError(
            "Refusing to update an existing public Space without explicit approval. "
            "Use --allow-public only when this public update has been authorized."
        )
    commit = api.create_commit(
        args.repo_id,
        repo_type="space",
        commit_message="Deploy live video QA demo",
        operations=[
            CommitOperationAdd(path_in_repo=target, path_or_fileobj=source)
            for target, source in files.items()
        ],
    )
    print(f"Uploaded: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Commit: {commit.oid}")
    if args.hardware and not created:
        request = {"hardware": SpaceHardware(args.hardware)}
        if args.sleep_seconds is not None:
            # One API call sets paid hardware and its safety timeout together.
            request["sleep_time"] = args.sleep_seconds
        api.request_space_hardware(args.repo_id, **request)
    print("Inspect runtime status before treating this deployment as ready.")


if __name__ == "__main__":
    main()
