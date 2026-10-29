"""Recreate the curated example thumbnails from real video frames (no model calls)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "space_library"
# Representative frames for recognition, not model-generated evidence or answers.
FRAMES = (
    ("nextqa-4882821564.mp4", 35, "christmas-sofa.jpg"),
    ("nextqa-4882821564.mp4", 15, "christmas-tree.jpg"),
    ("egoschema-0074f737.mp4", 45, "kitchen.jpg"),
    ("egoschema-00b9a0de.mp4", 70, "painting.jpg"),
)


def main():
    folder = ROOT / "thumbnails"
    if folder.is_symlink():
        raise ValueError("Thumbnail folder must not be a symlink.")
    folder.mkdir(exist_ok=True)
    for video, seconds, name in FRAMES:
        target = folder / name
        if target.is_symlink():
            raise ValueError("Thumbnail destination must not be a symlink.")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(seconds),
                "-i",
                str(ROOT / video),
                "-frames:v",
                "1",
                "-vf",
                "scale=384:216:force_original_aspect_ratio=decrease,"
                "pad=384:216:(ow-iw)/2:(oh-ih)/2:color=0x142731",
                "-q:v",
                "4",
                str(target),
            ],
            check=True,
            timeout=30,
        )
        if not 0 < target.stat().st_size <= 64 * 1024:
            raise ValueError(f"Thumbnail does not fit the 64 KiB limit: {name}")
        print(f"{name}: frame at {seconds}s, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
