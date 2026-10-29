"""Make a labelled, local contact sheet of actual evaluation frames for visual audit."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1] / "artifacts/svp_logic_2026-09-08"
BATCHES = {"logic": ROOT, "hybrid": ROOT.with_name("svp_hybrid_2026-09-08"),
           "structured": ROOT.with_name("structured_evidence_2026-09-08")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=tuple(BATCHES), default="logic")
    parser.add_argument("run", help="Existing run directory name in the fixed evaluation batch")
    args = parser.parse_args()
    root = BATCHES[args.batch].resolve()
    directory = (root / args.run).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ValueError("Choose an existing run inside the evaluation batch.")
    output = directory / "review-frames.jpg"
    if output.exists():
        raise ValueError("The review sheet already exists; it will not be overwritten.")
    bundle = json.loads((directory / "bundle.json").read_text())
    frames = [(scene["index"], t, path) for scene in bundle["selected"]
              for t, path in zip(scene["frame_times"], scene["frames"], strict=True)]
    sheet = Image.new("RGB", (1200, ((len(frames) + 2) // 3) * 260), "#101920")
    draw = ImageDraw.Draw(sheet)
    for i, (scene, timestamp, path) in enumerate(frames):
        source = Path(path).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("Review frame must be inside this evaluation batch.")
        with Image.open(source) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (396, 230))
        x, y = (i % 3) * 400, (i // 3) * 260
        sheet.paste(thumb, (x, y + 23))
        draw.text((x + 4, y + 3), f"Scene {scene} | requested time ~{timestamp:.2f}s", fill="white")
    sheet.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
