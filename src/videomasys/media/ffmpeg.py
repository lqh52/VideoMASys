from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    width: int
    height: int
    raw: dict[str, object]


@dataclass(frozen=True)
class Interval:
    start_seconds: float
    end_seconds: float


async def _run(*args: str) -> tuple[bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        tail = stderr.decode("utf-8", errors="replace")[-4000:]
        raise MediaError(f"command failed ({process.returncode}): {' '.join(args[:3])}\n{tail}")
    return stdout, stderr


async def probe(path: Path) -> MediaInfo:
    stdout, _ = await _run(
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    )
    payload = json.loads(stdout)
    stream = next(
        (item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None
    )
    if stream is None:
        raise MediaError("uploaded file contains no video stream")
    duration = float(payload.get("format", {}).get("duration") or stream.get("duration") or 0.0)
    if duration <= 0:
        raise MediaError("video duration is missing or invalid")
    return MediaInfo(
        duration_seconds=duration,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        raw=payload,
    )


async def detect_scene_intervals(
    path: Path,
    duration_seconds: float,
    threshold: float,
    min_scene_seconds: float,
    max_scene_seconds: float,
    max_scenes: int,
) -> list[Interval]:
    filter_graph = f"select=gt(scene\\,{threshold}),showinfo"
    _, stderr = await _run(
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-vf",
        filter_graph,
        "-vsync",
        "vfr",
        "-f",
        "null",
        "-",
    )
    timestamps = [
        float(match.group(1))
        for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", stderr.decode(errors="replace"))
    ]
    return intervals_from_boundaries(
        timestamps,
        duration_seconds=duration_seconds,
        min_scene_seconds=min_scene_seconds,
        max_scene_seconds=max_scene_seconds,
        max_scenes=max_scenes,
    )


def intervals_from_boundaries(
    boundaries: list[float],
    *,
    duration_seconds: float,
    min_scene_seconds: float,
    max_scenes: int,
    max_scene_seconds: float | None = None,
) -> list[Interval]:
    points = [0.0]
    for value in sorted(set(boundaries)):
        value = max(0.0, min(value, duration_seconds))
        enough_after_previous = value - points[-1] >= min_scene_seconds
        enough_before_end = duration_seconds - value >= min_scene_seconds
        if enough_after_previous and enough_before_end:
            points.append(value)
    points.append(duration_seconds)
    intervals = [Interval(start, end) for start, end in zip(points, points[1:], strict=False)]

    if max_scene_seconds is not None:
        bounded: list[Interval] = []
        for interval in intervals:
            cursor = interval.start_seconds
            while interval.end_seconds - cursor > max_scene_seconds:
                bounded.append(Interval(cursor, cursor + max_scene_seconds))
                cursor += max_scene_seconds
            if interval.end_seconds > cursor:
                bounded.append(Interval(cursor, interval.end_seconds))
        intervals = bounded

    while len(intervals) > max_scenes:
        # Merge the shortest neighboring pair; this preserves full temporal coverage.
        index = min(
            range(len(intervals) - 1),
            key=lambda item: intervals[item + 1].end_seconds - intervals[item].start_seconds,
        )
        merged = Interval(intervals[index].start_seconds, intervals[index + 1].end_seconds)
        intervals[index : index + 2] = [merged]
    return intervals


async def extract_representative_frames(
    video_path: Path,
    interval: Interval,
    output_dir: Path,
    frame_count: int,
) -> list[Path]:
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    span = interval.end_seconds - interval.start_seconds
    fractions = [(index + 1) / (frame_count + 1) for index in range(frame_count)]
    paths: list[Path] = []
    for index, fraction in enumerate(fractions):
        timestamp = min(interval.end_seconds - 0.01, interval.start_seconds + span * fraction)
        timestamp = max(0.0, timestamp)
        output_path = output_dir / f"frame_{index:02d}_{timestamp:.3f}.jpg"
        await _run(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(960,iw)':-2",
            "-q:v",
            "3",
            "-y",
            str(output_path),
        )
        paths.append(output_path)
    return paths


async def extract_proxy_clip(
    video_path: Path,
    interval: Interval,
    output_path: Path,
    fps: float,
) -> Path:
    """Create a small motion-preserving clip for one already-selected interval."""
    await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
    await _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{interval.start_seconds:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{interval.end_seconds - interval.start_seconds:.6f}",
        "-vf",
        f"fps={fps},scale='min(640,iw)':-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    )
    return output_path
