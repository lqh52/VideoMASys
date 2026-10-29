"""Bounded, endpoint-aware frame sampling without model or GPU dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

SAMPLING_PROFILE = "endpoint-v1"
MAX_FRAME_COUNT = 12
MAX_VIDEO_SECONDS = 180.0


def sample_timestamps(start: float, end: float, count: int) -> list[float]:
    """Return absolute seek targets inside a valid short-video interval.

    ``frame_times`` are requested seek times, not measured decoded-frame PTS.
    FFmpeg selects the first decodable frame at/after each requested position.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_FRAME_COUNT:
        raise ValueError(f"Frame count must be an integer between 1 and {MAX_FRAME_COUNT}.")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or not 0 <= start < end <= MAX_VIDEO_SECONDS
    ):
        raise ValueError("Frame sampling requires a finite interval within 0–180 seconds.")
    span = end - start
    if count == 1:
        return [min(start + span / 2, math.nextafter(end, start))]
    margin = min(0.25, span / 10)
    first = start + margin
    last = min(end - margin, math.nextafter(end, start))
    return [min(last, max(start, first + (last - first) * i / (count - 1))) for i in range(count)]


def _extract_frame(video_path: Path, target: Path, timestamp: float) -> None:
    """Publish a complete JPEG atomically; never cache a failed extraction."""
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-threads",
                    "2",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-ss",
                    f"{timestamp:.9f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale='min(640,iw)':-2",
                    "-threads",
                    "2",
                    "-y",
                    str(temporary),
                ],
                capture_output=True,
                timeout=40,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Frame sampling timed out. Try a shorter or smaller video.") from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            raise ValueError(
                "Cannot sample this video. Please use a standard MP4 or WebM."
            ) from exc
        if not temporary.is_file() or not temporary.stat().st_size:
            raise ValueError("No frame exists at the requested video timestamp.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def sample_scene_frames(video_path: Path, directory: Path, scene: dict, count: int) -> dict:
    """Copy scene metadata and attach cached frames from one bounded interval.

    Each cache name identifies the source file, profile, count, scene span, and
    requested times. Old quarter/three-quarter samples cannot be mistaken for
    the new endpoint-aware profile. Only sampled JPEGs are retained; no whole
    video is loaded into memory or supplied to a model.
    """
    times = sample_timestamps(scene.get("start"), scene.get("end"), count)
    index = scene.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Frame sampling requires a nonnegative integer scene index.")
    source = Path(video_path).resolve()
    if not source.is_file() or not source.stat().st_size:
        raise ValueError("Frame sampling requires a nonempty local video file.")
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    identity = json.dumps(
        [
            SAMPLING_PROFILE,
            str(source),
            stat.st_size,
            stat.st_mtime_ns,
            index,
            scene["start"],
            scene["end"],
            count,
            times,
        ],
        separators=(",", ":"),
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    frames = []
    for position, timestamp in enumerate(times):
        target = destination / f"scene_{index:02d}_n{count}_{key}_{position:02d}.jpg"
        if target.is_symlink():
            raise ValueError("Frame cache entries must be regular local files.")
        if not target.is_file() or not target.stat().st_size:
            _extract_frame(source, target, timestamp)
        frames.append(str(target))
    return dict(scene, frames=frames, frame_times=times)


def refine_scene_frames(
    video_path: Path,
    directory: Path,
    scene: dict,
    focus_time: float,
    count: int = 12,
    *,
    retain_context: bool = False,
) -> dict:
    """Sample a focused interval of at most four seconds inside this scene.

    The returned scene retains its original identity, caption, and retrieval span;
    ``detail_span`` records the focused sampling interval. The caller
    controls how many refinements are allowed across an analysis. With
    ``retain_context=True``, four focused samples plus the original first/last
    frames form a chronological six-frame view; a bad focus cannot erase both
    ends of the original window.
    """
    sample_timestamps(scene.get("start"), scene.get("end"), count)
    start, end = scene["start"], scene["end"]
    if (
        isinstance(focus_time, bool)
        or not isinstance(focus_time, (int, float))
        or not start <= focus_time <= end
        or not math.isfinite(focus_time)
    ):
        raise ValueError("Refinement focus must be a finite timestamp inside the original scene.")
    if not isinstance(retain_context, bool):
        raise ValueError("retain_context must be a boolean.")
    context = []
    if retain_context:
        if count != 6:
            raise ValueError("Context-preserving refinement requires count=6.")
        frames, times = scene.get("frames"), scene.get("frame_times")
        if (
            not isinstance(frames, list)
            or not isinstance(times, list)
            or len(frames) < 2
            or len(frames) != len(times)
            or any(
                isinstance(t, bool)
                or not isinstance(t, (int, float))
                or not start <= t < end
                or not math.isfinite(t)
                for t in times
            )
            or times != sorted(times)
        ):
            raise ValueError(
                "Context preservation requires aligned chronological frame_times and frames."
            )
        for index in (0, -1):
            if not isinstance(frames[index], (str, Path)) or not str(frames[index]):
                raise ValueError("Retained context frames must be nonempty local image files.")
            frame = Path(frames[index])
            if not frame.is_file() or not frame.stat().st_size:
                raise ValueError("Retained context frames must be nonempty local image files.")
            context.append((times[index], str(frame)))
    detail_start, detail_end = max(start, focus_time - 2.0), min(end, focus_time + 2.0)
    detailed = sample_scene_frames(
        video_path,
        directory,
        dict(scene, start=detail_start, end=detail_end),
        4 if retain_context else count,
    )
    result = dict(
        scene,
        frames=detailed["frames"],
        frame_times=detailed["frame_times"],
        detail_span=[detail_start, detail_end],
    )
    result.pop("retained_context_times", None)
    if retain_context:
        combined = list(zip(detailed["frame_times"], detailed["frames"], strict=True)) + context
        combined.sort(key=lambda item: item[0])
        result.update(
            frame_times=[item[0] for item in combined],
            frames=[item[1] for item in combined],
            retained_context_times=[item[0] for item in context],
        )
    return result
