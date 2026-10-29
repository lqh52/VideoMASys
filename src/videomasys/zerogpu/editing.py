"""Typed, single-source edits and bounded CPU rendering; no model execution.

Paths are supplied by trusted server code, never taken from an EditDocument.
The cache is deliberately bounded: the owner must clear it when it is full.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_SECONDS = 180.0
MAX_SOURCE_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_CACHE_BYTES = 512 * 1024 * 1024
RENDER_VERSION = "edit-render-v1"
RENDER_TIMEOUT = 240.0
_SOURCE_FORMATS = "mov,matroska,webm,avi"
_RENDER_LOCK = threading.Lock()


class EditClip(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    label: str = Field(default="Clip", max_length=80)
    mute: bool = False


class EditDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["1.0"] = "1.0"
    source_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    clips: list[EditClip] = Field(max_length=12)
    title: str = Field(default="", max_length=100)


def validate_document(value, source_id: str, duration: float) -> EditDocument:
    """Validate against server-owned identity/duration; never silently clamp edits."""
    if (
        isinstance(duration, bool) or not isinstance(duration, (float, int))
        or not math.isfinite(duration) or not 0.1 <= duration <= MAX_SECONDS
    ):
        raise ValueError("Editing requires a video between 0.1 and 180 seconds.")
    # Revalidate model instances too: callers may have mutated their lists/fields.
    document = EditDocument.model_validate(
        value.model_dump() if isinstance(value, EditDocument) else value
    )
    if document.source_id != source_id:
        raise ValueError("The edit belongs to a different source video.")
    if len({clip.id for clip in document.clips}) != len(document.clips):
        raise ValueError("Each clip needs a unique id.")
    if any(
        clip.end > duration or clip.end - clip.start < 0.1 - 1e-9
        for clip in document.clips
    ):
        raise ValueError("Clips must be at least 0.1 seconds and within the source video.")
    if sum(clip.end - clip.start for clip in document.clips) > MAX_SECONDS + 1e-9:
        raise ValueError("The edited video cannot exceed 180 seconds.")
    if "\x00" in document.title:
        raise ValueError("Titles cannot contain null characters.")
    return document


def initial_document(source_id: str, duration: float) -> dict:
    return validate_document(
        {"source_id": source_id, "clips": [{"id": "clip_1", "start": 0.0, "end": duration}]},
        source_id, duration,
    ).model_dump()


def _cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Video rendering was cancelled.")


def _command(args: list[str], directory: Path, deadline: float, cancel_event) -> bytes:
    """Poll so cancellation/timeout always kill and reap the current child."""
    _cancelled(cancel_event)
    if time.monotonic() >= deadline:
        raise ValueError("Video rendering timed out.")
    with tempfile.TemporaryFile() as capture:
        try:
            process = subprocess.Popen(
                args, cwd=directory, stdin=subprocess.DEVNULL, stdout=capture,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ValueError("Video rendering requires working ffmpeg and ffprobe.") from exc
        try:
            while process.poll() is None:
                _cancelled(cancel_event)
                if time.monotonic() >= deadline:
                    raise ValueError("Video rendering timed out.")
                if capture.tell() > 1024 * 1024:
                    raise ValueError("Video metadata exceeded the processing limit.")
                time.sleep(0.05)
            _cancelled(cancel_event)
            if process.returncode:
                raise ValueError("Cannot render this video; use a standard MP4 or WebM.")
            capture.seek(0)
            result = capture.read(1024 * 1024 + 1)
            if len(result) > 1024 * 1024:
                raise ValueError("Video metadata exceeded the processing limit.")
            return result
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _probe(path: Path, deadline: float, cancel_event) -> dict:
    try:
        metadata = json.loads(_command([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", _SOURCE_FORMATS,
            "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
            "-of", "json", str(path),
        ], path.parent, min(deadline, time.monotonic() + 15), cancel_event))
        video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        duration = float(metadata["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration.")
        if not 0 < video["width"] <= 3840 or not 0 < video["height"] <= 3840:
            raise ValueError("Video dimensions exceed the 4K limit.")
        return {"duration": duration, "video": video,
                "audio": any(s["codec_type"] == "audio" for s in metadata["streams"])}
    except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read valid video metadata.") from exc


def _source_hash(path: Path, cancel_event) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError("The source must be a nonempty local video up to 100 MiB.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _cancelled(cancel_event)
            digest.update(chunk)
    return digest.hexdigest()


def render_edit(
    source_path: Path, document: EditDocument, output_root: Path, *,
    preview: bool = False, cancel_event=None,
) -> Path:
    """Render clips in document order, preserving audio unless explicitly muted.

    Busy renders fail immediately (RuntimeError); cancellation is InterruptedError.
    Cache hits still validate the current source identity and all edit bounds.
    """
    if not isinstance(preview, bool):
        raise ValueError("preview must be a boolean.")
    if isinstance(document, EditDocument) and not document.clips:
        raise ValueError("Add at least one clip before rendering.")
    _cancelled(cancel_event)
    if not _RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError("Another video is rendering. Please try again shortly.")
    try:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = root / ".render.lock"
        if lock.is_symlink():
            raise ValueError("The render lock must be a regular local file.")
        with lock.open("ab") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another video is rendering. Please try again shortly.") from exc
            return _render(source_path, document, root, preview, cancel_event)
    finally:
        _RENDER_LOCK.release()


def _render(source_path, document, output_root, preview, cancel_event) -> Path:
    source, root = Path(source_path).resolve(), Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + RENDER_TIMEOUT
    source_id = _source_hash(source, cancel_event)
    metadata = _probe(source, deadline, cancel_event)
    document = validate_document(document, source_id, metadata["duration"])
    if not document.clips:
        raise ValueError("Add at least one clip before rendering.")
    width, height = (640, 360) if preview else (1280, 720)
    key = hashlib.sha256(json.dumps(
        [RENDER_VERSION, width, height, document.model_dump()], sort_keys=True,
    ).encode()).hexdigest()
    target = root / f"edit_{key}.mp4"
    if target.is_symlink() or target == source or (
        target.exists() and os.path.samefile(target, source)
    ):
        raise ValueError("An edit output must never overwrite the source.")
    span = sum(c.end - c.start for c in document.clips)
    if target.is_file() and 0 < target.stat().st_size <= MAX_OUTPUT_BYTES:
        cached = _probe(target, deadline, cancel_event)
        if abs(cached["duration"] - span) <= 0.6:
            return target
        raise ValueError("The cached edit is incomplete; the owner must clear it.")
    entries = list(root.glob("edit_*.mp4"))
    if len(entries) >= 32 or sum(p.stat().st_size for p in entries) + (
        2 * MAX_OUTPUT_BYTES
    ) > MAX_CACHE_BYTES:
        raise ValueError("The edit cache is full; the owner must clear it.")
    with tempfile.TemporaryDirectory(prefix="render_", dir=root) as temporary:
        directory = Path(temporary)
        (directory / "title.txt").write_text(document.title, encoding="utf-8")
        parts = []
        elapsed, frames = 0.0, 0
        for index, clip in enumerate(document.clips):
            length = clip.end - clip.start
            # Quantize cumulative boundaries, not every clip independently;
            # otherwise rounding 12 short clips can cut the last clips away.
            elapsed += length
            next_frames = round(elapsed * 30)
            encoded_length = (next_frames - frames) / 30
            frames = next_frames
            name = f"part_{index:02d}.mp4"
            filters = (
                f"setpts=PTS-STARTPTS,trim=duration={length:.9f},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
                f"tpad=stop_mode=clone:stop_duration={encoded_length:.9f},"
                f"trim=duration={encoded_length:.9f}"
            )
            args = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-threads", "2",
                    "-protocol_whitelist", "file,pipe", "-format_whitelist", _SOURCE_FORMATS,
                    "-ss", f"{clip.start:.9f}",
                    "-i", str(source)]
            if not metadata["audio"]:
                args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            args += [
                "-map", "0:v:0", "-map", "0:a:0" if metadata["audio"] else "1:a:0",
                "-vf", filters, "-af",
                f"aresample=48000:async=1:first_pts=0,atrim=duration={length:.9f},"
                f"apad,atrim=duration={encoded_length:.9f}"
                + (",volume=0" if clip.mute else ""),
                "-t", f"{encoded_length:.9f}", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-crf", "23", "-maxrate", "4M", "-bufsize", "8M",
                "-threads", "2", "-filter_threads", "1", "-c:a", "aac", "-b:a", "128k",
                "-ar", "48000", "-ac", "2", "-map_metadata", "-1", "-sn", "-dn",
                "-fs", str(MAX_OUTPUT_BYTES), str(directory / name),
            ]
            _command(args, directory, deadline, cancel_event)
            parts.append((name, encoded_length))
            if sum((directory / part).stat().st_size for part, _ in parts) > MAX_OUTPUT_BYTES:
                raise ValueError("The intermediate edit exceeded the processing limit.")
        (directory / "concat.txt").write_text(
            "".join(f"file '{name}'\nduration {length:.9f}\n" for name, length in parts),
            encoding="utf-8",
        )
        output = directory / "output.mp4"
        # Overlay after concatenation: timing is the actual output clock, not a
        # sum of requested clip lengths (AAC/frame boundaries can add rounding).
        video_args = ["-c:v", "copy"]
        if document.title:
            longest = max(len(line) for line in document.title.splitlines() or [""])
            font_size = max(10, min(height // 18, int(width / max(longest * 0.65, 1))))
            video_args = [
                "-vf", f"drawtext=textfile=title.txt:expansion=none:fontsize={font_size}:"
                "fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8:"
                "x=(w-tw)/2:y=h-th-24:enable='lt(t,3)'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-maxrate", "4M", "-bufsize", "8M", "-threads", "2", "-filter_threads", "1",
            ]
        _command([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-protocol_whitelist", "file,pipe",
            "-f", "concat", "-safe", "1", "-i", "concat.txt", *video_args, "-c:a", "copy",
            "-map_metadata", "-1", "-movflags", "+faststart", "-t", f"{span:.9f}",
            "-fs", str(MAX_OUTPUT_BYTES), str(output),
        ], directory, deadline, cancel_event)
        result = _probe(output, deadline, cancel_event)
        if not 0 < output.stat().st_size <= MAX_OUTPUT_BYTES or (
            abs(result["duration"] - span) > 0.6
        ):
            raise ValueError("The edit exceeded output limits or was incomplete.")
        if _source_hash(source, cancel_event) != source_id:
            raise ValueError("The source video changed during rendering.")
        _cancelled(cancel_event)
        output.replace(target)
    return target
