"""Offline contract and real-FFmpeg controls for single-video editing."""

from __future__ import annotations

import array
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from videomasys.zerogpu import editing


def run(*args: str) -> bytes:
    return subprocess.run(args, capture_output=True, check=True, timeout=30).stdout


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe are required for real-media controls")
    path = tmp_path_factory.mktemp("edit-source") / "colors.mp4"
    args = ["ffmpeg", "-v", "error", "-nostdin"]
    for color in ("red", "green", "blue"):
        args += ["-f", "lavfi", "-i", f"color=c={color}:s=160x90:r=30:d=1"]
    args += [
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-t", "3", "-threads", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ]
    run(*args)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def frame(path: Path, timestamp: float, size="1:1") -> bytes:
    return run(
        "ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={size}", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-threads", "1", "pipe:1",
    )


def rms(path: Path, timestamp: float) -> float:
    samples = array.array("h", run(
        "ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(path),
        "-t", "0.2", "-vn", "-ac", "1", "-f", "s16le", "pipe:1",
    ))
    return math.sqrt(sum(value * value for value in samples) / len(samples))


@pytest.mark.parametrize("change", [
    {"source_id": "b" * 64}, {"version": "2.0"}, {"path": "/etc/passwd"},
    {"title": "x" * 101}, {"title": "null\x00"},
    {"clips": [{"id": "bad/path", "start": 0, "end": 1}]},
    {"clips": [{"id": "c", "start": float("nan"), "end": 1}]},
    {"clips": [{"id": "c", "start": 0, "end": float("inf")}]},
    {"clips": [{"id": "c", "start": True, "end": 1}]},
    {"clips": [{"id": "c", "start": "0", "end": 1}]},
    {"clips": [{"id": "c", "start": 1, "end": 0.5}]},
    {"clips": [{"id": "c", "start": 0, "end": 0.09}]},
    {"clips": [{"id": "c", "start": 0, "end": 3.1}]},
    {"clips": [{"id": "c", "start": 0, "end": 1, "mute": "false"}]},
    {"clips": [{"id": "c", "start": 0, "end": 1}] * 2},
    {"clips": [{"id": f"c{i}", "start": 0, "end": 1} for i in range(13)]},
])
def test_document_rejects_malformed_input(change):
    value = editing.initial_document("a" * 64, 3)
    value.update(change)
    with pytest.raises(ValueError):
        editing.validate_document(value, "a" * 64, 3)


def test_document_duration_budget_and_mutation_revalidated():
    for duration in (float("nan"), float("inf"), True, 0.09, 181):
        with pytest.raises(ValueError):
            editing.initial_document("a" * 64, duration)
    value = editing.initial_document("a" * 64, 180)
    value["clips"].append({"id": "extra", "start": 0, "end": 1})
    with pytest.raises(ValueError, match="cannot exceed"):
        editing.validate_document(value, "a" * 64, 180)
    valid = editing.validate_document(editing.initial_document("a" * 64, 3), "a" * 64, 3)
    valid.clips[0].end = 5
    with pytest.raises(ValueError, match="within"):
        editing.validate_document(valid, "a" * 64, 3)


def test_empty_timeline_is_editable_but_cannot_render(tmp_path, monkeypatch):
    empty = editing.validate_document({"source_id": "a" * 64, "clips": []}, "a" * 64, 3)
    monkeypatch.setattr(editing.subprocess, "Popen", lambda *a, **kw: pytest.fail("No spawn"))
    with pytest.raises(ValueError, match="at least one"):
        editing.render_edit(tmp_path / "unused.mp4", empty, tmp_path)


def test_real_trim_reorder_audio_mute_cache_and_original(source, tmp_path):
    path, source_id = source
    document = editing.validate_document({"source_id": source_id, "clips": [
        {"id": "blue", "start": 2.2, "end": 2.8},
        {"id": "red", "start": 0.2, "end": 0.8, "mute": True},
        {"id": "green", "start": 1.2, "end": 1.8},
    ]}, source_id, 3)
    output = editing.render_edit(path, document, tmp_path, preview=True)
    assert output != path and hashlib.sha256(path.read_bytes()).hexdigest() == source_id
    assert output.parent == tmp_path and not list(tmp_path.glob("render_*"))
    for timestamp, dominant in ((0.3, 2), (0.9, 0), (1.5, 1)):
        pixel = frame(output, timestamp)
        assert pixel[dominant] > 80 and pixel[dominant] > 3 * max(
            pixel[channel] for channel in range(3) if channel != dominant
        )
    assert rms(output, 0.2) > 100 and rms(output, 1.45) > 100
    assert rms(output, 0.85) < 5
    metadata = json.loads(run(
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output),
    ))
    video, audio = metadata["streams"]
    assert (video["codec_name"], video["width"], video["height"]) == ("h264", 640, 360)
    assert video["avg_frame_rate"] == "30/1" and audio["codec_name"] == "aac"
    assert float(metadata["format"]["duration"]) == pytest.approx(1.8, abs=0.08)
    modified = output.stat().st_mtime_ns
    assert editing.render_edit(path, document, tmp_path, preview=True) == output
    assert output.stat().st_mtime_ns == modified
    export = editing.render_edit(path, document, tmp_path)
    assert export != output
    export_video = json.loads(run(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "json", str(export),
    ))["streams"][0]
    assert export_video == {"width": 1280, "height": 720}


def test_real_literal_title_only_first_three_output_seconds(source, tmp_path):
    path, source_id = source
    value = {"source_id": source_id, "clips": [
        {"id": "one", "start": 0, "end": 1.8},
        {"id": "two", "start": 0, "end": 1.8},
    ]}
    plain = editing.render_edit(path, editing.validate_document(value, source_id, 3), tmp_path,
                                preview=True)
    value["title"] = "100% %{pts}: a 'literal' [title]; \\ safe"
    titled = editing.render_edit(path, editing.validate_document(value, source_id, 3), tmp_path,
                                 preview=True)
    for timestamp, title_visible in ((0.3, True), (2.5, True), (3.4, False)):
        before, after = frame(plain, timestamp, "160:90"), frame(titled, timestamp, "160:90")
        difference = sum(abs(a - b) for a, b in zip(before, after, strict=True)) / len(before)
        assert difference > 0.5 if title_visible else difference < 0.5


def test_real_source_without_audio_gets_silent_aac(source, tmp_path):
    path, _ = source
    silent_source = tmp_path / "silent.mp4"
    run("ffmpeg", "-v", "error", "-i", str(path), "-c:v", "copy", "-an", str(silent_source))
    source_id = hashlib.sha256(silent_source.read_bytes()).hexdigest()
    document = editing.validate_document(editing.initial_document(source_id, 3), source_id, 3)
    output = editing.render_edit(silent_source, document, tmp_path / "edits", preview=True)
    assert rms(output, 0.5) == 0


def test_short_fractional_clips_keep_last_clip_and_thirty_fps(source, tmp_path):
    path, source_id = source
    clips = [{"id": f"c{i}", "start": 0.2, "end": 0.31} for i in range(11)]
    clips.append({"id": "last_blue", "start": 2.2, "end": 2.31})
    document = editing.validate_document({"source_id": source_id, "clips": clips}, source_id, 3)
    output = editing.render_edit(path, document, tmp_path, preview=True)
    pixel = frame(output, 1.28)
    assert pixel[2] > 180 and pixel[0] < 30
    video = json.loads(run(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,avg_frame_rate", "-of", "json", str(output),
    ))["streams"][0]
    assert int(video["nb_frames"]) == round(12 * 0.11 * 30)
    assert video["avg_frame_rate"] == "30/1"


def test_renderer_rejects_wrong_source_busy_and_precancel(source, tmp_path):
    path, source_id = source
    document = editing.validate_document(editing.initial_document(source_id, 3), source_id, 3)
    wrong = document.model_copy(update={"source_id": "a" * 64})
    with pytest.raises(ValueError, match="different source"):
        editing.render_edit(path, wrong, tmp_path)
    with editing._RENDER_LOCK, pytest.raises(RuntimeError, match="Another"):
        editing.render_edit(path, document, tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        editing.render_edit(path, document, tmp_path, cancel_event=cancelled)
    assert not list(tmp_path.glob("*.mp4"))


def test_source_hardlink_is_never_replaced_and_cache_budget_is_enforced(
    source, tmp_path, monkeypatch,
):
    path, source_id = source
    document = editing.validate_document(editing.initial_document(source_id, 3), source_id, 3)
    key = hashlib.sha256(json.dumps(
        [editing.RENDER_VERSION, 640, 360, document.model_dump()], sort_keys=True,
    ).encode()).hexdigest()
    target = tmp_path / f"edit_{key}.mp4"
    target.hardlink_to(path)
    with pytest.raises(ValueError, match="never overwrite"):
        editing.render_edit(path, document, tmp_path, preview=True)
    assert target.samefile(path) and hashlib.sha256(path.read_bytes()).hexdigest() == source_id
    monkeypatch.setattr(editing, "MAX_CACHE_BYTES", 1)
    with pytest.raises(ValueError, match="cache is full"):
        editing.render_edit(path, document, tmp_path / "full", preview=True)


def test_process_lock_and_cancelled_render_cleanup(source, tmp_path, monkeypatch):
    path, source_id = source
    document = editing.validate_document(editing.initial_document(source_id, 3), source_id, 3)
    with (tmp_path / ".render.lock").open("ab") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another"):
            editing.render_edit(path, document, tmp_path)
    original = editing._command

    def stop(args, *other):
        if args[0] == "ffmpeg":
            raise InterruptedError("cancelled")
        return original(args, *other)

    monkeypatch.setattr(editing, "_command", stop)
    with pytest.raises(InterruptedError):
        editing.render_edit(path, document, tmp_path)
    assert not editing._RENDER_LOCK.locked()
    assert not list(tmp_path.glob("render_*")) and not list(tmp_path.glob("*.mp4"))


@pytest.mark.parametrize("cancel", [False, True])
def test_running_child_is_killed_and_reaped_on_timeout_or_cancel(tmp_path, monkeypatch, cancel):
    event = threading.Event()

    class Process:
        returncode = None
        killed = waited = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            self.waited = True

    process = Process()
    monkeypatch.setattr(editing.subprocess, "Popen", lambda *a, **kw: process)
    times = iter([0, 2])
    monkeypatch.setattr(editing.time, "monotonic", lambda: next(times))
    if cancel:
        def launch(*args, **kwargs):
            event.set()
            return process
        monkeypatch.setattr(editing.subprocess, "Popen", launch)
    with pytest.raises(InterruptedError if cancel else ValueError):
        editing._command(["test"], tmp_path, 1, event)
    assert process.killed and process.waited
