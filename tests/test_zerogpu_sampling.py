"""Sampling tests use generated short clips, never model/evaluation answers."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from videomasys.zerogpu.media_sampling import (
    refine_scene_frames,
    sample_scene_frames,
    sample_timestamps,
)


@pytest.mark.parametrize("count", [1, 4, 6, 12])
@pytest.mark.parametrize("start,end", [(0.0, 20.0), (72.0, 90.0), (179.9, 180.0), (0.0, 1e-9)])
def test_samples_are_deterministic_ordered_and_bounded(start, end, count):
    times = sample_timestamps(start, end, count)
    assert times == sample_timestamps(start, end, count)
    assert len(times) == count
    assert times == sorted(times)
    assert all(start <= timestamp < end for timestamp in times)
    if count > 1:
        assert times[0] == pytest.approx(start + min(0.25, (end - start) / 10))
        assert times[-1] == pytest.approx(end - min(0.25, (end - start) / 10))
        gaps = [right - left for left, right in zip(times, times[1:], strict=False)]
        assert all(gap == pytest.approx(gaps[0]) for gap in gaps)
    else:
        assert times == [pytest.approx((start + end) / 2)]


@pytest.mark.parametrize("count", [0, -1, 13, 1.5, True, "4", None])
def test_invalid_frame_counts_are_rejected(count):
    with pytest.raises(ValueError, match="Frame count"):
        sample_timestamps(0, 20, count)


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, 20),
        (20, 20),
        (21, 20),
        (0, 181),
        (0, math.inf),
        (math.nan, 10),
        (False, 10),
        (0, True),
        ("0", 10),
        (None, 10),
    ],
)
def test_invalid_spans_are_rejected(start, end):
    with pytest.raises(ValueError, match="finite interval"):
        sample_timestamps(start, end, 4)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
def test_real_extraction_cache_reuse_and_profile_separation(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=800x450:rate=20",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-y",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    directory = tmp_path / "frames"
    scene = {"index": 0, "start": 0.0, "end": 1.0, "caption": "Test video", "tag": "kept"}
    cached = sample_scene_frames(video, directory, scene, 4)
    inspected = sample_scene_frames(video, directory, scene, 6)
    assert "frames" not in scene  # The caller's cached scene is never mutated.
    assert cached["caption"] == scene["caption"] and cached["tag"] == "kept"
    assert len(cached["frames"]) == 4 and len(inspected["frames"]) == 6
    assert not set(cached["frames"]) & set(inspected["frames"])
    assert cached["frame_times"][0] == inspected["frame_times"][0] == 0.1
    assert cached["frame_times"][-1] == inspected["frame_times"][-1] == 0.9
    for filename in cached["frames"] + inspected["frames"]:
        assert Path(filename).is_file() and Path(filename).stat().st_size > 0
    metadata = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                cached["frames"][0],
            ],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
    )
    assert metadata["streams"][0]["width"] == 640

    def no_new_process(*args, **kwargs):
        raise AssertionError("Cached frames should not invoke FFmpeg again")

    monkeypatch.setattr(subprocess, "run", no_new_process)
    assert sample_scene_frames(video, directory, scene, 4) == cached
    assert sample_scene_frames(video, directory, scene, 6) == inspected


def test_invalid_local_source_and_scene_id_do_not_create_cache(tmp_path):
    directory = tmp_path / "frames"
    scene = {"index": 0, "start": 0, "end": 1}
    with pytest.raises(ValueError, match="nonempty local video"):
        sample_scene_frames(tmp_path / "missing.mp4", directory, scene, 4)
    with pytest.raises(ValueError, match="scene index"):
        sample_scene_frames(tmp_path / "missing.mp4", directory, dict(scene, index="../x"), 4)
    assert not directory.exists()


@pytest.mark.parametrize("failure", ["timeout", "decode", "empty"])
def test_failed_extractions_never_leave_cached_frames(tmp_path, monkeypatch, failure):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"Isolated subprocess failure fixture")
    directory = tmp_path / "frames"

    def fail(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 40)
        if failure == "decode":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError):
        sample_scene_frames(video, directory, {"index": 0, "start": 0, "end": 1}, 4)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("focus,detail", [(20, [20, 22]), (30, [28, 32]), (40, [38, 40])])
@pytest.mark.parametrize("count", [6, 12])
def test_refinement_clips_detail_span_and_keeps_original_metadata(
    monkeypatch, tmp_path, focus, detail, count
):
    from videomasys.zerogpu import media_sampling

    calls = []

    def sample(video, directory, scene, count):
        calls.append((video, directory, scene, count))
        return dict(
            scene,
            frames=[f"detail-{index}.jpg" for index in range(count)],
            frame_times=sample_timestamps(scene["start"], scene["end"], count),
        )

    monkeypatch.setattr(media_sampling, "sample_scene_frames", sample)
    original = {
        "index": 1,
        "start": 20,
        "end": 40,
        "caption": "Original isolated caption.",
        "frames": ["original.jpg"],
        "frame_times": [25],
        "custom": "preserved",
    }
    result = refine_scene_frames(
        tmp_path / "video.mp4", tmp_path / "frames", original, focus, count
    )
    assert result["start"] == 20 and result["end"] == 40 and result["index"] == 1
    assert result["caption"] == original["caption"] and result["custom"] == "preserved"
    assert result["detail_span"] == detail
    assert len(result["frames"]) == len(result["frame_times"]) == count
    assert all(detail[0] < timestamp < detail[1] for timestamp in result["frame_times"])
    assert original["frames"] == ["original.jpg"] and "detail_span" not in original
    assert [calls[0][2]["start"], calls[0][2]["end"]] == detail
    assert calls[0][3] == count


@pytest.mark.parametrize(
    "focus", [None, "30", True, float("nan"), float("inf"), 0, 19, 41, 10**400]
)
def test_refinement_rejects_invalid_or_outside_focus_before_reading_media(tmp_path, focus):
    with pytest.raises(ValueError, match="finite timestamp inside"):
        refine_scene_frames(
            tmp_path / "missing.mp4",
            tmp_path / "frames",
            {"index": 1, "start": 20, "end": 40},
            focus,
        )
    assert not (tmp_path / "frames").exists()


@pytest.mark.parametrize("count", [0, 13, True, 6.5])
def test_refinement_enforces_shared_frame_limit(tmp_path, count):
    with pytest.raises(ValueError, match="Frame count"):
        refine_scene_frames(
            tmp_path / "missing.mp4",
            tmp_path / "frames",
            {"index": 0, "start": 0, "end": 1},
            0.5,
            count,
        )


@pytest.mark.parametrize("count", [4, 12])
def test_context_retention_requires_exactly_six_frames(tmp_path, count):
    with pytest.raises(ValueError, match="requires count=6"):
        refine_scene_frames(
            tmp_path / "missing.mp4",
            tmp_path / "frames",
            {"index": 0, "start": 0, "end": 4},
            1,
            count,
            retain_context=True,
        )


@pytest.mark.parametrize(
    "times", [None, [], [0.25], [3.75, 0.25], [0.25, float("nan")], [0.25, 4.1], [".25", 3.75]]
)
def test_context_retention_rejects_missing_or_misaligned_times(tmp_path, times):
    scene = {
        "index": 0,
        "start": 0,
        "end": 4,
        "frames": ["first.jpg", "last.jpg"],
        "frame_times": times,
    }
    with pytest.raises(ValueError, match="aligned chronological"):
        refine_scene_frames(
            tmp_path / "missing.mp4", tmp_path / "frames", scene, 1, 6, retain_context=True
        )


def test_context_retention_requires_existing_endpoint_frames(tmp_path):
    scene = {
        "index": 0,
        "start": 0,
        "end": 4,
        "frames": [str(tmp_path / "first.jpg"), str(tmp_path / "last.jpg")],
        "frame_times": [0.25, 3.75],
    }
    with pytest.raises(ValueError, match="nonempty local image files"):
        refine_scene_frames(
            tmp_path / "missing.mp4", tmp_path / "frames", scene, 1, 6, retain_context=True
        )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
def test_real_refinement_extracts_only_narrow_requested_span_and_reuses_cache(
    tmp_path, monkeypatch
):
    video = tmp_path / "detail.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=20",
            "-t",
            "4",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-y",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    scene = {"index": 0, "start": 0, "end": 4, "caption": "Isolated motion test video."}
    result = refine_scene_frames(video, tmp_path / "frames", scene, 0)
    assert result["detail_span"] == [0, 2]
    assert result["start"] == 0 and result["end"] == 4
    assert len(result["frames"]) == 12
    assert result["frame_times"][0] == 0.2 and result["frame_times"][-1] == 1.8
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in result["frames"])
    end_result = refine_scene_frames(video, tmp_path / "frames", scene, 4, count=6)
    assert end_result["detail_span"] == [2, 4]
    assert all(2 < timestamp < 4 for timestamp in end_result["frame_times"])
    assert not set(result["frames"]) & set(end_result["frames"])
    original = sample_scene_frames(video, tmp_path / "frames", scene, 6)
    retained = refine_scene_frames(
        video, tmp_path / "frames", original, 0, count=6, retain_context=True
    )
    assert retained["detail_span"] == [0, 2]
    assert len(retained["frames"]) == len(retained["frame_times"]) == 6
    assert retained["frame_times"] == sorted(retained["frame_times"])
    assert retained["retained_context_times"] == [0.25, 3.75]
    pairs = list(zip(retained["frame_times"], retained["frames"], strict=True))
    assert (0.25, original["frames"][0]) in pairs
    assert (3.75, original["frames"][-1]) in pairs
    assert (
        retained["frames"][-1] == original["frames"][-1]
    )  # Outside the chosen zoom remains visible.
    focused = [(timestamp, path) for timestamp, path in pairs if path not in original["frames"]]
    assert len(focused) == 4 and all(0 < timestamp < 2 for timestamp, _ in focused)
    assert all(Path(path).is_file() for _, path in focused)
    assert all(0 <= timestamp < 4 for timestamp in retained["frame_times"])
    assert "retained_context_times" not in original

    def no_new_process(*args, **kwargs):
        raise AssertionError("A cached refinement must not decode again.")

    monkeypatch.setattr(subprocess, "run", no_new_process)
    assert refine_scene_frames(video, tmp_path / "frames", scene, 0) == result
    assert refine_scene_frames(video, tmp_path / "frames", scene, 4, count=6) == end_result
    assert (
        refine_scene_frames(video, tmp_path / "frames", original, 0, count=6, retain_context=True)
        == retained
    )
