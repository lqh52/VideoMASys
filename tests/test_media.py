from videomasys.media.ffmpeg import Interval, intervals_from_boundaries


def test_scene_intervals_cover_video_and_drop_short_scenes() -> None:
    intervals = intervals_from_boundaries(
        [0.2, 3.0, 3.5, 8.0],
        duration_seconds=10.0,
        min_scene_seconds=2.0,
        max_scenes=10,
    )
    assert intervals == [Interval(0.0, 3.0), Interval(3.0, 8.0), Interval(8.0, 10.0)]


def test_scene_intervals_respect_hard_budget() -> None:
    intervals = intervals_from_boundaries(
        [2.0, 4.0, 6.0, 8.0],
        duration_seconds=10.0,
        min_scene_seconds=1.0,
        max_scenes=2,
    )
    assert len(intervals) == 2
    assert intervals[0].start_seconds == 0.0
    assert intervals[-1].end_seconds == 10.0


def test_long_continuous_scene_is_split_into_bounded_windows() -> None:
    intervals = intervals_from_boundaries(
        [],
        duration_seconds=55.0,
        min_scene_seconds=2.0,
        max_scenes=10,
        max_scene_seconds=20.0,
    )
    assert intervals == [Interval(0.0, 20.0), Interval(20.0, 40.0), Interval(40.0, 55.0)]
