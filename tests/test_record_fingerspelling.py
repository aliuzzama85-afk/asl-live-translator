"""Tests for pose_library.record_fingerspelling's camera-free logic.

The interactive window (`run_session`) needs a real webcam and a person, so
it isn't unit-tested; everything it relies on -- naming, resume position,
fps measurement, atomic video writing, the tracking check -- is.
"""

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pose_library.record_fingerspelling import (
    REQUESTED_FPS,
    check_take,
    first_unrecorded_index,
    measured_fps,
    parse_letters,
    write_video_atomic,
)


def _frames(n, size=(64, 48)):
    width, height = size
    return [np.full((height, width, 3), i * 10 % 256, dtype=np.uint8) for i in range(n)]


def test_parse_letters_defaults_to_whole_alphabet():
    assert parse_letters(None) == list("abcdefghijklmnopqrstuvwxyz")


def test_parse_letters_accepts_mixed_separators_and_case_in_alphabet_order():
    assert parse_letters("Z, j a j") == ["a", "j", "z"]


def test_parse_letters_rejects_non_letters():
    with pytest.raises(ValueError):
        parse_letters("a1")


def test_first_unrecorded_index_resumes_at_first_gap(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "c.mp4").write_bytes(b"x")
    assert first_unrecorded_index(["a", "b", "c"], tmp_path) == 1


def test_first_unrecorded_index_starts_over_when_complete(tmp_path):
    for letter in "ab":
        (tmp_path / f"{letter}.mp4").write_bytes(b"x")
    assert first_unrecorded_index(["a", "b"], tmp_path) == 0


def test_measured_fps_uses_actual_frame_timing():
    # 16 frames over 1.0s -> 15fps, whatever the camera claimed.
    timestamps = [i / 15 for i in range(16)]
    assert measured_fps(timestamps) == pytest.approx(15.0)


def test_measured_fps_falls_back_when_unmeasurable():
    assert measured_fps([]) == REQUESTED_FPS
    assert measured_fps([1.0]) == REQUESTED_FPS
    assert measured_fps([1.0, 1.0]) == REQUESTED_FPS


def test_write_video_atomic_round_trips_through_opencv(tmp_path):
    path = tmp_path / "raw" / "a.mp4"
    write_video_atomic(_frames(12), 15.0, path)

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(15.0)
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 12
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind


def test_write_video_atomic_keeps_the_old_take_if_writing_fails(tmp_path):
    path = tmp_path / "a.mp4"
    path.write_bytes(b"previous good take")
    broken_writer = SimpleNamespace(isOpened=lambda: False, release=lambda: None)
    with patch(
        "pose_library.record_fingerspelling.cv2.VideoWriter", return_value=broken_writer
    ), pytest.raises(OSError):
        write_video_atomic(_frames(3), 30.0, path)
    assert path.read_bytes() == b"previous good take"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("frames, fps", [([], 30.0), (None, 0.0)])
def test_write_video_atomic_rejects_bad_input(tmp_path, frames, fps):
    with pytest.raises(ValueError):
        write_video_atomic(
            _frames(2) if frames is None else frames, fps, tmp_path / "a.mp4"
        )


def test_check_take_skips_when_model_missing(tmp_path):
    assert (
        check_take(tmp_path / "a.mp4", "a", model_path=tmp_path / "missing.task")
        is None
    )


@pytest.mark.parametrize("kept, ok", [(58, True), (20, False)])
def test_check_take_reports_hand_tracking(tmp_path, kept, ok):
    model = tmp_path / "model.task"
    model.write_bytes(b"stub")
    stats = SimpleNamespace(total_frames_decoded=60, frames_kept=kept)
    with patch(
        "pose_library.extract.extract_pose_sequence", return_value=(None, stats)
    ):
        result = check_take(tmp_path / "a.mp4", "a", model_path=model)
    assert result.ok is ok
    assert f"{kept}/60" in result.message
