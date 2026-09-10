"""Tests for pose_library.extract.

The real MediaPipe Holistic Landmarker model file is not required for these
tests: `_make_landmarker` is mocked out with a fake landmarker whose
`detect_for_video` results are fully controlled, per this project's
established pattern of mocking model I/O rather than depending on real
model files in unit tests (see e.g. `tests/test_inference.py`).
"""

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pose_library.extract import (
    HAND_LANDMARK_NAMES,
    LANDMARK_NAMES,
    POSE_SUBSET_NAMES,
    extract_pose_sequence,
)


def _write_synthetic_video(path, num_frames: int, size=(20, 16)) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _hand(value: float):
    return [SimpleNamespace(x=value, y=value, z=value) for _ in range(21)]


def _pose():
    return [SimpleNamespace(x=i * 0.01, y=i * 0.02, z=i * 0.03) for i in range(33)]


class _FakeLandmarker:
    """Returns one canned result per `detect_for_video` call, in order."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.calls.append(timestamp_ms)
        return self._results.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture()
def model_path(tmp_path):
    # extract_pose_sequence only checks this path exists before mocking
    # _make_landmarker takes over -- content doesn't matter.
    path = tmp_path / "holistic_landmarker.task"
    path.write_bytes(b"not a real model, extraction is mocked")
    return path


def test_extract_drops_frames_with_no_hand_detected(tmp_path, model_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=3)

    no_hands = SimpleNamespace(
        left_hand_landmarks=[], right_hand_landmarks=[], pose_landmarks=[]
    )
    left_only = SimpleNamespace(
        left_hand_landmarks=_hand(0.1), right_hand_landmarks=[], pose_landmarks=_pose()
    )
    both_hands = SimpleNamespace(
        left_hand_landmarks=_hand(0.2),
        right_hand_landmarks=_hand(0.3),
        pose_landmarks=_pose(),
    )
    fake_landmarker = _FakeLandmarker([no_hands, left_only, both_hands])

    with patch("pose_library.extract._make_landmarker", return_value=fake_landmarker):
        sequence, stats = extract_pose_sequence(
            video_path, gloss="test", source="unit-test"
        )

    assert stats.total_frames_decoded == 3
    assert stats.frames_kept == 2
    assert stats.dropped_frame_indices == [0]
    assert len(sequence.frames) == 2
    assert fake_landmarker.closed


def test_extract_uppercases_gloss_and_sets_source(tmp_path, model_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=1)
    both_hands = SimpleNamespace(
        left_hand_landmarks=_hand(0.1),
        right_hand_landmarks=_hand(0.2),
        pose_landmarks=_pose(),
    )
    with patch(
        "pose_library.extract._make_landmarker",
        return_value=_FakeLandmarker([both_hands]),
    ):
        sequence, _ = extract_pose_sequence(
            video_path, gloss="help", source="wlasl:x:1"
        )

    assert sequence.gloss == "HELP"
    assert sequence.source == "wlasl:x:1"


def test_extract_zero_fills_missing_hand_but_keeps_frame(tmp_path, model_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=1)
    right_only = SimpleNamespace(
        left_hand_landmarks=[], right_hand_landmarks=_hand(0.5), pose_landmarks=_pose()
    )
    with patch(
        "pose_library.extract._make_landmarker",
        return_value=_FakeLandmarker([right_only]),
    ):
        sequence, stats = extract_pose_sequence(video_path, gloss="x", source="s")

    assert stats.frames_kept == 1
    frame = sequence.frames[0]
    left_wrist_idx = sequence.landmark_names.index("left_hand_wrist")
    right_wrist_idx = sequence.landmark_names.index("right_hand_wrist")
    assert frame[left_wrist_idx] == (0.0, 0.0, 0.0)
    assert frame[right_wrist_idx] == (0.5, 0.5, 0.5)


def test_extract_landmark_names_shape_matches_frame_length(tmp_path, model_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=1)
    both_hands = SimpleNamespace(
        left_hand_landmarks=_hand(0.1),
        right_hand_landmarks=_hand(0.2),
        pose_landmarks=_pose(),
    )
    with patch(
        "pose_library.extract._make_landmarker",
        return_value=_FakeLandmarker([both_hands]),
    ):
        sequence, _ = extract_pose_sequence(video_path, gloss="x", source="s")

    assert len(LANDMARK_NAMES) == 2 * len(HAND_LANDMARK_NAMES) + len(POSE_SUBSET_NAMES)
    assert len(sequence.landmark_names) == len(sequence.frames[0])


def test_extract_raises_file_not_found_for_missing_video(tmp_path, model_path):
    with pytest.raises(FileNotFoundError):
        extract_pose_sequence(
            tmp_path / "missing.mp4", gloss="x", source="s", model_path=model_path
        )


def test_extract_raises_file_not_found_for_missing_model(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, num_frames=1)
    with pytest.raises(FileNotFoundError):
        extract_pose_sequence(
            video_path, gloss="x", source="s", model_path=tmp_path / "missing.task"
        )
