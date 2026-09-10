"""Tests for pose_library.lookup.get_pose_sequence.

Uses a small fixture directory instead of the real (gitignored,
network-derived) pose library, per this project's testing convention of not
depending on real I/O in unit tests.
"""

import json

import pytest

from pose_library.lookup import get_pose_sequence
from pose_library.types import PoseSequence


@pytest.fixture()
def poses_dir(tmp_path):
    sequence = PoseSequence(
        gloss="HELP",
        fps=25.0,
        landmark_names=("left_hand_wrist",),
        frames=[[(0.1, 0.2, 0.3)]],
        source="wlasl:aslbrick:69364",
    )
    (tmp_path / "help.json").write_text(
        json.dumps(sequence.to_dict()), encoding="utf-8"
    )
    return tmp_path


def test_get_pose_sequence_hit_returns_pose_sequence(poses_dir):
    result = get_pose_sequence("help", poses_dir=poses_dir)
    assert isinstance(result, PoseSequence)
    assert result.gloss == "HELP"
    assert result.frames == [[(0.1, 0.2, 0.3)]]


def test_get_pose_sequence_is_case_insensitive(poses_dir):
    assert get_pose_sequence("HELP", poses_dir=poses_dir) is not None
    assert get_pose_sequence("HeLp", poses_dir=poses_dir) is not None


def test_get_pose_sequence_strips_whitespace(poses_dir):
    assert get_pose_sequence("  help  ", poses_dir=poses_dir) is not None


def test_get_pose_sequence_miss_returns_none_not_raise(poses_dir):
    result = get_pose_sequence("this-word-does-not-exist", poses_dir=poses_dir)
    assert result is None


def test_get_pose_sequence_raises_typeerror_on_non_string(poses_dir):
    with pytest.raises(TypeError):
        get_pose_sequence(123, poses_dir=poses_dir)


def test_get_pose_sequence_raises_valueerror_on_empty_string(poses_dir):
    with pytest.raises(ValueError):
        get_pose_sequence("", poses_dir=poses_dir)


def test_get_pose_sequence_raises_valueerror_on_whitespace_only(poses_dir):
    with pytest.raises(ValueError):
        get_pose_sequence("   ", poses_dir=poses_dir)
