"""Tests for pose_library.types.PoseSequence."""

import dataclasses

import numpy as np
import pytest

from pose_library.types import PoseSequence


def _sample_sequence() -> PoseSequence:
    return PoseSequence(
        gloss="HELLO",
        fps=25.0,
        landmark_names=("left_hand_wrist", "right_hand_wrist"),
        frames=[
            [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            [(0.7, 0.8, 0.9), (1.0, 1.1, 1.2)],
        ],
        source="wlasl:aslbrick:12345",
    )


def test_to_dict_is_json_serializable_and_round_trips():
    import json

    sequence = _sample_sequence()
    serialized = json.dumps(sequence.to_dict())
    restored = PoseSequence.from_dict(json.loads(serialized))
    assert restored == sequence


def test_to_dict_converts_tuples_to_lists():
    data = _sample_sequence().to_dict()
    assert isinstance(data["frames"][0][0], list)
    assert isinstance(data["landmark_names"], list)


def test_from_dict_converts_lists_back_to_tuples():
    sequence = _sample_sequence()
    restored = PoseSequence.from_dict(sequence.to_dict())
    assert isinstance(restored.frames[0][0], tuple)
    assert isinstance(restored.landmark_names, tuple)


def test_as_array_has_expected_shape():
    sequence = _sample_sequence()
    array = sequence.as_array()
    assert array.shape == (2, 2, 3)
    assert np.allclose(array[0, 0], [0.1, 0.2, 0.3])


def test_pose_sequence_is_frozen():
    sequence = _sample_sequence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sequence.gloss = "OTHER"
