"""Tests for pose_library.build_fingerspelling.

`extract_pose_sequence` is mocked (the `test_pose_extract.py` /
`test_build_library.py` pattern) -- no real MediaPipe model and no real
recordings are needed. The raw `.mp4` files here are empty placeholders;
only their existence matters to the orchestration under test.
"""

import json
from unittest.mock import patch

from pose_library.build_fingerspelling import build_alphabet
from pose_library.extract import LANDMARK_NAMES, ExtractionStats
from pose_library.fingerspelling import FINGERSPELLING_SOURCE, hand_indices
from pose_library.manifest import FINGERSPELLING_LICENSE_TAG, load_manifest
from pose_library.types import PoseSequence

HANDS = hand_indices(LANDMARK_NAMES)


def _still_right_hand_sequence(n=40):
    frame = [(0.0, 0.0, 0.0)] * len(LANDMARK_NAMES)
    for i in HANDS["right"]:
        frame[i] = (0.5, 0.5, 0.0)
    return PoseSequence(
        gloss="A",
        fps=30.0,
        landmark_names=LANDMARK_NAMES,
        frames=[list(frame) for _ in range(n)],
        source=FINGERSPELLING_SOURCE,
    )


def _dirs(tmp_path, recorded="ab"):
    raw = tmp_path / "raw"
    raw.mkdir()
    for letter in recorded:
        (raw / f"{letter}.mp4").write_bytes(b"placeholder")
    return raw, tmp_path / "poses"


def _ok_extraction(*_args, **_kwargs):
    stats = ExtractionStats(
        total_frames_decoded=40, frames_kept=40, dropped_frame_indices=[]
    )
    return _still_right_hand_sequence(), stats


def test_builds_every_recorded_letter_into_json_and_manifest(tmp_path):
    raw, poses = _dirs(tmp_path, "ab")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=_ok_extraction,
    ) as mock_extract:
        results = build_alphabet(raw_dir=raw, poses_dir=poses)

    assert [r.letter for r in results] == ["a", "b"]
    assert all(r.success for r in results)
    assert mock_extract.call_count == 2

    manifest = load_manifest(poses / "manifest.json")
    assert set(manifest) == {"a", "b"}
    entry = manifest["a"]
    assert entry["filename"] == "a.json"
    assert entry["kind"] == "static"
    assert entry["license"] == FINGERSPELLING_LICENSE_TAG
    assert entry["recording"]["total_frames_decoded"] == 40
    assert entry["low_confidence"] is False

    stored = json.loads((poses / "a.json").read_text(encoding="utf-8"))
    assert stored["gloss"] == "A"
    assert stored["source"] == FINGERSPELLING_SOURCE
    assert (
        len(stored["frames"]) == entry["frames_kept"] == entry["total_frames_decoded"]
    )


def test_an_unrecorded_letter_is_reported_not_raised(tmp_path):
    raw, poses = _dirs(tmp_path, "a")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=_ok_extraction,
    ):
        results = build_alphabet(["a", "q"], raw_dir=raw, poses_dir=poses)

    assert results[0].success
    assert not results[1].success and results[1].error == "not recorded"
    assert set(load_manifest(poses / "manifest.json")) == {"a"}


def test_a_failed_rebuild_removes_the_stale_entry_and_json(tmp_path):
    raw, poses = _dirs(tmp_path, "a")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=_ok_extraction,
    ):
        build_alphabet(raw_dir=raw, poses_dir=poses)
    assert (poses / "a.json").is_file()

    # Re-recorded, but the new take has no hand in any frame.
    empty = PoseSequence(
        gloss="A", fps=30.0, landmark_names=LANDMARK_NAMES, frames=[], source="x"
    )
    no_hand = ExtractionStats(
        total_frames_decoded=5, frames_kept=0, dropped_frame_indices=[0, 1, 2, 3, 4]
    )
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        return_value=(empty, no_hand),
    ):
        results = build_alphabet(raw_dir=raw, poses_dir=poses)

    assert not results[0].success
    assert "no hand detected" in results[0].error
    assert not (poses / "a.json").exists()
    assert load_manifest(poses / "manifest.json") == {}


def test_extraction_exceptions_are_captured(tmp_path):
    raw, poses = _dirs(tmp_path, "a")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=ValueError("No frames could be decoded"),
    ):
        results = build_alphabet(raw_dir=raw, poses_dir=poses)
    assert not results[0].success
    assert "No frames could be decoded" in results[0].error


def test_building_one_letter_keeps_the_others(tmp_path):
    raw, poses = _dirs(tmp_path, "ab")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=_ok_extraction,
    ):
        build_alphabet(raw_dir=raw, poses_dir=poses)
        build_alphabet(["b"], raw_dir=raw, poses_dir=poses)
    assert set(load_manifest(poses / "manifest.json")) == {"a", "b"}


def test_no_recordings_writes_no_manifest(tmp_path):
    raw, poses = _dirs(tmp_path, "")
    assert build_alphabet(raw_dir=raw, poses_dir=poses) == []
    assert not (poses / "manifest.json").exists()


def test_losing_the_last_letter_removes_the_manifest_file(tmp_path):
    raw, poses = _dirs(tmp_path, "a")
    with patch(
        "pose_library.build_fingerspelling.extract_pose_sequence",
        side_effect=_ok_extraction,
    ):
        build_alphabet(raw_dir=raw, poses_dir=poses)
    (raw / "a.mp4").unlink()
    build_alphabet(["a"], raw_dir=raw, poses_dir=poses)
    assert not (poses / "manifest.json").exists()
