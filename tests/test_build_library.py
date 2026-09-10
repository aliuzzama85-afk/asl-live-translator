"""Tests for pose_library.build_library.

`download_word`, `extract_pose_sequence`, and manifest I/O are all mocked --
this module is pure orchestration glue, so its tests verify the wiring
(what gets called with what, how failures at each stage map to a
`BuildResult`) rather than re-testing those collaborators' own behavior
(covered in their own test files).
"""

import json
from unittest.mock import patch

from pose_library.build_library import build_word
from pose_library.download import DownloadResult
from pose_library.extract import ExtractionStats
from pose_library.types import PoseSequence


def _fake_sequence() -> PoseSequence:
    return PoseSequence(
        gloss="HELP",
        fps=30.0,
        landmark_names=("left_hand_wrist",),
        frames=[[(0.1, 0.2, 0.3)]],
        source="wlasl:aslbrick:69364",
    )


def test_build_word_success_writes_pose_json_and_manifest_entry(tmp_path):
    poses_dir = tmp_path / "poses"
    download_result = DownloadResult(
        gloss="help",
        success=True,
        trimmed_path=tmp_path / "help.mp4",
        source="aslbrick",
        video_id="69364",
        instance_id=1,
    )
    stats = ExtractionStats(
        total_frames_decoded=75, frames_kept=57, dropped_frame_indices=[0, 1, 2]
    )

    with (
        patch("pose_library.build_library.config.POSES_DIR", poses_dir),
        patch(
            "pose_library.build_library.download_word", return_value=download_result
        ) as mock_download,
        patch(
            "pose_library.build_library.extract_pose_sequence",
            return_value=(_fake_sequence(), stats),
        ) as mock_extract,
        patch(
            "pose_library.build_library.manifest_module.load_manifest",
            return_value={},
        ),
        patch("pose_library.build_library.manifest_module.save_manifest") as mock_save,
    ):
        result = build_word("help", gloss_index={"help": {}})

    assert result.success
    assert result.stage == ""
    assert result.frames_kept == 57
    assert result.frames_dropped == 3

    mock_download.assert_called_once_with("help", {"help": {}})
    mock_extract.assert_called_once_with(
        download_result.trimmed_path, gloss="help", source="wlasl:aslbrick:69364"
    )

    pose_path = poses_dir / "help.json"
    assert pose_path.is_file()
    assert json.loads(pose_path.read_text(encoding="utf-8"))["gloss"] == "HELP"

    mock_save.assert_called_once()
    saved_library = mock_save.call_args[0][0]
    assert saved_library["help"]["wlasl_source"] == "aslbrick"
    assert saved_library["help"]["frames_kept"] == 57


def test_build_word_reports_download_failure_without_calling_extract(tmp_path):
    download_result = DownloadResult(
        gloss="water", success=False, error="DNS resolution failed"
    )

    with (
        patch("pose_library.build_library.download_word", return_value=download_result),
        patch("pose_library.build_library.extract_pose_sequence") as mock_extract,
    ):
        result = build_word("water", gloss_index={})

    assert not result.success
    assert result.stage == "download"
    assert result.error == "DNS resolution failed"
    mock_extract.assert_not_called()


def test_build_word_reports_extract_exception_without_writing_manifest(tmp_path):
    download_result = DownloadResult(
        gloss="foo",
        success=True,
        trimmed_path=tmp_path / "foo.mp4",
        source="aslbrick",
        video_id="1",
        instance_id=1,
    )

    with (
        patch("pose_library.build_library.download_word", return_value=download_result),
        patch(
            "pose_library.build_library.extract_pose_sequence",
            side_effect=ValueError("no frames could be decoded"),
        ),
        patch("pose_library.build_library.manifest_module.save_manifest") as mock_save,
    ):
        result = build_word("foo", gloss_index={})

    assert not result.success
    assert result.stage == "extract"
    assert "no frames could be decoded" in result.error
    mock_save.assert_not_called()


def test_build_word_treats_zero_frames_kept_as_extract_failure(tmp_path):
    download_result = DownloadResult(
        gloss="foo",
        success=True,
        trimmed_path=tmp_path / "foo.mp4",
        source="aslbrick",
        video_id="1",
        instance_id=1,
    )
    stats = ExtractionStats(
        total_frames_decoded=10, frames_kept=0, dropped_frame_indices=list(range(10))
    )

    with (
        patch("pose_library.build_library.download_word", return_value=download_result),
        patch(
            "pose_library.build_library.extract_pose_sequence",
            return_value=(_fake_sequence(), stats),
        ),
        patch("pose_library.build_library.manifest_module.save_manifest") as mock_save,
    ):
        result = build_word("foo", gloss_index={})

    assert not result.success
    assert result.stage == "extract"
    assert "no hand detected" in result.error
    mock_save.assert_not_called()


def test_build_words_processes_each_word_independently(tmp_path):
    ok = DownloadResult(gloss="a", success=True, trimmed_path=tmp_path / "a.mp4")
    bad = DownloadResult(gloss="b", success=False, error="not in WLASL")
    stats = ExtractionStats(
        total_frames_decoded=5, frames_kept=5, dropped_frame_indices=[]
    )

    def _fake_download_word(word, gloss_index):
        return ok if word == "a" else bad

    with (
        patch("pose_library.build_library.load_wlasl_metadata", return_value=[]),
        patch("pose_library.build_library.build_gloss_index", return_value={}),
        patch(
            "pose_library.build_library.download_word",
            side_effect=_fake_download_word,
        ),
        patch(
            "pose_library.build_library.extract_pose_sequence",
            return_value=(_fake_sequence(), stats),
        ),
        patch("pose_library.build_library.config.POSES_DIR", tmp_path / "poses"),
        patch(
            "pose_library.build_library.manifest_module.load_manifest",
            return_value={},
        ),
        patch("pose_library.build_library.manifest_module.save_manifest"),
    ):
        from pose_library.build_library import build_words

        results = build_words(["a", "b"])

    assert [r.word for r in results] == ["a", "b"]
    assert results[0].success
    assert not results[1].success
    assert results[1].stage == "download"
