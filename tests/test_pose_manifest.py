"""Tests for pose_library.manifest."""

from pose_library.manifest import (
    build_entry,
    compute_quality_flags,
    load_manifest,
    save_manifest,
)


def test_load_manifest_returns_empty_dict_when_missing(tmp_path):
    assert load_manifest(tmp_path / "does-not-exist.json") == {}


def test_save_and_load_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    entry = build_entry(
        word="help",
        source="aslbrick",
        video_id="69364",
        instance_id=1,
        total_frames_decoded=75,
        frames_kept=57,
        dropped_frame_indices=[0, 1, 2],
    )
    save_manifest({"help": entry}, path)

    loaded = load_manifest(path)
    assert loaded == {"help": entry}


def test_build_entry_includes_license_tag():
    entry = build_entry(
        word="help",
        source="aslbrick",
        video_id="69364",
        instance_id=1,
        total_frames_decoded=10,
        frames_kept=10,
        dropped_frame_indices=[],
    )
    assert "no commercial use" in entry["license"]
    assert entry["filename"] == "help.json"


def test_build_entry_marks_clean_extraction_as_not_low_confidence():
    entry = build_entry(
        word="help",
        source="aslbrick",
        video_id="69364",
        instance_id=1,
        total_frames_decoded=75,
        frames_kept=57,
        dropped_frame_indices=[
            0,
            1,
            2,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
        ],
    )
    assert entry["low_confidence"] is False
    assert entry["quality_notes"] is None


def test_compute_quality_flags_clean_boundary_only_drops_is_confident():
    # All drops are a leading run (0-2) plus a trailing run (60-74) -- a
    # normal "not yet in frame" / "already left frame" pattern.
    low_confidence, notes = compute_quality_flags(
        frames_kept=57,
        total_frames_decoded=75,
        dropped_frame_indices=[0, 1, 2, *range(60, 75)],
    )
    assert low_confidence is False
    assert notes is None


def test_compute_quality_flags_flags_low_frame_count():
    low_confidence, notes = compute_quality_flags(
        frames_kept=21, total_frames_decoded=30, dropped_frame_indices=[0, 1, 2]
    )
    assert low_confidence is True
    assert "low frame count" in notes


def test_compute_quality_flags_flags_low_retention():
    low_confidence, notes = compute_quality_flags(
        frames_kept=36,
        total_frames_decoded=99,
        dropped_frame_indices=list(range(63)),
    )
    assert low_confidence is True
    assert "low retention" in notes


def test_compute_quality_flags_flags_many_interior_gaps():
    # Drops scattered through the middle of the clip, not just the edges --
    # e.g. intermittent hand-tracking loss (occlusion, face proximity).
    low_confidence, notes = compute_quality_flags(
        frames_kept=39,
        total_frames_decoded=68,
        dropped_frame_indices=list(range(15, 44)),
    )
    assert low_confidence is True
    assert "mid-clip tracking gaps" in notes


def test_compute_quality_flags_reports_all_tripped_reasons():
    # Leading run (0-2), a large interior block (10-60) that touches neither
    # boundary, and a trailing run (65-67) -- kept frames are 3-9 and 61-64
    # (11 total), so all three thresholds trip at once, like the real
    # "phone" extraction this function was written to catch.
    dropped = [0, 1, 2, *range(10, 61), 65, 66, 67]
    low_confidence, notes = compute_quality_flags(
        frames_kept=11, total_frames_decoded=68, dropped_frame_indices=dropped
    )
    assert low_confidence is True
    assert "low frame count" in notes
    assert "low retention" in notes
    assert "mid-clip tracking gaps" in notes


def _letter_entry(issues):
    from pose_library.manifest import build_letter_entry

    return build_letter_entry(
        letter="J",
        kind="motion",
        signing_hand="right",
        source="fingerspelling:self-recorded",
        recording_total_frames=75,
        frames_with_hand=70,
        segment_start_frame=20,
        total_frames_decoded=30,
        frames_kept=29,
        dropped_frame_indices=[7],
        issues=issues,
    )


def test_build_letter_entry_shares_wlasl_timeline_fields_and_drops_wlasl_ids():
    entry = _letter_entry([])
    # The fields reconstructTimeline() reads, with the same meaning as WLASL's.
    assert entry["total_frames_decoded"] == 30
    assert entry["frames_kept"] == 29
    assert entry["dropped_frame_indices"] == [7]
    assert entry["filename"] == "j.json"
    assert entry["recording"]["filename"] == "raw/j.mp4"
    assert not any(key.startswith("wlasl_") for key in entry)
    assert entry["low_confidence"] is False
    assert entry["quality_notes"] is None


def test_build_letter_entry_flags_quality_issues():
    entry = _letter_entry(["a", "b"])
    assert entry["low_confidence"] is True
    assert entry["quality_notes"] == "a; b"
