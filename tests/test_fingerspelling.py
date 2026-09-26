"""Tests for pose_library.fingerspelling (letter rules and segment selection).

All pose data here is small and synthetic -- arithmetic-friendly points, not
real handshapes. The real 26 letters haven't been recorded yet; these tests
pin down the selection logic that will run on them.
"""

import pytest

from pose_library.extract import LANDMARK_NAMES
from pose_library.fingerspelling import (
    FINGERSPELLING_SOURCE,
    LETTER_HOLD_SECONDS,
    MOTION_PAD_FRAMES,
    build_letter_segment,
    choose_signing_hand,
    frame_displacement,
    hand_indices,
    is_motion_letter,
    kept_original_indices,
    normalize_letter,
    pose_json_path,
    raw_video_path,
    recording_seconds,
    select_motion_span,
    select_static_window,
    zero_other_hand,
)
from pose_library.types import PoseSequence

HANDS = hand_indices(LANDMARK_NAMES)
ZERO = (0.0, 0.0, 0.0)


def _frame(right_offset=None, left_offset=None):
    """A full 48-landmark frame. A hand is present when its offset isn't None;
    every landmark of that hand sits at (0.5 + offset, 0.5 + offset)."""
    frame = [ZERO] * len(LANDMARK_NAMES)
    for side, offset in (("right", right_offset), ("left", left_offset)):
        if offset is None:
            continue
        for i in HANDS[side]:
            frame[i] = (0.5 + offset, 0.5 + offset, 0.0)
    return frame


def _sequence(frames, fps=30.0):
    return PoseSequence(
        gloss="X", fps=fps, landmark_names=LANDMARK_NAMES, frames=frames, source="test"
    )


# --- letter rules / paths ---------------------------------------------------


def test_normalize_letter_lowercases_and_strips():
    assert normalize_letter(" Q ") == "q"


@pytest.mark.parametrize("bad", ["", "ab", "1", "é", "-"])
def test_normalize_letter_rejects_non_letters(bad):
    with pytest.raises(ValueError):
        normalize_letter(bad)


def test_normalize_letter_rejects_non_strings():
    with pytest.raises(TypeError):
        normalize_letter(3)


def test_only_j_and_z_are_motion_letters():
    motion = [c for c in "abcdefghijklmnopqrstuvwxyz" if is_motion_letter(c)]
    assert motion == ["j", "z"]


def test_motion_letters_get_a_longer_recording():
    assert recording_seconds("J") > recording_seconds("a")


def test_paths_are_lowercase_per_letter(tmp_path):
    assert raw_video_path("B", tmp_path) == tmp_path / "b.mp4"
    assert pose_json_path("B", tmp_path) == tmp_path / "b.json"


# --- small helpers ----------------------------------------------------------


def test_hand_indices_resolve_21_landmarks_per_hand_by_name():
    assert len(HANDS["left"]) == 21 and len(HANDS["right"]) == 21
    assert not set(HANDS["left"]) & set(HANDS["right"])


def test_frame_displacement_ignores_undetected_points():
    a = _frame(right_offset=0.0)
    b = _frame(right_offset=0.03)
    assert frame_displacement(a, b, HANDS["right"]) == pytest.approx(
        (2 * 0.03**2) ** 0.5
    )
    assert frame_displacement(a, b, HANDS["left"]) is None


def test_kept_original_indices_skips_drops():
    assert kept_original_indices(6, [0, 3]) == [1, 2, 4, 5]


def test_choose_signing_hand_picks_the_most_tracked_hand():
    frames = [
        _frame(left_offset=0.0),
        _frame(left_offset=0.0),
        _frame(right_offset=0.0),
    ]
    assert choose_signing_hand(frames, HANDS) == "left"


def test_zero_other_hand_removes_the_resting_hand_only():
    frames = [_frame(right_offset=0.0, left_offset=0.1)]
    out = zero_other_hand(frames, "right", HANDS)
    assert all(out[0][i] == ZERO for i in HANDS["left"])
    assert all(out[0][i] != ZERO for i in HANDS["right"])
    assert frames[0][HANDS["left"][0]] != ZERO  # input not mutated


# --- static hold window -----------------------------------------------------


def test_static_window_picks_the_stillest_run():
    # Frames 0-4 wobble, frames 5-9 are perfectly still.
    offsets = [0.0, 0.05, 0.0, 0.05, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02]
    frames = [_frame(right_offset=o) for o in offsets]
    start, end, full = select_static_window(frames, list(range(10)), 4, HANDS["right"])
    assert full
    assert (start, end) == (5, 8)


def test_static_window_never_straddles_a_dropped_frame():
    # Perfectly still throughout, but original index 3 is missing: runs are
    # [0,1,2] and [4..8]; a 4-frame window must come from the second run.
    frames = [_frame(right_offset=0.0)] * 8
    originals = [0, 1, 2, 4, 5, 6, 7, 8]
    start, end, full = select_static_window(frames, originals, 4, HANDS["right"])
    assert full
    assert originals[start] >= 4
    assert originals[end] - originals[start] == 3


def test_static_window_falls_back_to_longest_run_when_none_is_long_enough():
    frames = [_frame(right_offset=0.0)] * 5
    originals = [0, 1, 3, 4, 5]
    start, end, full = select_static_window(frames, originals, 4, HANDS["right"])
    assert not full
    assert (start, end) == (2, 4)


def test_static_window_rejects_empty_input():
    with pytest.raises(ValueError):
        select_static_window([], [], 4, HANDS["right"])


# --- motion span ------------------------------------------------------------


def test_motion_span_trims_still_lead_in_and_lead_out():
    # 10 still frames, 6 moving frames, 10 still frames.
    offsets = [0.0] * 10 + [0.03 * k for k in range(1, 7)] + [0.18] * 10
    frames = [_frame(right_offset=o) for o in offsets]
    start, end, found = select_motion_span(
        frames, list(range(len(frames))), HANDS["right"]
    )
    assert found
    # First moving step is 9->10, last is 14->15; padded by MOTION_PAD_FRAMES.
    assert start == 9 - MOTION_PAD_FRAMES
    assert end == 15 + MOTION_PAD_FRAMES


def test_motion_span_reports_no_motion_and_keeps_everything():
    frames = [_frame(right_offset=0.0)] * 6
    start, end, found = select_motion_span(frames, list(range(6)), HANDS["right"])
    assert not found
    assert (start, end) == (0, 5)


# --- build_letter_segment ---------------------------------------------------


def test_static_letter_segment_is_a_gap_free_hold_window():
    fps = 30.0
    window = round(LETTER_HOLD_SECONDS * fps)
    frames = [_frame(right_offset=0.0, left_offset=0.2)] * 40
    segment = build_letter_segment("a", _sequence(frames, fps), 40, [])

    assert segment.kind == "static"
    assert segment.signing_hand == "right"
    assert len(segment.sequence.frames) == window
    assert segment.total_frames_decoded == window
    assert segment.dropped_frame_indices == []
    assert segment.sequence.gloss == "A"
    assert segment.sequence.source == FINGERSPELLING_SOURCE
    assert segment.issues == []
    # The resting left hand is zero-filled in the stored frames.
    assert all(segment.sequence.frames[0][i] == ZERO for i in HANDS["left"])


def test_motion_letter_segment_keeps_interior_drops_segment_relative():
    offsets = [0.0] * 8 + [0.03 * k for k in range(1, 9)] + [0.24] * 8
    frames = [_frame(right_offset=o) for o in offsets]
    # Original frame 12 was dropped by extraction (no hand at all).
    kept = frames[:12] + frames[13:]
    segment = build_letter_segment("z", _sequence(kept), 24, [12])

    assert segment.kind == "motion"
    assert segment.issues == []
    rel = 12 - segment.segment_start_frame
    assert segment.dropped_frame_indices == [rel]
    assert len(segment.sequence.frames) == segment.total_frames_decoded - 1


def test_segment_flags_poor_hand_tracking():
    # 40 decoded frames, but only 10 had a hand.
    frames = [_frame(right_offset=0.0)] * 10
    segment = build_letter_segment("b", _sequence(frames), 40, list(range(10, 40)))
    assert any("tracked in 25%" in issue for issue in segment.issues)


def test_segment_flags_a_motion_letter_that_did_not_move():
    frames = [_frame(right_offset=0.0)] * 30
    segment = build_letter_segment("j", _sequence(frames), 30, [])
    assert "no hand motion detected for a motion letter" in segment.issues


def test_segment_rejects_mismatched_stats():
    with pytest.raises(ValueError, match="mismatch"):
        build_letter_segment("a", _sequence([_frame(right_offset=0.0)] * 3), 10, [])


def test_segment_rejects_a_take_with_no_hand_at_all():
    with pytest.raises(ValueError):
        build_letter_segment("a", _sequence([]), 5, [0, 1, 2, 3, 4])
