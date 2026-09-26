"""Fingerspelling alphabet: letter rules, file layout, and segment selection.

Pure logic shared by the recording tool (`record_fingerspelling.py`) and the
build step (`build_fingerspelling.py`) -- no camera, no MediaPipe, so all of
it is unit-testable. See `pose_library/FINGERSPELLING_PLAN.md` Section 2 for
why a letter's stored pose is a short *segment* of its raw recording (a
stable hold window for static letters, the motion span for J/Z) rather than
the whole take.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pose_library import config
from pose_library.types import PoseSequence

ALPHABET: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")

# The two ASL letters drawn as a motion (a hook for J, a "Z" stroke for Z)
# rather than held as a static handshape -- pose_library/PLAN.md Section 5.
MOTION_LETTERS: frozenset[str] = frozenset({"j", "z"})

FINGERSPELLING_SOURCE = "fingerspelling:self-recorded"

# How long each raw take lasts. Static letters only need a steady hold to
# pick a window from; motion letters need room for the whole stroke.
STATIC_RECORD_SECONDS = 2.0
MOTION_RECORD_SECONDS = 2.5

LETTER_HOLD_SECONDS = 0.4
"""Length of the stable window kept per static letter. With stitchTimelines'
200ms transition that's ~0.6s per letter (~1.7 letters/s) -- deliberately
slower than fluent fingerspelling, for readability; FINGERSPELLING_PLAN.md
Section 2."""

# Motion-span detection thresholds, in normalized image units per frame.
# Picked from real WLASL data (FINGERSPELLING_PLAN.md Section 2): a
# near-still hand moves ~0.001-0.005/frame, a moving one ~0.02-0.05.
MOTION_MIN_DISPLACEMENT = 0.008
MOTION_PEAK_FRACTION = 0.25
MOTION_PAD_FRAMES = 2

LETTER_MIN_HAND_RETENTION = 0.8
"""A take where the signing hand is tracked in fewer than this fraction of
frames is flagged low-confidence (and the recorder suggests a re-take)."""

_ZERO_POINT: tuple[float, float, float] = (0.0, 0.0, 0.0)

Point = tuple[float, float, float]
Frame = list[Point]


def normalize_letter(letter: str) -> str:
    """Validates and lowercases a single alphabet letter.

    Args:
        letter: One character, `a`-`z`, either case.

    Returns:
        The lowercase letter.

    Raises:
        TypeError: If `letter` is not a string.
        ValueError: If `letter` isn't exactly one ASCII letter a-z.
    """
    if not isinstance(letter, str):
        raise TypeError(f"letter must be a str, got {type(letter).__name__}")
    normalized = letter.strip().lower()
    if normalized not in ALPHABET:
        raise ValueError(f"not a single letter a-z: {letter!r}")
    return normalized


def is_motion_letter(letter: str) -> bool:
    """Returns whether a letter is signed as a motion (J, Z).

    Args:
        letter: One letter a-z, either case.

    Returns:
        True for J and Z, False for the 24 static letters.
    """
    return normalize_letter(letter) in MOTION_LETTERS


def recording_seconds(letter: str) -> float:
    """Returns how long the recorder should capture for a letter.

    Args:
        letter: One letter a-z, either case.

    Returns:
        `MOTION_RECORD_SECONDS` for J/Z, `STATIC_RECORD_SECONDS` otherwise.
    """
    return MOTION_RECORD_SECONDS if is_motion_letter(letter) else STATIC_RECORD_SECONDS


def raw_video_path(letter: str, raw_dir: Path = config.FINGERSPELLING_RAW_DIR) -> Path:
    """Returns where a letter's raw recording lives.

    Args:
        letter: One letter a-z, either case.
        raw_dir: Directory of raw recordings; overridable for tests.

    Returns:
        `<raw_dir>/<letter>.mp4`, lowercase.
    """
    return Path(raw_dir) / f"{normalize_letter(letter)}.mp4"


def pose_json_path(
    letter: str, poses_dir: Path = config.FINGERSPELLING_POSES_DIR
) -> Path:
    """Returns where a letter's extracted pose JSON lives.

    Args:
        letter: One letter a-z, either case.
        poses_dir: Directory of letter pose JSONs; overridable for tests.

    Returns:
        `<poses_dir>/<letter>.json`, lowercase.
    """
    return Path(poses_dir) / f"{normalize_letter(letter)}.json"


def hand_indices(landmark_names: tuple[str, ...] | list[str]) -> dict[str, list[int]]:
    """Maps each hand to its landmark column indices, resolved by name.

    Args:
        landmark_names: A sequence's own `landmark_names`.

    Returns:
        `{"left": [...], "right": [...]}` -- indices of `left_hand_*` and
        `right_hand_*` landmarks (the pose-subset points belong to neither).
    """
    return {
        side: [
            i
            for i, name in enumerate(landmark_names)
            if name.startswith(f"{side}_hand_")
        ]
        for side in ("left", "right")
    }


def _is_zero(point: Point) -> bool:
    return point[0] == 0 and point[1] == 0 and point[2] == 0


def _hand_detected(frame: Frame, indices: list[int]) -> bool:
    return any(not _is_zero(frame[i]) for i in indices)


def frame_displacement(a: Frame, b: Frame, indices: list[int]) -> float | None:
    """Mean 2D distance the given landmarks moved between two frames.

    Args:
        a: The earlier frame.
        b: The later frame.
        indices: Landmark columns to compare (one hand's indices).

    Returns:
        The mean Euclidean (x, y) displacement over landmarks detected in
        both frames, or `None` if no landmark is detected in both.
    """
    distances = [
        ((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2) ** 0.5
        for i in indices
        if not _is_zero(a[i]) and not _is_zero(b[i])
    ]
    if not distances:
        return None
    return sum(distances) / len(distances)


def kept_original_indices(
    total_frames_decoded: int, dropped_frame_indices: list[int]
) -> list[int]:
    """Recovers each kept frame's original video-frame index.

    Args:
        total_frames_decoded: Frames decoded from the recording.
        dropped_frame_indices: Frames `extract.py` dropped (no hand at all).

    Returns:
        The original indices of the kept frames, in order -- same walk
        `frontend/src/lib/reconstructTimeline.js` does.
    """
    dropped = set(dropped_frame_indices)
    return [i for i in range(total_frames_decoded) if i not in dropped]


def choose_signing_hand(frames: list[Frame], hands: dict[str, list[int]]) -> str:
    """Picks the hand that's doing the fingerspelling.

    Fingerspelling is one-handed; the other hand may still be detected at
    rest in frame. The signing hand is the one detected in the most frames.

    Args:
        frames: The recording's kept frames.
        hands: From `hand_indices`.

    Returns:
        `"left"` or `"right"` (MediaPipe's label). Ties go to `"right"`.
    """
    left = sum(_hand_detected(f, hands["left"]) for f in frames)
    right = sum(_hand_detected(f, hands["right"]) for f in frames)
    return "left" if left > right else "right"


def zero_other_hand(
    frames: list[Frame], keep: str, hands: dict[str, list[int]]
) -> list[Frame]:
    """Zero-fills the non-signing hand with the `(0,0,0)` "not detected" sentinel.

    Args:
        frames: Frames to process (not modified).
        keep: The signing hand, `"left"` or `"right"`.
        hands: From `hand_indices`.

    Returns:
        New frames with the other hand's landmarks set to `(0, 0, 0)`, so a
        resting second hand can't pull the content-fit camera out.
    """
    other = set(hands["right" if keep == "left" else "left"])
    return [
        [_ZERO_POINT if i in other else point for i, point in enumerate(f)]
        for f in frames
    ]


def _consecutive_runs(original_indices: list[int]) -> list[tuple[int, int]]:
    """Splits positions into runs whose original indices are consecutive."""
    runs = []
    start = 0
    for pos in range(1, len(original_indices) + 1):
        if (
            pos == len(original_indices)
            or original_indices[pos] != original_indices[pos - 1] + 1
        ):
            runs.append((start, pos - 1))
            start = pos
    return runs


def select_static_window(
    frames: list[Frame],
    original_indices: list[int],
    window_len: int,
    indices: list[int],
) -> tuple[int, int, bool]:
    """Finds the stillest gap-free run of `window_len` frames.

    Args:
        frames: Usable frames (signing hand detected), in order.
        original_indices: Each frame's original video-frame index.
        window_len: Frames wanted in the hold window.
        indices: The signing hand's landmark columns.

    Returns:
        `(start_pos, end_pos, full_length)` -- inclusive positions into
        `frames`. The window never straddles a dropped frame. If no gap-free
        run is `window_len` long, returns the longest run (earliest on ties)
        with `full_length=False`, for the caller to flag.

    Raises:
        ValueError: If `frames` is empty or `window_len < 1`.
    """
    if not frames:
        raise ValueError("no usable frames to select a window from")
    if window_len < 1:
        raise ValueError("window_len must be >= 1")

    steps = [
        frame_displacement(frames[k], frames[k + 1], indices)
        for k in range(len(frames) - 1)
    ]
    best: tuple[float, int] | None = None
    longest = (0, 0)
    for run_start, run_end in _consecutive_runs(original_indices):
        if run_end - run_start > longest[1] - longest[0]:
            longest = (run_start, run_end)
        for start in range(run_start, run_end - window_len + 2):
            # A step with no comparable landmarks counts as maximal motion, so
            # it's never preferred over a window with real, measured stillness.
            cost = sum(
                1.0 if steps[k] is None else steps[k]
                for k in range(start, start + window_len - 1)
            )
            if best is None or cost < best[0]:
                best = (cost, start)

    if best is None:
        return longest[0], longest[1], False
    return best[1], best[1] + window_len - 1, True


def select_motion_span(
    frames: list[Frame], original_indices: list[int], indices: list[int]
) -> tuple[int, int, bool]:
    """Trims a motion letter's take to the span where the hand actually moves.

    Args:
        frames: Usable frames (signing hand detected), in order.
        original_indices: Each frame's original video-frame index.
        indices: The signing hand's landmark columns.

    Returns:
        `(start_pos, end_pos, motion_found)` -- inclusive positions into
        `frames`, covering every step above `max(MOTION_MIN_DISPLACEMENT,
        MOTION_PEAK_FRACTION * peak)` plus `MOTION_PAD_FRAMES` either side.
        If the hand never moves more than `MOTION_MIN_DISPLACEMENT`, returns
        the whole take with `motion_found=False`, for the caller to flag.

    Raises:
        ValueError: If `frames` is empty.
    """
    if not frames:
        raise ValueError("no usable frames to select a span from")

    per_frame = []
    for k in range(len(frames) - 1):
        d = frame_displacement(frames[k], frames[k + 1], indices)
        gap = original_indices[k + 1] - original_indices[k]
        per_frame.append(0.0 if d is None else d / gap)

    peak = max(per_frame, default=0.0)
    if peak < MOTION_MIN_DISPLACEMENT:
        return 0, len(frames) - 1, False

    threshold = max(MOTION_MIN_DISPLACEMENT, MOTION_PEAK_FRACTION * peak)
    moving = [k for k, d in enumerate(per_frame) if d >= threshold]
    start = max(0, moving[0] - MOTION_PAD_FRAMES)
    end = min(len(frames) - 1, moving[-1] + 1 + MOTION_PAD_FRAMES)
    return start, end, True


@dataclass
class LetterSegment:
    """A letter's stored segment, cut from its raw recording.

    Attributes:
        sequence: The trimmed `PoseSequence` to write as `<letter>.json`.
        kind: `"static"` or `"motion"`.
        signing_hand: `"left"` or `"right"`.
        frames_with_hand: Frames in the *whole recording* where the signing
            hand was tracked.
        segment_start_frame: Original frame index where the segment starts.
        total_frames_decoded: Original frames the segment spans (the same
            "frames in the stored clip" meaning WLASL entries use).
        dropped_frame_indices: Segment-relative frames missing from
            `sequence` (signing hand not tracked). Always `[]` for static
            letters, whose window is gap-free by construction.
        issues: Human-readable quality problems for the manifest.
    """

    sequence: PoseSequence
    kind: str
    signing_hand: str
    frames_with_hand: int
    segment_start_frame: int
    total_frames_decoded: int
    dropped_frame_indices: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def build_letter_segment(
    letter: str,
    sequence: PoseSequence,
    total_frames_decoded: int,
    dropped_frame_indices: list[int],
) -> LetterSegment:
    """Cuts a letter's stored segment out of its full extracted recording.

    Args:
        letter: The letter this recording is of.
        sequence: `extract_pose_sequence`'s output for the raw recording
            (kept frames only).
        total_frames_decoded: From the extraction stats.
        dropped_frame_indices: From the extraction stats.

    Returns:
        The `LetterSegment` to store.

    Raises:
        ValueError: If no frame has any hand detected, or the stats don't
            match the sequence's frame count.
    """
    letter = normalize_letter(letter)
    originals = kept_original_indices(total_frames_decoded, dropped_frame_indices)
    if len(originals) != len(sequence.frames):
        raise ValueError(
            f"extraction stats mismatch: {len(originals)} kept indices vs "
            f"{len(sequence.frames)} frames"
        )
    if not sequence.frames:
        raise ValueError(f"no hand detected in any frame of {letter!r}")

    hands = hand_indices(sequence.landmark_names)
    signing_hand = choose_signing_hand(sequence.frames, hands)
    signing = hands[signing_hand]
    frames = zero_other_hand(sequence.frames, signing_hand, hands)

    usable = [k for k, f in enumerate(frames) if _hand_detected(f, signing)]
    usable_frames = [frames[k] for k in usable]
    usable_originals = [originals[k] for k in usable]

    issues = []
    kind = "motion" if letter in MOTION_LETTERS else "static"
    if kind == "static":
        window_len = max(2, round(LETTER_HOLD_SECONDS * sequence.fps))
        start, end, full = select_static_window(
            usable_frames, usable_originals, window_len, signing
        )
        if not full:
            issues.append(
                f"only {end - start + 1} consecutive tracked frames "
                f"(hold window needs {window_len})"
            )
    else:
        start, end, found = select_motion_span(usable_frames, usable_originals, signing)
        if not found:
            issues.append("no hand motion detected for a motion letter")

    span_start = usable_originals[start]
    span_end = usable_originals[end]
    kept_in_span = set(usable_originals[start : end + 1])
    segment_dropped = [
        i - span_start for i in range(span_start, span_end + 1) if i not in kept_in_span
    ]

    retention = len(usable) / total_frames_decoded
    if retention < LETTER_MIN_HAND_RETENTION:
        issues.append(
            f"signing hand tracked in {retention:.0%} of {total_frames_decoded} "
            f"frames (threshold <{LETTER_MIN_HAND_RETENTION:.0%})"
        )

    trimmed = PoseSequence(
        gloss=letter.upper(),
        fps=sequence.fps,
        landmark_names=sequence.landmark_names,
        frames=usable_frames[start : end + 1],
        source=FINGERSPELLING_SOURCE,
    )
    return LetterSegment(
        sequence=trimmed,
        kind=kind,
        signing_hand=signing_hand,
        frames_with_hand=len(usable),
        segment_start_frame=span_start,
        total_frames_decoded=span_end - span_start + 1,
        dropped_frame_indices=segment_dropped,
        issues=issues,
    )
