"""Webcam recorder for the fingerspelling alphabet: one short clip per letter.

Usage (from the repo root, with the venv active):

    python -m pose_library.record_fingerspelling            # resume at first unrecorded letter
    python -m pose_library.record_fingerspelling --letters jz   # re-record specific letters
    python -m pose_library.record_fingerspelling --camera 1     # a different webcam

Keys in the window: SPACE record, R re-record, N next letter, P previous,
Q / ESC quit. Each take is saved to
`pose_library/fingerspelling/raw/<letter>.mp4` and immediately checked with
the real MediaPipe Holistic extraction, so a take where the hand wasn't
tracked is caught on the spot. See `pose_library/FINGERSPELLING_PLAN.md`
Section 1 for the format decisions (640x480, measured fps, `mp4v`, saved
unmirrored).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pose_library import config
from pose_library.fingerspelling import (
    ALPHABET,
    LETTER_MIN_HAND_RETENTION,
    is_motion_letter,
    normalize_letter,
    raw_video_path,
    recording_seconds,
)

# Matches the WLASL trimmed clips the extraction pipeline was validated on
# (checked: 640x480, ~30fps) -- FINGERSPELLING_PLAN.md Section 1.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
REQUESTED_FPS = 30.0
COUNTDOWN_SECONDS = 3

# mp4v, not H.264: OpenCV's avc1 writer needs a separate OpenH264 DLL on
# Windows and fails to open without it (FINGERSPELLING_PLAN.md Section 1).
_FOURCC = "mp4v"

_WINDOW = "Fingerspelling recorder"
_KEY_ESC = 27


def parse_letters(arg: str | None) -> list[str]:
    """Parses the `--letters` argument into an ordered, de-duplicated list.

    Args:
        arg: Letters to record, e.g. `"jz"`, `"j,z"`, or `"J Z"`; `None` or
            empty means the whole alphabet.

    Returns:
        Lowercase letters in alphabet order, each once.

    Raises:
        ValueError: If `arg` contains anything other than letters,
            commas, and spaces.
    """
    if not arg:
        return list(ALPHABET)
    wanted = set()
    for char in arg:
        if char in ", ":
            continue
        wanted.add(normalize_letter(char))
    if not wanted:
        raise ValueError(f"no letters in {arg!r}")
    return [letter for letter in ALPHABET if letter in wanted]


def first_unrecorded_index(
    letters: list[str], raw_dir: Path = config.FINGERSPELLING_RAW_DIR
) -> int:
    """Finds where to resume a session: the first letter with no recording.

    Args:
        letters: The letters in this session, in order.
        raw_dir: Directory of raw recordings.

    Returns:
        Index into `letters` of the first letter whose `<letter>.mp4`
        doesn't exist, or 0 if every letter is already recorded (so a
        complete set opens at the start for review/re-takes).
    """
    for i, letter in enumerate(letters):
        if not raw_video_path(letter, raw_dir).is_file():
            return i
    return 0


def measured_fps(timestamps: list[float]) -> float:
    """Computes the real frame rate from capture timestamps.

    Webcams often report a nominal 30fps while delivering fewer frames; since
    `extract.py` trusts the file header's fps for timing, the file must carry
    the rate frames actually arrived at.

    Args:
        timestamps: Monotonic capture times (seconds), one per frame.

    Returns:
        `(n - 1) / elapsed`, or `REQUESTED_FPS` if there are fewer than two
        frames or no elapsed time to measure over.
    """
    if len(timestamps) < 2:
        return REQUESTED_FPS
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return REQUESTED_FPS
    return (len(timestamps) - 1) / elapsed


def write_video_atomic(frames: list[np.ndarray], fps: float, path: Path) -> None:
    """Writes BGR frames to an `.mp4`, replacing `path` only once complete.

    Writes to a temp file in the same directory, then renames over `path`,
    so an interrupted or failed write never replaces a good earlier take
    with a truncated one.

    Args:
        frames: BGR `uint8` frames, all the same size.
        fps: Frame rate to record in the file header.
        path: Destination `.mp4` path (parent directories are created).

    Raises:
        ValueError: If `frames` is empty or `fps` isn't positive.
        OSError: If the video writer can't be opened.
    """
    if not frames:
        raise ValueError("no frames to write")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer = cv2.VideoWriter(
            str(tmp_path), cv2.VideoWriter_fourcc(*_FOURCC), fps, (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"could not open a video writer for {tmp_path}")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@dataclass
class TakeCheck:
    """Result of checking a saved take with the real extraction.

    Attributes:
        frames_with_hand: Frames where at least one hand was detected.
        total_frames: Frames decoded from the take.
        message: One-line summary for the recorder window.
        ok: Whether the take looks usable.
    """

    frames_with_hand: int
    total_frames: int
    message: str
    ok: bool


def check_take(
    video_path: Path, letter: str, model_path: Path = config.HOLISTIC_MODEL_PATH
) -> TakeCheck | None:
    """Runs the real Holistic extraction on a take to confirm the hand is tracked.

    Args:
        video_path: The saved take.
        letter: Which letter it is.
        model_path: The Holistic Landmarker model bundle.

    Returns:
        A `TakeCheck`, or `None` if the model file isn't present (checking
        is then skipped, not fatal -- the build step reports it later).
    """
    if not Path(model_path).is_file():
        return None
    # Imported here so the recorder's pure helpers (and their tests) don't
    # pay for loading MediaPipe.
    from pose_library.extract import extract_pose_sequence

    _, stats = extract_pose_sequence(
        video_path, gloss=letter, source="check", model_path=model_path
    )
    total = stats.total_frames_decoded
    kept = stats.frames_kept
    ok = total > 0 and kept / total >= LETTER_MIN_HAND_RETENTION
    verdict = "OK" if ok else "LOW - press R to re-record"
    return TakeCheck(
        frames_with_hand=kept,
        total_frames=total,
        message=f"Hand detected in {kept}/{total} frames - {verdict}",
        ok=ok,
    )


def _draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2) -> None:
    # Dark outline first so text stays readable over any background.
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3
    )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _overlay(frame, letter, index, count, status_lines, status_color=(255, 255, 255)):
    """Mirrors a camera frame for display and draws the recorder's HUD."""
    view = cv2.flip(frame, 1)
    height, width = view.shape[:2]
    _draw_text(view, letter.upper(), (20, 90), scale=3.0, thickness=6)
    _draw_text(view, f"{index + 1}/{count}", (width - 110, 40))
    if is_motion_letter(letter):
        _draw_text(view, "MOTION LETTER - draw it once", (20, 130), scale=0.6)
    for i, line in enumerate(status_lines):
        _draw_text(
            view,
            line,
            (20, height - 20 - 30 * (len(status_lines) - 1 - i)),
            scale=0.6,
            color=status_color,
        )
    return view


def _capture(cap, seconds: float, letter, index, count):
    """Records `seconds` of frames, showing a REC indicator. Returns (frames, times)."""
    frames, times = [], []
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok:
            break
        frames.append(frame)
        times.append(now)
        elapsed = now - start
        prompt = (
            "REC - draw the letter now"
            if is_motion_letter(letter)
            else "REC - hold it steady"
        )
        view = _overlay(frame, letter, index, count, [prompt], (60, 60, 255))
        bar = int((view.shape[1] - 40) * min(1.0, elapsed / seconds))
        cv2.rectangle(view, (20, 150), (20 + bar, 162), (60, 60, 255), -1)
        cv2.imshow(_WINDOW, view)
        cv2.waitKey(1)
        if elapsed >= seconds:
            break
    return frames, times


def _countdown(cap, letter, index, count) -> bool:
    """Shows a countdown; returns False if the camera stopped delivering frames."""
    end = time.monotonic() + COUNTDOWN_SECONDS
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            return False
        remaining = int(end - time.monotonic()) + 1
        view = _overlay(frame, letter, index, count, ["Form the handshape now..."])
        _draw_text(
            view,
            str(remaining),
            (view.shape[1] // 2 - 30, view.shape[0] // 2),
            4.0,
            (87, 200, 255),
            8,
        )
        cv2.imshow(_WINDOW, view)
        cv2.waitKey(1)
    return True


def run_session(
    letters: list[str],
    camera_index: int = 0,
    raw_dir: Path = config.FINGERSPELLING_RAW_DIR,
    check: bool = True,
) -> list[str]:
    """Runs the interactive recording window until the user quits.

    Args:
        letters: Letters to step through, in order.
        camera_index: OpenCV camera index.
        raw_dir: Where takes are saved.
        check: Whether to run the real extraction on each take.

    Returns:
        The letters recorded (saved at least once) during this session.

    Raises:
        SystemExit: If the camera can't be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera_index}. Close other apps using the "
            "webcam, or try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

    index = first_unrecorded_index(letters, raw_dir)
    recorded: list[str] = []
    review: list[str] | None = None  # set after a take, until the user moves on
    review_color = (255, 255, 255)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("The camera stopped delivering frames.")
            letter = letters[index]
            if review is None:
                existing = raw_video_path(letter, raw_dir).is_file()
                lines = [
                    "SPACE record   N next   P prev   Q quit",
                    (
                        "(already recorded - SPACE re-records)"
                        if existing
                        else "One hand, other hand down"
                    ),
                ]
                view = _overlay(frame, letter, index, len(letters), lines)
            else:
                view = _overlay(
                    frame, letter, index, len(letters), review, review_color
                )
            cv2.imshow(_WINDOW, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), _KEY_ESC):
                break
            if key == ord("n") or (review is not None and key == ord(" ")):
                if index == len(letters) - 1 and review is not None:
                    break  # finished the last letter
                index = min(index + 1, len(letters) - 1)
                review = None
            elif key == ord("p"):
                index = max(index - 1, 0)
                review = None
            elif key == ord(" ") or key == ord("r"):
                if not _countdown(cap, letter, index, len(letters)):
                    raise SystemExit("The camera stopped delivering frames.")
                frames, times = _capture(
                    cap, recording_seconds(letter), letter, index, len(letters)
                )
                path = raw_video_path(letter, raw_dir)
                write_video_atomic(frames, measured_fps(times), path)
                recorded.append(letter)
                review = [f"Saved {path.name} ({len(frames)} frames)"]
                review_color = (255, 255, 255)
                if check:
                    view = _overlay(
                        frames[-1],
                        letter,
                        index,
                        len(letters),
                        ["Checking hand tracking..."],
                    )
                    cv2.imshow(_WINDOW, view)
                    cv2.waitKey(1)
                    result = check_take(path, letter)
                    if result is None:
                        review.append("(model not found - tracking check skipped)")
                    else:
                        review.append(result.message)
                        review_color = (120, 230, 120) if result.ok else (80, 80, 255)
                review.append("SPACE/N next   R re-record   P prev   Q quit")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return recorded


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: records fingerspelling letters from a webcam."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--letters", help="Letters to record, e.g. 'jz' (default: all 26)."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default 0)."
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the hand-tracking check after each take.",
    )
    args = parser.parse_args(argv)

    try:
        letters = parse_letters(args.letters)
    except ValueError as exc:
        parser.error(str(exc))

    recorded = run_session(letters, camera_index=args.camera, check=not args.no_check)
    have = [ltr for ltr in ALPHABET if raw_video_path(ltr).is_file()]
    missing = [ltr for ltr in ALPHABET if ltr not in have]
    print(f"Recorded this session: {', '.join(recorded) or 'none'}")
    print(f"Letters with a recording: {len(have)}/26")
    if missing:
        print(f"Still missing: {' '.join(missing)}")
    print("Next: python -m pose_library.build_fingerspelling")
    sys.exit(0)


if __name__ == "__main__":
    main()
