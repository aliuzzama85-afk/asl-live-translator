"""Builds the fingerspelling alphabet: raw letter recordings -> pose JSON + manifest.

For each letter with a recording at `pose_library/fingerspelling/raw/<letter>.mp4`,
runs the same MediaPipe Holistic extraction every WLASL word goes through
(`extract.py`, unchanged), cuts the stored segment (`fingerspelling.py`), and
writes `pose_library/fingerspelling/poses/<letter>.json` plus an entry in
`pose_library/fingerspelling/poses/manifest.json`. Mirrors
`build_library.py`'s conventions: per-item results, expected failures reported
rather than raised. See `pose_library/FINGERSPELLING_PLAN.md` Section 2.

Usage (from the repo root, with the venv active):

    python -m pose_library.build_fingerspelling              # every recorded letter
    python -m pose_library.build_fingerspelling --letters jz # just these
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from pose_library import config
from pose_library import manifest as manifest_module
from pose_library.extract import extract_pose_sequence
from pose_library.fingerspelling import (
    ALPHABET,
    FINGERSPELLING_SOURCE,
    build_letter_segment,
    pose_json_path,
    raw_video_path,
)
from pose_library.record_fingerspelling import parse_letters

logger = logging.getLogger(__name__)


@dataclass
class LetterBuildResult:
    """Outcome of building one letter.

    Attributes:
        letter: The letter (lowercase).
        success: Whether a pose JSON and manifest entry were written.
        error: Short failure reason, if not successful.
        frames_kept: Frames stored in the pose JSON, if successful.
        low_confidence: Whether the entry was flagged, if successful.
        quality_notes: Why it was flagged, if it was.
    """

    letter: str
    success: bool
    error: str | None = None
    frames_kept: int | None = None
    low_confidence: bool = False
    quality_notes: str | None = None


def build_letter(
    letter: str,
    library: dict,
    raw_dir: Path = config.FINGERSPELLING_RAW_DIR,
    poses_dir: Path = config.FINGERSPELLING_POSES_DIR,
    model_path: Path = config.HOLISTIC_MODEL_PATH,
) -> LetterBuildResult:
    """Extracts one letter's recording and records it in `library`.

    On failure the letter's old pose JSON and manifest entry (if any) are
    removed, so the manifest never describes a different take than the one
    sitting in `raw/`.

    Args:
        letter: The letter to build.
        library: The in-memory manifest dict, updated in place.
        raw_dir: Directory of raw recordings.
        poses_dir: Directory to write the pose JSON into.
        model_path: The Holistic Landmarker model bundle.

    Returns:
        A `LetterBuildResult`. Never raises for expected failures (no
        recording, decode failure, no hand tracked).
    """
    video_path = raw_video_path(letter, raw_dir)
    json_path = pose_json_path(letter, poses_dir)

    def fail(reason: str) -> LetterBuildResult:
        library.pop(letter, None)
        if json_path.is_file():
            json_path.unlink()
        return LetterBuildResult(letter=letter, success=False, error=reason)

    if not video_path.is_file():
        return fail("not recorded")

    try:
        sequence, stats = extract_pose_sequence(
            video_path,
            gloss=letter,
            source=FINGERSPELLING_SOURCE,
            model_path=model_path,
        )
        segment = build_letter_segment(
            letter, sequence, stats.total_frames_decoded, stats.dropped_frame_indices
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the result, not raised
        logger.warning("Building %r failed: %s", letter, exc)
        return fail(str(exc))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(segment.sequence.to_dict(), f)

    entry = manifest_module.build_letter_entry(
        letter=letter,
        kind=segment.kind,
        signing_hand=segment.signing_hand,
        source=FINGERSPELLING_SOURCE,
        recording_total_frames=stats.total_frames_decoded,
        frames_with_hand=segment.frames_with_hand,
        segment_start_frame=segment.segment_start_frame,
        total_frames_decoded=segment.total_frames_decoded,
        frames_kept=len(segment.sequence.frames),
        dropped_frame_indices=segment.dropped_frame_indices,
        issues=segment.issues,
    )
    library[letter] = entry
    return LetterBuildResult(
        letter=letter,
        success=True,
        frames_kept=entry["frames_kept"],
        low_confidence=entry["low_confidence"],
        quality_notes=entry["quality_notes"],
    )


def build_alphabet(
    letters: list[str] | None = None,
    raw_dir: Path = config.FINGERSPELLING_RAW_DIR,
    poses_dir: Path = config.FINGERSPELLING_POSES_DIR,
    manifest_path: Path | None = None,
    model_path: Path = config.HOLISTIC_MODEL_PATH,
) -> list[LetterBuildResult]:
    """Builds the requested letters and saves the manifest.

    Args:
        letters: Letters to build; `None` means every letter that has a
            recording in `raw_dir`.
        raw_dir: Directory of raw recordings.
        poses_dir: Directory for pose JSONs.
        manifest_path: Manifest location; defaults to
            `<poses_dir>/manifest.json`.
        model_path: The Holistic Landmarker model bundle.

    Returns:
        One result per letter built, in alphabet order.
    """
    manifest_path = Path(manifest_path or Path(poses_dir) / "manifest.json")
    if letters is None:
        letters = [ltr for ltr in ALPHABET if raw_video_path(ltr, raw_dir).is_file()]

    library = manifest_module.load_manifest(manifest_path)
    results = [
        build_letter(
            ltr, library, raw_dir=raw_dir, poses_dir=poses_dir, model_path=model_path
        )
        for ltr in letters
    ]
    # An empty manifest is never written (and a now-empty one is removed):
    # the frontend treats a missing manifest as "alphabet not recorded yet",
    # which is the honest state -- an empty `{}` would instead read as "the
    # alphabet exists but every letter is missing".
    if library:
        manifest_module.save_manifest(library, manifest_path)
    elif manifest_path.is_file():
        manifest_path.unlink()
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: builds the fingerspelling alphabet from raw recordings."""
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--letters", help="Letters to build, e.g. 'jz' (default: all recorded)."
    )
    args = parser.parse_args(argv)

    if not config.HOLISTIC_MODEL_PATH.is_file():
        print(
            f"MediaPipe model not found at {config.HOLISTIC_MODEL_PATH}.\n"
            f"Download it from {config.HOLISTIC_MODEL_URL} and save it there."
        )
        sys.exit(1)

    try:
        letters = parse_letters(args.letters) if args.letters else None
    except ValueError as exc:
        parser.error(str(exc))

    results = build_alphabet(letters)
    if not results:
        print(
            f"No recordings found in {config.FINGERSPELLING_RAW_DIR}.\n"
            "Record them first: python -m pose_library.record_fingerspelling"
        )
        sys.exit(1)

    for r in results:
        if r.success:
            flag = f"  LOW-CONFIDENCE: {r.quality_notes}" if r.low_confidence else ""
            print(f"{r.letter.upper()}: OK ({r.frames_kept} frames){flag}")
        else:
            print(f"{r.letter.upper()}: FAILED ({r.error})")

    built = manifest_module.load_manifest(config.FINGERSPELLING_MANIFEST_PATH)
    missing = [ltr for ltr in ALPHABET if ltr not in built]
    print(f"\nAlphabet: {len(built)}/26 letters built.")
    if missing:
        print(f"Missing: {' '.join(missing)}")
    sys.exit(0 if all(r.success for r in results) else 1)


if __name__ == "__main__":
    main()
