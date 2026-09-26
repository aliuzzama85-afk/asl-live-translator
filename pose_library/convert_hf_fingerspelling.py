"""Converts the MIT-licensed sid220/asl-now-fingerspelling dataset into letter poses.

Replaces "record every letter yourself" as the fingerspelling data source for
the 24 static letters. For each letter A-Z except J and Z, it picks one
representative real sample from the dataset (the per-letter medoid) and
writes it in exactly the format `build_fingerspelling.py` produces for a
recorded letter: `pose_library/fingerspelling/poses/<letter>.json` (a
`PoseSequence`) plus a `manifest.json` entry. The player can't tell the two
sources apart except by provenance. See
`pose_library/FINGERSPELLING_PLAN.md` Section 2b for the inspection findings
behind every decision here, and
`pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`
for the license.

Usage (from the repo root, with the venv active):

    python -m pose_library.convert_hf_fingerspelling

Downloads the pinned dataset revision into the gitignored
`pose_library/data/asl_now_fingerspelling/` cache on first run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pose_library import config
from pose_library import manifest as manifest_module
from pose_library.extract import LANDMARK_NAMES
from pose_library.fingerspelling import (
    ALPHABET,
    FINGERSPELLING_SOURCE,
    LETTER_HOLD_SECONDS,
    MOTION_LETTERS,
    hand_indices,
    pose_json_path,
)
from pose_library.types import PoseSequence

logger = logging.getLogger(__name__)

DATASET_SOURCE = f"hf:{config.ASL_NOW_DATASET_ID}"

HOLD_FPS = 30.0
"""The dataset has no fps (each sample is one still frame). 30fps matches
the output rate stitchTimelines resamples everything onto, so a held pose
needs no resampling."""

HOLD_FRAMES = round(LETTER_HOLD_SECONDS * HOLD_FPS)
"""A sample is held for the same 0.4s window recorded static letters use."""

HAND_LANDMARK_COUNT = 21
_WRIST, _MIDDLE_MCP = 0, 9

# Placed in the right-hand slot. The dataset has no handedness label and it
# couldn't be recovered reliably (FINGERSPELLING_PLAN.md Section 2b); the
# renderer draws both slots identically, so the slot is only a convention.
_HAND_SLOT = "right"

MOTION_LETTER_REASON = (
    "motion letter; the dataset has only single static frames (no motion data), "
    "so it is not converted rather than faked"
)


@dataclass
class Sample:
    """One dataset sample: a single frame of 21 hand landmarks.

    Attributes:
        path: Path relative to the dataset root, e.g. `"A/<uuid>.json"`.
        points: 21 `(x, y, z)` tuples in MediaPipe hand-landmark order.
    """

    path: str
    points: list[tuple[float, float, float]]


@dataclass
class LoadReport:
    """Per-letter bookkeeping from loading and filtering samples.

    Attributes:
        total: Files found for the letter.
        invalid: Files skipped as malformed (wrong shape, non-numeric, NaN).
        out_of_frame: Valid samples skipped because a landmark falls outside
            the image (`x` or `y` outside `[0, 1]`) -- the hand was partly out
            of frame, so its shape may be clipped or extrapolated.
    """

    total: int = 0
    invalid: list[str] = field(default_factory=list)
    out_of_frame: list[str] = field(default_factory=list)


def parse_sample(data: object) -> list[tuple[float, float, float]] | None:
    """Validates one sample's JSON against the dataset card's documented format.

    Args:
        data: A parsed sample file.

    Returns:
        The 21 landmarks as `(x, y, z)` tuples, or `None` if the sample isn't
        exactly a list of 21 `{x, y, z}` objects with finite numeric values.
    """
    if not isinstance(data, list) or len(data) != HAND_LANDMARK_COUNT:
        return None
    points = []
    for point in data:
        if not isinstance(point, dict) or set(point) != {"x", "y", "z"}:
            return None
        values = [point[k] for k in ("x", "y", "z")]
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            return None
        points.append(tuple(float(v) for v in values))
    return points


def is_in_frame(points: list[tuple[float, float, float]]) -> bool:
    """Whether every landmark lies inside the image (`x`, `y` in `[0, 1]`).

    Args:
        points: 21 landmarks.

    Returns:
        False if any landmark was extrapolated past the frame edge.
    """
    return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y, _ in points)


def load_letter_samples(
    dataset_dir: Path, letter: str
) -> tuple[list[Sample], LoadReport]:
    """Loads, validates, and in-frame-filters one letter's samples.

    Args:
        dataset_dir: The downloaded dataset root (with `A/`...`Z/` folders).
        letter: The letter to load.

    Returns:
        `(usable_samples, report)`, samples sorted by path for determinism.
    """
    report = LoadReport()
    samples = []
    for path in sorted((Path(dataset_dir) / letter.upper()).glob("*.json")):
        report.total += 1
        rel = f"{letter.upper()}/{path.name}"
        try:
            points = parse_sample(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            points = None
        if points is None:
            report.invalid.append(rel)
        elif not is_in_frame(points):
            report.out_of_frame.append(rel)
        else:
            samples.append(Sample(path=rel, points=points))
    return samples, report


def normalize_shape(points: list[tuple[float, float, float]]) -> np.ndarray:
    """Removes position and size so samples compare by handshape alone.

    Translates the wrist to the origin and scales by the wrist-to-middle-MCP
    distance. Orientation is deliberately kept: for some letters it is part of
    the sign (G/H point sideways, P/Q point down). `z` is dropped because the
    card describes it only as roughly x-scaled relative depth, which is too
    loose to weigh equally in a shape distance.

    Args:
        points: 21 landmarks.

    Returns:
        A `(21, 2)` array.
    """
    xy = np.array([(x, y) for x, y, _ in points], dtype=float)
    xy -= xy[_WRIST]
    scale = np.linalg.norm(xy[_MIDDLE_MCP])
    return xy / scale if scale > 0 else xy


def select_representative(samples: list[Sample]) -> tuple[int, float]:
    """Picks the per-letter medoid: the sample most typical of the rest.

    A per-landmark centroid/median isn't used. The dataset mixes hands, and
    averaging a hand with a mirrored one produces a shape nobody signed. The
    medoid is always one real, whole sample, and it falls inside the densest
    group, so a mis-signed or badly tracked outlier can't be chosen.
    Concretely: the sample with the lowest *median* distance (mean per-landmark
    Euclidean distance, after `normalize_shape`) to every other sample. Using
    the median rather than the sum keeps a few extreme outliers from skewing
    the choice.

    Args:
        samples: A letter's usable samples (at least one).

    Returns:
        `(index, median_distance)` of the chosen sample. Ties go to the
        earliest (by path), for determinism.

    Raises:
        ValueError: If `samples` is empty.
    """
    if not samples:
        raise ValueError("no samples to select from")
    if len(samples) == 1:
        return 0, 0.0
    shapes = np.stack([normalize_shape(s.points) for s in samples])
    diffs = shapes[:, None, :, :] - shapes[None, :, :, :]
    distances = np.linalg.norm(diffs, axis=-1).mean(axis=-1)  # (n, n)
    np.fill_diagonal(distances, np.nan)
    medians = np.nanmedian(distances, axis=1)
    best = int(np.argmin(medians))
    return best, float(medians[best])


def sample_to_frame(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Places a 21-landmark hand into this project's 48-landmark frame.

    MediaPipe's Web Hand Landmarker and the Python Holistic Landmarker share
    the 21-point hand topology and index order (the same `HAND_LANDMARK_NAMES`
    `extract.py` uses), with x/y normalized to the image and z relative to the
    wrist, so the values carry over unchanged. The other hand and the 6
    pose-subset points are the `(0, 0, 0)` "not detected" sentinel, the same
    as a one-handed recorded letter.

    Args:
        points: 21 landmarks.

    Returns:
        48 `(x, y, z)` points in `LANDMARK_NAMES` order.
    """
    frame = [(0.0, 0.0, 0.0)] * len(LANDMARK_NAMES)
    for i, point in zip(hand_indices(LANDMARK_NAMES)[_HAND_SLOT], points, strict=True):
        frame[i] = point
    return frame


@dataclass
class ConvertResult:
    """Outcome for one letter.

    Attributes:
        letter: The letter (lowercase).
        converted: Whether a pose JSON and manifest entry were written.
        reason: Why not, if not converted.
        sample: The chosen sample's path, if converted.
        usable: Samples that passed validation and the in-frame filter.
        total: Samples in the dataset for this letter.
    """

    letter: str
    converted: bool
    reason: str | None = None
    sample: str | None = None
    usable: int = 0
    total: int = 0


def convert_letter(
    letter: str, dataset_dir: Path, library: dict, poses_dir: Path
) -> ConvertResult:
    """Converts one letter, updating `library` in place.

    Never overwrites a self-recorded letter: recording one is an explicit
    choice to override the dataset (FINGERSPELLING_PLAN.md Section 2b).

    Args:
        letter: The letter.
        dataset_dir: The downloaded dataset root.
        library: The in-memory fingerspelling manifest.
        poses_dir: Where to write `<letter>.json`.

    Returns:
        A `ConvertResult`.
    """
    letter = letter.lower()
    if letter in MOTION_LETTERS:
        return ConvertResult(letter, False, reason=MOTION_LETTER_REASON)
    if library.get(letter, {}).get("source") == FINGERSPELLING_SOURCE:
        return ConvertResult(letter, False, reason="self-recorded version kept")

    samples, report = load_letter_samples(dataset_dir, letter)
    if not samples:
        return ConvertResult(
            letter, False, reason="no usable samples", total=report.total
        )

    index, median_distance = select_representative(samples)
    chosen = samples[index]
    sequence = PoseSequence(
        gloss=letter.upper(),
        fps=HOLD_FPS,
        landmark_names=LANDMARK_NAMES,
        frames=[sample_to_frame(chosen.points)] * HOLD_FRAMES,
        source=f"{DATASET_SOURCE}:{chosen.path}",
    )
    json_path = pose_json_path(letter, poses_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(sequence.to_dict(), f)

    library[letter] = manifest_module.build_dataset_letter_entry(
        letter=letter,
        source=DATASET_SOURCE,
        source_url=config.ASL_NOW_SOURCE_URL,
        license_tag=manifest_module.ASL_NOW_LICENSE_TAG,
        dataset_sample={
            "dataset": config.ASL_NOW_DATASET_ID,
            "revision": config.ASL_NOW_REVISION,
            "path": chosen.path,
            "selection": "medoid (lowest median shape distance to the letter's other samples)",
            "median_shape_distance": round(median_distance, 4),
            "samples_total": report.total,
            "samples_usable": len(samples),
            "samples_excluded_out_of_frame": len(report.out_of_frame),
            "samples_excluded_invalid": len(report.invalid),
        },
        frames_kept=HOLD_FRAMES,
        issues=[],
    )
    return ConvertResult(
        letter, True, sample=chosen.path, usable=len(samples), total=report.total
    )


def convert_dataset(
    dataset_dir: Path = config.ASL_NOW_CACHE_DIR,
    poses_dir: Path = config.FINGERSPELLING_POSES_DIR,
    manifest_path: Path | None = None,
    letters: list[str] | None = None,
) -> list[ConvertResult]:
    """Converts the requested letters (default: all 26) and saves the manifest.

    Merges into any existing manifest instead of replacing it, so
    self-recorded letters (J and Z in practice) keep their entries.

    Args:
        dataset_dir: The downloaded dataset root.
        poses_dir: Where to write pose JSONs.
        manifest_path: Defaults to `<poses_dir>/manifest.json`.
        letters: Letters to convert; `None` means all 26 (J/Z are then
            reported as not converted).

    Returns:
        One `ConvertResult` per letter, in alphabet order.
    """
    manifest_path = Path(manifest_path or Path(poses_dir) / "manifest.json")
    library = manifest_module.load_manifest(manifest_path)
    results = [
        convert_letter(ltr, dataset_dir, library, poses_dir)
        for ltr in (letters or ALPHABET)
    ]
    if library:
        manifest_module.save_manifest(library, manifest_path)
    return results


def download_dataset(dataset_dir: Path = config.ASL_NOW_CACHE_DIR) -> Path:
    """Downloads the pinned dataset revision (a no-op if already cached).

    Args:
        dataset_dir: Where to put it (gitignored).

    Returns:
        The dataset root.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=config.ASL_NOW_DATASET_ID,
            repo_type="dataset",
            revision=config.ASL_NOW_REVISION,
            local_dir=dataset_dir,
        )
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: downloads (if needed) and converts the dataset."""
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use the existing cache only; fail if it's missing.",
    )
    args = parser.parse_args(argv)

    dataset_dir = config.ASL_NOW_CACHE_DIR
    if not args.no_download:
        dataset_dir = download_dataset(dataset_dir)
    elif not dataset_dir.is_dir():
        print(f"Dataset cache not found at {dataset_dir}; run without --no-download.")
        sys.exit(1)

    results = convert_dataset(dataset_dir)
    for r in results:
        if r.converted:
            print(
                f"{r.letter.upper()}: OK  {r.sample}  ({r.usable}/{r.total} usable samples)"
            )
        else:
            print(f"{r.letter.upper()}: not converted ({r.reason})")
    built = manifest_module.load_manifest(config.FINGERSPELLING_MANIFEST_PATH)
    missing = [ltr for ltr in ALPHABET if ltr not in built]
    print(f"\nAlphabet: {len(built)}/26 letters available.")
    if missing:
        print(
            f"Missing: {' '.join(missing)} -- record with "
            f"python -m pose_library.record_fingerspelling --letters {''.join(missing)}"
        )


if __name__ == "__main__":
    main()
