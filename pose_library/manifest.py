"""Reads/writes `pose_library/data/poses/manifest.json`.

A small index recording, per gloss word: its pose JSON filename, WLASL
provenance (source site, video_id, instance_id), a license tag, and
extraction stats (frame counts / dropped-frame gaps) -- satisfying the
C-UDA Section 3.1.1 attribution-retention requirement discussed in
`pose_library/PLAN.md` Section 1/3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pose_library import config

logger = logging.getLogger(__name__)

# The WLASL README's own Disclaimer section (PLAN.md Section 1): academic/
# computational use only, no commercial usage. Recorded per-entry so it
# travels with the data even if entries are inspected in isolation.
WLASL_LICENSE_TAG = (
    "C-UDA-1.0; WLASL README: academic/computational use only, no commercial use"
)


def load_manifest(path: Path = config.MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    """Loads the manifest, or an empty dict if it doesn't exist yet.

    Args:
        path: Path to `manifest.json`.

    Returns:
        A dict mapping gloss word (lowercase) to its manifest entry.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_manifest(
    manifest: dict[str, dict[str, Any]], path: Path = config.MANIFEST_PATH
) -> None:
    """Writes the manifest to disk as indented, sorted JSON.

    Args:
        manifest: The full word -> entry manifest dict.
        path: Path to write `manifest.json` to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# Thresholds behind `compute_quality_flags`, picked from the real distribution
# across the first full 118-word build (see PROJECT_STATUS.md's Stage 4 quality
# review): each has a natural gap in the actual data rather than being a round
# guessed number.
LOW_FRAME_COUNT_THRESHOLD = 25
"""Frames kept below this is well under 1s of motion at WLASL's ~30fps clips."""

LOW_RETENTION_THRESHOLD = 0.42
"""Fraction of decoded frames actually kept; the real distribution has a clear
gap between 0.418 (the highest flagged case) and 0.427 (the lowest unflagged
one)."""

HIGH_INTERIOR_GAPS_THRESHOLD = 10
"""Dropped frames that fall strictly between the leading/trailing dropped
runs (i.e. mid-clip, not just a "not yet in frame" / "already left frame"
boundary pattern). The real distribution drops sharply after 10 (29, 22, 10,
then 6, 5, 5, 5, 4, ...), so >=10 catches genuinely unstable mid-sign tracking
without flagging the single stray mid-clip frame that's common even in clean
extractions."""


def compute_quality_flags(
    frames_kept: int,
    total_frames_decoded: int,
    dropped_frame_indices: list[int],
) -> tuple[bool, str | None]:
    """Flags a word's extraction as low-confidence if it looks data-poor.

    Distinguishes two things that look similar but aren't: a naturally short
    or one-handed sign (fine) versus an extraction where MediaPipe lost hand
    tracking for a large or scattered portion of the clip (risky to animate
    from). See `pose_library/PLAN.md` Section 2's frame-dropping policy --
    this is the follow-up quality pass over the honest gaps it records.

    Args:
        frames_kept: Frames retained in the stored `PoseSequence`.
        total_frames_decoded: Total frames read from the trimmed clip.
        dropped_frame_indices: 0-indexed frame numbers dropped (no hand
            detected in either hand).

    Returns:
        A `(low_confidence, quality_notes)` tuple. `quality_notes` is `None`
        when `low_confidence` is `False`, otherwise a semicolon-joined,
        human-readable explanation of every threshold that tripped.
    """
    dropped_set = set(dropped_frame_indices)
    lead = 0
    while lead in dropped_set:
        lead += 1
    trail = 0
    while total_frames_decoded and (total_frames_decoded - 1 - trail) in dropped_set:
        trail += 1
    boundary = set(range(lead)) | set(
        range(total_frames_decoded - trail, total_frames_decoded)
    )
    interior_gaps = len(dropped_set - boundary)
    retention = frames_kept / total_frames_decoded if total_frames_decoded else 1.0

    reasons = []
    if frames_kept < LOW_FRAME_COUNT_THRESHOLD:
        reasons.append(
            f"low frame count ({frames_kept} kept, "
            f"threshold <{LOW_FRAME_COUNT_THRESHOLD})"
        )
    if retention < LOW_RETENTION_THRESHOLD:
        reasons.append(
            f"low retention ({retention:.1%} of {total_frames_decoded} decoded "
            f"frames kept, threshold <{LOW_RETENTION_THRESHOLD:.0%})"
        )
    if interior_gaps >= HIGH_INTERIOR_GAPS_THRESHOLD:
        reasons.append(
            f"{interior_gaps} mid-clip tracking gaps, not just leading/trailing "
            f"(threshold >={HIGH_INTERIOR_GAPS_THRESHOLD})"
        )

    if not reasons:
        return False, None
    return True, "; ".join(reasons)


def build_entry(
    word: str,
    source: str,
    video_id: str,
    instance_id: int | None,
    total_frames_decoded: int,
    frames_kept: int,
    dropped_frame_indices: list[int],
) -> dict[str, Any]:
    """Builds one manifest entry for a successfully extracted word.

    Args:
        word: The gloss word (lowercase), used as the manifest key.
        source: WLASL "source" site (e.g. "handspeak").
        video_id: WLASL "video_id" for the instance used.
        instance_id: WLASL "instance_id" for the instance used, if known.
        total_frames_decoded: Total frames read from the trimmed clip.
        frames_kept: Frames retained in the stored `PoseSequence`.
        dropped_frame_indices: 0-indexed frame numbers dropped (no hand
            detected in either hand) -- an honest gap signal, per PLAN.md
            Section 2.

    Returns:
        A JSON-serializable manifest entry dict. Includes `low_confidence`
        and `quality_notes` (see `compute_quality_flags`) so Stage 5 can
        skip or deprioritize data-poor extractions without deleting them.
    """
    low_confidence, quality_notes = compute_quality_flags(
        frames_kept=frames_kept,
        total_frames_decoded=total_frames_decoded,
        dropped_frame_indices=dropped_frame_indices,
    )
    return {
        "filename": f"{word.lower()}.json",
        "wlasl_source": source,
        "wlasl_video_id": video_id,
        "wlasl_instance_id": instance_id,
        "license": WLASL_LICENSE_TAG,
        "total_frames_decoded": total_frames_decoded,
        "frames_kept": frames_kept,
        "dropped_frame_indices": dropped_frame_indices,
        "low_confidence": low_confidence,
        "quality_notes": quality_notes,
    }
