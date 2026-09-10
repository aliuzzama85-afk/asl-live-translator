"""Gloss-word-to-pose lookup: `get_pose_sequence()`.

See `pose_library/PLAN.md` Section 4 for the design rationale (why a miss
returns `None` instead of raising, why lookup is a plain per-word file read
rather than an in-memory index).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pose_library import config
from pose_library.types import PoseSequence

logger = logging.getLogger(__name__)


def get_pose_sequence(
    gloss_word: str, poses_dir: Path = config.POSES_DIR
) -> PoseSequence | None:
    """Looks up a single gloss word's pose sequence.

    Args:
        gloss_word: A single ASL gloss token (case-insensitive; internally
            normalized to lowercase to match stored filenames).
        poses_dir: Directory containing one `<word>.json` file per gloss
            word. Defaults to the project's real pose library location;
            overridable for tests.

    Returns:
        The word's `PoseSequence` if present in the library, otherwise
        `None`. A miss is an expected, common outcome (see PLAN.md Section
        1/5's finding that this library's ~150-250 word vocabulary is a
        small curated subset, not full ASL) -- not an error condition -- so
        callers should treat `None` as "try the fingerspelling fallback,"
        not as a bug signal.

    Raises:
        TypeError: If `gloss_word` is not a string.
        ValueError: If `gloss_word` is empty or whitespace-only. This is a
            caller bug (e.g. an upstream tokenization error), not a
            vocabulary gap, so it fails loudly rather than returning `None`.
    """
    if not isinstance(gloss_word, str):
        raise TypeError(f"gloss_word must be a str, got {type(gloss_word).__name__}")
    if not gloss_word.strip():
        raise ValueError("gloss_word must not be empty or whitespace-only")

    path = Path(poses_dir) / f"{gloss_word.strip().lower()}.json"
    if not path.is_file():
        logger.debug("No pose sequence for %r (looked in %s)", gloss_word, path)
        return None

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return PoseSequence.from_dict(data)
