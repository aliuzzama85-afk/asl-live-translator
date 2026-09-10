"""Orchestrates download -> extract -> store for a list of gloss words.

This is the end-to-end driver used for the Stage 4 dry run and (once
approved) the full-scale build: for each requested word, downloads/trims
one WLASL video instance (`download.py`), runs MediaPipe Holistic
extraction (`extract.py`), writes `pose_library/data/poses/<word>.json`,
and records provenance/extraction-stats in `manifest.json`
(`manifest.py`). See `pose_library/PLAN.md` Section 6 for the staged
rollout this supports (small dry run first, not the full vocabulary).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

from pose_library import config
from pose_library import manifest as manifest_module
from pose_library.download import download_word
from pose_library.extract import extract_pose_sequence
from pose_library.wlasl_metadata import build_gloss_index, load_wlasl_metadata

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Outcome of building one word's pose sequence end-to-end.

    Attributes:
        word: The gloss word (lowercase).
        success: Whether a pose JSON file was written.
        stage: Which stage failed ("download", "extract", or "" on success).
        error: A short human-readable failure reason, if not successful.
        frames_kept: Number of frames in the stored sequence, if successful.
        frames_dropped: Number of frames dropped (no hand detected), if
            successful.
    """

    word: str
    success: bool
    stage: str = ""
    error: str | None = None
    frames_kept: int | None = None
    frames_dropped: int | None = None


def build_word(word: str, gloss_index: dict) -> BuildResult:
    """Downloads, extracts, and stores one word's pose sequence.

    Args:
        word: The target gloss word.
        gloss_index: A WLASL gloss index, from `wlasl_metadata.build_gloss_index`.

    Returns:
        A `BuildResult` describing the outcome. Never raises for expected
        failure modes (dead link, no hand detected, decode failure) -- all
        are captured in the returned result.
    """
    word = word.lower()
    download_result = download_word(word, gloss_index)
    if not download_result.success:
        return BuildResult(
            word=word, success=False, stage="download", error=download_result.error
        )

    source = f"wlasl:{download_result.source}:{download_result.video_id}"
    try:
        sequence, stats = extract_pose_sequence(
            download_result.trimmed_path, gloss=word, source=source
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in BuildResult, not raised
        logger.warning("Extraction failed for %r: %s", word, exc)
        return BuildResult(word=word, success=False, stage="extract", error=str(exc))

    if stats.frames_kept == 0:
        return BuildResult(
            word=word,
            success=False,
            stage="extract",
            error="no hand detected in any frame",
        )

    pose_path = config.POSES_DIR / f"{word}.json"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    with pose_path.open("w", encoding="utf-8") as f:
        json.dump(sequence.to_dict(), f)

    library = manifest_module.load_manifest()
    library[word] = manifest_module.build_entry(
        word=word,
        source=download_result.source,
        video_id=download_result.video_id,
        instance_id=download_result.instance_id,
        total_frames_decoded=stats.total_frames_decoded,
        frames_kept=stats.frames_kept,
        dropped_frame_indices=stats.dropped_frame_indices,
    )
    manifest_module.save_manifest(library)

    return BuildResult(
        word=word,
        success=True,
        frames_kept=stats.frames_kept,
        frames_dropped=len(stats.dropped_frame_indices),
    )


def build_words(words: list[str]) -> list[BuildResult]:
    """Builds pose sequences for a list of target words.

    Args:
        words: Target gloss words to process.

    Returns:
        One `BuildResult` per word, in the same order.
    """
    metadata = load_wlasl_metadata()
    gloss_index = build_gloss_index(metadata)
    return [build_word(word, gloss_index) for word in words]


def main() -> None:
    """CLI entry point: builds pose sequences for a list of gloss words."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="+", help="Target gloss words to process.")
    args = parser.parse_args()

    results = build_words(args.words)
    for result in results:
        if result.success:
            print(
                f"{result.word}: OK (frames_kept={result.frames_kept}, "
                f"frames_dropped={result.frames_dropped})"
            )
        else:
            print(f"{result.word}: FAILED at {result.stage} ({result.error})")


if __name__ == "__main__":
    main()
