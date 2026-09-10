"""Builds the reproducible target vocabulary for the pose library.

Cross-references the content-word stems in `vocab_augmentation.csv` (Stage
3's hand-authored everyday-communication vocabulary, see
`gloss_model/VOCAB_DIAGNOSIS.md`) against WLASL's real 2,000-gloss metadata
index, and writes the resolved list to `pose_library/data/target_vocab.json`
so it's inspectable and reproducible rather than implicit in code. See
`pose_library/PLAN.md` Section 1 for the full method and rationale.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pose_library import config
from pose_library.wlasl_metadata import (
    build_gloss_index,
    extract_vocab_stems,
    load_wlasl_metadata,
)

logger = logging.getLogger(__name__)


def build_target_vocab(
    vocab_csv_path: Path = config.VOCAB_AUGMENTATION_CSV_PATH,
    wlasl_metadata_path: Path = config.WLASL_METADATA_PATH,
) -> dict[str, list[str]]:
    """Resolves the vocab_augmentation.csv stems against WLASL's gloss list.

    Args:
        vocab_csv_path: Path to the (text, gloss) CSV to extract stems from.
        wlasl_metadata_path: Path to a local copy of `WLASL_v0.3.json`.

    Returns:
        A dict with two keys: "matched" (stems with an exact case-insensitive
        WLASL gloss match, sorted) and "unmatched" (stems with no match,
        sorted) -- the "matched" list is this project's actual target
        vocabulary for pose extraction.
    """
    stems = extract_vocab_stems(vocab_csv_path)
    metadata = load_wlasl_metadata(wlasl_metadata_path)
    gloss_index = build_gloss_index(metadata)

    matched = sorted(stem for stem in stems if stem in gloss_index)
    unmatched = sorted(stem for stem in stems if stem not in gloss_index)
    logger.info(
        "Resolved %d/%d vocab_augmentation.csv stems against WLASL (%d unmatched)",
        len(matched),
        len(stems),
        len(unmatched),
    )
    return {"matched": matched, "unmatched": unmatched}


def save_target_vocab(
    result: dict[str, list[str]], output_path: Path = config.TARGET_VOCAB_PATH
) -> None:
    """Writes the resolved target vocabulary to disk as JSON.

    Args:
        result: The dict returned by `build_target_vocab`.
        output_path: Where to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    logger.info("Wrote target vocabulary to %s", output_path)


def main() -> None:
    """CLI entry point: builds and saves the target vocabulary."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-csv",
        type=Path,
        default=config.VOCAB_AUGMENTATION_CSV_PATH,
        help="Path to the (text, gloss) CSV to extract stems from.",
    )
    parser.add_argument(
        "--wlasl-metadata",
        type=Path,
        default=config.WLASL_METADATA_PATH,
        help="Path to a local copy of WLASL_v0.3.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.TARGET_VOCAB_PATH,
        help="Where to write the resolved target vocabulary JSON.",
    )
    args = parser.parse_args()

    result = build_target_vocab(args.vocab_csv, args.wlasl_metadata)
    save_target_vocab(result, args.output)
    print(f"matched: {len(result['matched'])}, unmatched: {len(result['unmatched'])}")


if __name__ == "__main__":
    main()
