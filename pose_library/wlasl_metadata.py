"""Parsing and querying WLASL's metadata index (`WLASL_v0.3.json`).

WLASL ships a metadata-only JSON file (no video bytes) -- one entry per
gloss word, each with a list of `instances` pointing at third-party source
URLs. See `pose_library/PLAN.md` Section 1 for the full access/licensing
write-up this module implements against.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pose_library import config

logger = logging.getLogger(__name__)

# Video-file extensions/sources known to cause download or decode trouble
# with current tooling -- PLAN.md Section 1 flags `.swf` (Flash, served by
# the `aslpro` source) specifically. Preferred instance selection avoids
# these when the metadata offers a choice for the same gloss word.
_PROBLEMATIC_URL_SUBSTRINGS: tuple[str, ...] = (".swf", "aslpro")

_YOUTUBE_URL_SUBSTRINGS: tuple[str, ...] = ("youtube.com", "youtu.be")

# Source domains that demonstrated real access problems in the Stage 4 dry
# run: `handspeak` was DNS-unreachable from this network, and
# `signingsavvy` returns HTTP 403 for non-browser requests. Neither is
# unusable in principle (someone on a different network, or with browser-
# like headers, might succeed), so they're deprioritized rather than
# excluded outright -- only picked when no other instance exists for that
# gloss word.
_DEPRIORITIZED_URL_SUBSTRINGS: tuple[str, ...] = ("handspeak", "signingsavvy")


def load_wlasl_metadata(
    path: Path = config.WLASL_METADATA_PATH,
) -> list[dict[str, Any]]:
    """Loads WLASL's gloss/instance metadata index from disk.

    Args:
        path: Path to a local copy of `WLASL_v0.3.json`.

    Returns:
        The parsed list of gloss entries, each with a "gloss" string and an
        "instances" list (see PLAN.md Section 1 for the per-instance
        fields: gloss, bbox, fps, frame_start, frame_end, instance_id,
        signer_id, source, split, url, variation_id, video_id).
    """
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def build_gloss_index(
    metadata: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Builds a case-insensitive lookup from gloss word to its metadata entry.

    Args:
        metadata: The parsed list from `load_wlasl_metadata`.

    Returns:
        A dict mapping lowercased gloss word to its full entry dict.
    """
    return {entry["gloss"].lower(): entry for entry in metadata}


def extract_vocab_stems(
    csv_path: Path = config.VOCAB_AUGMENTATION_CSV_PATH,
) -> set[str]:
    """Extracts distinct content-word stems from a (text, gloss) CSV's gloss column.

    Mirrors `pose_library/PLAN.md` Section 1's method: split each gloss
    string on whitespace, drop `X-`-prefixed tokens entirely (these are
    pronoun markers like `X-MY`/`X-YOU`, not independent signs), strip the
    `DESC-` prefix from descriptor tokens (keeping the word itself, since
    it's an independent adjective sign), drop pure punctuation tokens, and
    lowercase the remainder.

    Args:
        csv_path: Path to a CSV file with a "gloss" column (e.g.
            `gloss_model/data/vocab_augmentation.csv`).

    Returns:
        A set of distinct, lowercased content-word stems.
    """
    import csv

    stems: set[str] = set()
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for token in row["gloss"].split():
                if token.startswith("X-"):
                    continue  # pronoun marker, not an independent sign
                if not re.search(r"[A-Za-z]", token):
                    continue  # pure punctuation, e.g. "." or "?"
                token = token.removeprefix("DESC-")
                token = token.strip(".,?!").lower()
                if token:
                    stems.add(token)
    return stems


def select_instance(
    entry: dict[str, Any],
    prefer_direct: bool = True,
) -> dict[str, Any] | None:
    """Selects one video instance for a gloss entry.

    Args:
        entry: A single WLASL gloss entry (from `build_gloss_index`), with
            an "instances" list.
        prefer_direct: If True (default), prefer a direct-file URL (e.g.
            `.mp4`) over YouTube or known-problematic sources (`.swf`,
            `aslpro`) when the metadata offers more than one instance to
            choose from -- per PLAN.md Section 6's "prefer a durable direct
            .mp4 source" recommendation. Sources with demonstrated real
            access problems (`handspeak`: DNS-unreachable; `signingsavvy`:
            HTTP 403 on non-browser requests) are deprioritized below both
            YouTube and other direct sources, and are only selected if no
            other instance is available for that gloss word.

    Returns:
        One instance dict, or `None` if the entry has no instances at all.
    """
    instances = entry.get("instances", [])
    if not instances:
        return None
    if not prefer_direct:
        return instances[0]

    def _rank(instance: dict[str, Any]) -> int:
        url = instance.get("url", "")
        if any(bad in url for bad in _PROBLEMATIC_URL_SUBSTRINGS):
            return 2
        if any(bad in url for bad in _DEPRIORITIZED_URL_SUBSTRINGS):
            return 3
        if any(yt in url for yt in _YOUTUBE_URL_SUBSTRINGS):
            return 1
        return 0  # a direct, non-YouTube, non-problematic file URL

    return min(instances, key=_rank)


def is_youtube_url(url: str) -> bool:
    """Reports whether a WLASL instance URL points at YouTube.

    Args:
        url: The instance's "url" field.

    Returns:
        True if the URL is a YouTube link (needs `yt-dlp`), False otherwise
        (a direct file URL, downloadable with a plain HTTP GET).
    """
    return any(yt in url for yt in _YOUTUBE_URL_SUBSTRINGS)
