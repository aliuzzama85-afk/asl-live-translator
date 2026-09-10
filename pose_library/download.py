"""Downloads and trims one WLASL video instance per target gloss word.

For each requested word, looks up one instance from WLASL's metadata JSON
(preferring a direct file URL over YouTube/`.swf`/other problematic sources,
per `pose_library/PLAN.md` Section 6), downloads the raw clip (via `yt-dlp`
for YouTube URLs, a plain HTTP GET otherwise), and trims it to the
gloss-relevant segment using the instance's `frame_start`/`frame_end` --
following the trimming approach in WLASL's own
`start_kit/preprocess.py` (frame numbers are 1-indexed there; an
`frame_end <= 0` means "use the whole clip, no trimming needed").

Raw/trimmed clips are stored under `pose_library/data/` (gitignored -- see
PLAN.md Section 1's licensing discussion: WLASL-derived video is a local,
non-redistributed development asset only).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2

from pose_library import config
from pose_library.wlasl_metadata import (
    build_gloss_index,
    is_youtube_url,
    load_wlasl_metadata,
    select_instance,
)

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) "
    "Gecko/2009021910 Firefox/3.0.7"
)

# Below this, a "successful" (HTTP 200) download is almost certainly an
# error/redirect HTML stub rather than real video -- the dry run caught a
# 114-byte HTML redirect stub from aslsearch.com this way (see PLAN.md's
# dry-run findings), but only after wasting a full MediaPipe extraction
# attempt on it. Chosen well below any plausible real clip (even a
# fraction-of-a-second, low-res sign clip is comfortably tens of KB).
_MIN_VIDEO_BYTES = 2048

# Magic-number prefixes for video containers WLASL instances plausibly use
# (MP4/QuickTime family: "....ftyp"; legacy AVI: "RIFF"). Checked as a
# fallback when the response has no (or an untrustworthy) Content-Type
# header -- not exhaustive, just enough to catch "this obviously isn't a
# video file" cases like HTML error/redirect stubs.
_VIDEO_MAGIC_PREFIXES: tuple[bytes, ...] = (b"RIFF",)
_MP4_FTYP_MARKER = b"ftyp"


class NotVideoError(ValueError):
    """Raised when a plain HTTP GET response doesn't look like real video."""


def _looks_like_video(content_type: str | None, body: bytes) -> bool:
    """Sanity-checks a downloaded body before treating it as a real video.

    Args:
        content_type: The response's `Content-Type` header value, if any.
        body: The full downloaded response body.

    Returns:
        True if the response plausibly contains real video: a
        `Content-Type` starting with `video/`, or (as a fallback, since some
        direct-file hosts mislabel or omit `Content-Type`) a body that's
        both above `_MIN_VIDEO_BYTES` and starts with a recognized video
        container magic number.
    """
    if content_type and content_type.strip().lower().startswith("video/"):
        return True
    if len(body) < _MIN_VIDEO_BYTES:
        return False
    return body.startswith(_VIDEO_MAGIC_PREFIXES) or _MP4_FTYP_MARKER in body[:16]


@dataclass
class DownloadResult:
    """Outcome of attempting to download and trim one gloss word's clip.

    Attributes:
        gloss: The target gloss word (lowercase).
        success: Whether a trimmed clip was produced.
        trimmed_path: Path to the trimmed clip, if `success` is True.
        source: WLASL "source" field of the instance used, if one was found.
        video_id: WLASL "video_id" field of the instance used, if found.
        instance_id: WLASL "instance_id" field of the instance used, if found.
        url: The instance URL that was attempted, if found.
        error: A short human-readable reason for failure, if not successful.
    """

    gloss: str
    success: bool
    trimmed_path: Path | None = None
    source: str | None = None
    video_id: str | None = None
    instance_id: int | None = None
    url: str | None = None
    error: str | None = None


def _download_direct(url: str, dest_path: Path) -> None:
    """Downloads a direct file URL via a plain HTTP GET.

    Verifies the response plausibly contains real video (per
    `_looks_like_video`) before writing it out -- a 200 response with an
    HTML redirect/error stub body (observed from one source in the Stage 4
    dry run) is treated as a failure here, not silently passed on to
    `extract.py` to waste a full MediaPipe pass discovering the same thing.

    Args:
        url: A direct video file URL (not YouTube).
        dest_path: Where to save the downloaded bytes.

    Raises:
        urllib.error.URLError: If the request fails.
        NotVideoError: If the response doesn't look like real video (see
            `_looks_like_video`).
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type")
        body = response.read()

    if not _looks_like_video(content_type, body):
        raise NotVideoError(
            f"not a video: got {len(body)} bytes, content-type "
            f"{content_type!r} from {url}"
        )
    dest_path.write_bytes(body)


def _download_youtube(url: str, dest_path: Path) -> None:
    """Downloads a YouTube video via `yt-dlp`, saved to an exact filename.

    Args:
        url: A youtube.com/youtu.be URL.
        dest_path: Exact output path (extension controls the container).

    Raises:
        subprocess.CalledProcessError: If `yt-dlp` exits non-zero.
    """
    subprocess.run(
        [
            "yt-dlp",
            url,
            "-f",
            "mp4/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(dest_path),
        ],
        check=True,
        capture_output=True,
    )


def download_raw(url: str, dest_path: Path) -> None:
    """Downloads one WLASL instance's raw video to `dest_path`.

    Args:
        url: The instance's "url" field.
        dest_path: Where to save the raw (untrimmed) video.

    Raises:
        Exception: Whatever the underlying downloader raises on failure
            (network error, non-zero `yt-dlp` exit, etc.) -- left
            unwrapped so `download_word`'s caller sees the real cause.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if is_youtube_url(url):
        _download_youtube(url, dest_path)
    else:
        _download_direct(url, dest_path)


def _is_cached_raw_valid(raw_path: Path, url: str) -> bool:
    """Reports whether a previously-downloaded raw file is still usable.

    Guards against a stale cache masking the `_looks_like_video` check: a
    bad (e.g. HTML redirect stub) file downloaded by a prior run, before
    this check existed, would otherwise be reused forever since
    `download_word` only re-downloads when `raw_path` is missing. Only
    applies to the plain-HTTP-GET path -- `yt-dlp` output is trusted as-is,
    since it already validates/transcodes the stream itself.

    Args:
        raw_path: Path to the cached raw video file.
        url: The instance's "url" field (used to tell direct downloads from
            YouTube ones).

    Returns:
        True if the cached file should be reused, False if it should be
        re-downloaded.
    """
    if is_youtube_url(url):
        return True
    return _looks_like_video(None, raw_path.read_bytes())


def trim_video(
    src_path: Path, dest_path: Path, frame_start: int, frame_end: int
) -> None:
    """Trims a video to `[frame_start, frame_end]` (WLASL's 1-indexed range).

    Mirrors the approach in WLASL's `start_kit/preprocess.py`: frame numbers
    in the metadata are 1-indexed, and `frame_end <= 0` (WLASL uses `-1`)
    means "use the whole clip" -- no trimming, just a copy.

    Args:
        src_path: Path to the downloaded raw video.
        dest_path: Path to write the trimmed video.
        frame_start: WLASL's 1-indexed `frame_start`.
        frame_end: WLASL's 1-indexed `frame_end`, or `-1` for "to the end".

    Raises:
        ValueError: If no frames could be read from `src_path` at all.
    """
    if frame_end is not None and frame_end <= 0 and frame_start <= 1:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(src_path.read_bytes())
        return

    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_idx = max(frame_start - 1, 0)
    end_idx = frame_end - 1 if frame_end and frame_end > 0 else None

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start_idx and (end_idx is None or idx <= end_idx):
            frames.append(frame)
        idx += 1
        if end_idx is not None and idx > end_idx:
            break
    cap.release()

    if not frames:
        raise ValueError(
            f"No frames read from {src_path} in range [{start_idx}, {end_idx}]"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(dest_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def download_word(
    word: str,
    gloss_index: dict,
    raw_dir: Path = config.RAW_VIDEOS_DIR,
    trimmed_dir: Path = config.TRIMMED_VIDEOS_DIR,
) -> DownloadResult:
    """Downloads and trims one video instance for a single target word.

    Args:
        word: The target gloss word (case-insensitive).
        gloss_index: A WLASL gloss index, from `wlasl_metadata.build_gloss_index`.
        raw_dir: Where to save the untrimmed downloaded clip.
        trimmed_dir: Where to save the trimmed clip.

    Returns:
        A `DownloadResult` describing the outcome (never raises for
        expected failure modes -- dead links, missing gloss, `.swf`/decode
        issues are all captured in `DownloadResult.error`).
    """
    entry = gloss_index.get(word.lower())
    if entry is None:
        return DownloadResult(gloss=word.lower(), success=False, error="not in WLASL")

    instance = select_instance(entry)
    if instance is None:
        return DownloadResult(gloss=word.lower(), success=False, error="no instances")

    url = instance["url"]
    video_id = instance.get("video_id", "unknown")
    raw_ext = ".mp4"
    raw_path = raw_dir / f"{video_id}{raw_ext}"
    trimmed_path = trimmed_dir / f"{word.lower()}.mp4"

    try:
        if not raw_path.is_file() or not _is_cached_raw_valid(raw_path, url):
            download_raw(url, raw_path)
        trim_video(
            raw_path, trimmed_path, instance["frame_start"], instance["frame_end"]
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in DownloadResult, not raised
        logger.warning("Failed to download/trim %r from %s: %s", word, url, exc)
        return DownloadResult(
            gloss=word.lower(),
            success=False,
            source=instance.get("source"),
            video_id=video_id,
            instance_id=instance.get("instance_id"),
            url=url,
            error=str(exc),
        )

    return DownloadResult(
        gloss=word.lower(),
        success=True,
        trimmed_path=trimmed_path,
        source=instance.get("source"),
        video_id=video_id,
        instance_id=instance.get("instance_id"),
        url=url,
    )


def download_words(words: list[str]) -> list[DownloadResult]:
    """Downloads and trims one instance per word for a list of target words.

    Args:
        words: Target gloss words to download.

    Returns:
        One `DownloadResult` per word, in the same order.
    """
    metadata = load_wlasl_metadata()
    gloss_index = build_gloss_index(metadata)
    return [download_word(word, gloss_index) for word in words]


def main() -> None:
    """CLI entry point: downloads/trims clips for a list of gloss words."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="+", help="Target gloss words to download.")
    args = parser.parse_args()

    results = download_words(args.words)
    for result in results:
        status = "OK" if result.success else f"FAILED ({result.error})"
        print(f"{result.gloss}: {status}")


if __name__ == "__main__":
    main()
