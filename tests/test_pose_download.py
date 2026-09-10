"""Tests for pose_library.download.

Network/`yt-dlp` calls are mocked; `trim_video` is exercised against a real
small synthetic video (fast, local, no network) since it's pure `cv2` logic
worth testing directly.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from pose_library.download import (
    DownloadResult,
    NotVideoError,
    _download_direct,
    download_word,
    trim_video,
)


def test_download_word_redownloads_a_stale_non_video_cached_raw_file(tmp_path):
    """Regression test: a raw file cached by a run *before* the sanity check
    existed (e.g. an old HTML redirect stub) must not be silently reused
    forever just because `raw_path.is_file()` is True."""
    gloss_index = {
        "foo": {
            "instances": [
                {
                    "url": "https://example.com/foo.mp4",
                    "video_id": "00001",
                    "instance_id": 7,
                    "source": "example",
                    "frame_start": 1,
                    "frame_end": -1,
                }
            ]
        }
    }
    raw_dir = tmp_path / "raw"
    trimmed_dir = tmp_path / "trimmed"
    raw_dir.mkdir()
    # Simulate a stale, pre-sanity-check cached "download": a tiny HTML stub.
    (raw_dir / "00001.mp4").write_bytes(b"<html>redirect</html>")

    with patch("pose_library.download.download_raw") as mock_download_raw:

        def _fake_download_raw(url, dest_path):
            dest_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096)

        mock_download_raw.side_effect = _fake_download_raw

        result = download_word(
            "foo", gloss_index, raw_dir=raw_dir, trimmed_dir=trimmed_dir
        )

    mock_download_raw.assert_called_once()
    assert result.success
    assert (
        result.trimmed_path.read_bytes() == b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096
    )


def test_download_word_reuses_valid_cached_raw_file(tmp_path):
    gloss_index = {
        "foo": {
            "instances": [
                {
                    "url": "https://example.com/foo.mp4",
                    "video_id": "00001",
                    "instance_id": 7,
                    "source": "example",
                    "frame_start": 1,
                    "frame_end": -1,
                }
            ]
        }
    }
    raw_dir = tmp_path / "raw"
    trimmed_dir = tmp_path / "trimmed"
    raw_dir.mkdir()
    real_looking_video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096
    (raw_dir / "00001.mp4").write_bytes(real_looking_video)

    with patch("pose_library.download.download_raw") as mock_download_raw:
        result = download_word(
            "foo", gloss_index, raw_dir=raw_dir, trimmed_dir=trimmed_dir
        )

    mock_download_raw.assert_not_called()
    assert result.success
    assert result.trimmed_path.read_bytes() == real_looking_video


def _write_synthetic_video(path, num_frames: int, size=(20, 16)) -> None:
    """Writes a tiny video where frame N's pixel value encodes N, for asserting on."""
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_two_tone_video(path, num_frames: int, split_at: int, size=(20, 16)) -> None:
    """Writes frames far apart in brightness (dark, then bright) so lossy mp4v
    re-encoding can't blur which "half" a decoded frame came from -- used
    instead of per-frame-unique values, which mp4v's compression rounds away.
    """
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    for i in range(num_frames):
        value = 10 if i < split_at else 245
        writer.write(np.full((height, width, 3), fill_value=value, dtype=np.uint8))
    writer.release()


def _read_all_frames(path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_trim_video_extracts_requested_1indexed_frame_range(tmp_path):
    src = tmp_path / "src.mp4"
    dest = tmp_path / "trimmed.mp4"
    # Frames 0-4 dark, frames 5-9 bright.
    _write_two_tone_video(src, num_frames=10, split_at=5)

    # WLASL frame_start/frame_end are 1-indexed; request frames 7..9 (1-indexed)
    # i.e. 0-indexed frames 6, 7, 8 -> 3 bright frames, none dark.
    trim_video(src, dest, frame_start=7, frame_end=9)

    frames = _read_all_frames(dest)
    assert len(frames) == 3
    assert all(frame.mean() > 200 for frame in frames)  # all from the bright half


def test_trim_video_copies_whole_clip_when_frame_end_non_positive(tmp_path):
    src = tmp_path / "src.mp4"
    dest = tmp_path / "trimmed.mp4"
    _write_synthetic_video(src, num_frames=10)

    trim_video(src, dest, frame_start=1, frame_end=-1)

    assert dest.read_bytes() == src.read_bytes()


def test_trim_video_raises_if_range_is_out_of_bounds(tmp_path):
    src = tmp_path / "src.mp4"
    dest = tmp_path / "trimmed.mp4"
    _write_synthetic_video(src, num_frames=5)

    with pytest.raises(ValueError):
        trim_video(src, dest, frame_start=100, frame_end=200)


def test_download_word_reports_word_not_in_wlasl():
    result = download_word("not-a-real-gloss-word", gloss_index={})
    assert isinstance(result, DownloadResult)
    assert not result.success
    assert result.error == "not in WLASL"


def test_download_word_reports_no_instances():
    gloss_index = {"foo": {"instances": []}}
    result = download_word("foo", gloss_index=gloss_index)
    assert not result.success
    assert result.error == "no instances"


def test_download_word_success_path(tmp_path):
    gloss_index = {
        "foo": {
            "instances": [
                {
                    "url": "https://example.com/foo.mp4",
                    "video_id": "00001",
                    "instance_id": 7,
                    "source": "example",
                    "frame_start": 1,
                    "frame_end": -1,
                }
            ]
        }
    }
    raw_dir = tmp_path / "raw"
    trimmed_dir = tmp_path / "trimmed"

    with (
        patch("pose_library.download.download_raw") as mock_download_raw,
        patch("pose_library.download.trim_video") as mock_trim_video,
    ):

        def _fake_download_raw(url, dest_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"fake video bytes")

        mock_download_raw.side_effect = _fake_download_raw

        result = download_word(
            "foo", gloss_index, raw_dir=raw_dir, trimmed_dir=trimmed_dir
        )

    assert result.success
    assert result.source == "example"
    assert result.video_id == "00001"
    assert result.instance_id == 7
    assert result.trimmed_path == trimmed_dir / "foo.mp4"
    mock_trim_video.assert_called_once()


def _mock_response(body: bytes, content_type: str | None):
    """Builds a mock `urlopen`-context-manager result for `_download_direct`."""
    response = MagicMock()
    response.headers.get.return_value = content_type
    response.read.return_value = body
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_download_direct_rejects_non_video_content_type(tmp_path):
    dest = tmp_path / "out.mp4"
    small_html_body = b"<html>redirect</html>"
    response = _mock_response(small_html_body, "text/html; charset=utf-8")

    with (
        patch("pose_library.download.urllib.request.urlopen", return_value=response),
        pytest.raises(NotVideoError, match="not a video: got 21 bytes"),
    ):
        _download_direct("https://example.com/word.mp4", dest)

    assert not dest.exists()


def test_download_direct_rejects_small_body_even_without_content_type(tmp_path):
    """Some hosts omit/mislabel Content-Type -- fall back to a size/magic-number
    check so a small non-video stub still isn't accepted."""
    dest = tmp_path / "out.mp4"
    tiny_stub = b"redirect stub"
    response = _mock_response(tiny_stub, None)

    with (
        patch("pose_library.download.urllib.request.urlopen", return_value=response),
        pytest.raises(NotVideoError),
    ):
        _download_direct("https://example.com/word.mp4", dest)

    assert not dest.exists()


def test_download_direct_accepts_video_content_type(tmp_path):
    dest = tmp_path / "out.mp4"
    fake_video_body = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    response = _mock_response(fake_video_body, "video/mp4")

    with patch("pose_library.download.urllib.request.urlopen", return_value=response):
        _download_direct("https://example.com/word.mp4", dest)

    assert dest.read_bytes() == fake_video_body


def test_download_direct_accepts_large_body_with_video_magic_number_fallback(tmp_path):
    """No/untrustworthy Content-Type, but a real-looking mp4 magic number and
    a large-enough body -- accepted via the size/magic-number fallback."""
    dest = tmp_path / "out.mp4"
    fake_video_body = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096
    response = _mock_response(fake_video_body, None)

    with patch("pose_library.download.urllib.request.urlopen", return_value=response):
        _download_direct("https://example.com/word.mp4", dest)

    assert dest.read_bytes() == fake_video_body


def test_download_word_reports_not_a_video_as_download_failure():
    gloss_index = {
        "foo": {
            "instances": [
                {
                    "url": "https://example.com/foo.mp4",
                    "video_id": "00001",
                    "instance_id": 7,
                    "source": "example",
                    "frame_start": 1,
                    "frame_end": -1,
                }
            ]
        }
    }
    with patch(
        "pose_library.download.download_raw",
        side_effect=NotVideoError(
            "not a video: got 114 bytes, content-type 'text/html'"
        ),
    ):
        result = download_word("foo", gloss_index)

    assert not result.success
    assert "not a video" in result.error


def test_download_word_captures_download_failure_without_raising():
    gloss_index = {
        "foo": {
            "instances": [
                {
                    "url": "https://example.com/foo.mp4",
                    "video_id": "00001",
                    "instance_id": 7,
                    "source": "example",
                    "frame_start": 1,
                    "frame_end": -1,
                }
            ]
        }
    }
    with patch("pose_library.download.download_raw", side_effect=OSError("boom")):
        result = download_word("foo", gloss_index)

    assert not result.success
    assert "boom" in result.error
