"""Tests for pose_library.wlasl_metadata."""

import pytest

from pose_library.wlasl_metadata import (
    build_gloss_index,
    extract_vocab_stems,
    is_youtube_url,
    select_instance,
)

_METADATA = [
    {
        "gloss": "Coffee",
        "instances": [{"url": "http://x.com/coffee.mp4", "source": "x"}],
    },
    {"gloss": "water", "instances": []},
]


def test_build_gloss_index_lowercases_keys():
    index = build_gloss_index(_METADATA)
    assert "coffee" in index
    assert "Coffee" not in index
    assert index["coffee"]["instances"][0]["source"] == "x"


def test_extract_vocab_stems_drops_pronoun_markers(tmp_path):
    csv_path = tmp_path / "vocab.csv"
    csv_path.write_text(
        "text,gloss\n"
        '"My mother is a teacher .","X-MY MOTHER BE TEACHER ."\n'
        '"She is sick .","X-SHE BE DESC-SICK ."\n',
        encoding="utf-8",
    )
    stems = extract_vocab_stems(csv_path)
    assert stems == {"mother", "be", "teacher", "sick"}
    # X-MY / X-SHE (pronoun markers) must not leak through as "my" / "she".
    assert "my" not in stems
    assert "she" not in stems


def test_extract_vocab_stems_strips_desc_prefix_but_keeps_word(tmp_path):
    csv_path = tmp_path / "vocab.csv"
    csv_path.write_text(
        'text,gloss\n"I am hungry .","X-I BE DESC-HUNGRY ."\n', encoding="utf-8"
    )
    stems = extract_vocab_stems(csv_path)
    assert "hungry" in stems
    assert "desc-hungry" not in stems


def test_extract_vocab_stems_drops_pure_punctuation_tokens(tmp_path):
    csv_path = tmp_path / "vocab.csv"
    csv_path.write_text(
        'text,gloss\n"Is it here ?","BE X-IT DESC-HERE ?"\n', encoding="utf-8"
    )
    stems = extract_vocab_stems(csv_path)
    assert "." not in stems
    assert "?" not in stems


def test_select_instance_prefers_direct_mp4_over_youtube_and_swf():
    entry = {
        "instances": [
            {"url": "http://aslpro.com/x.swf"},
            {"url": "https://youtube.com/watch?v=abc"},
            {"url": "https://media.spreadthesign.com/word/x.mp4"},
        ]
    }
    chosen = select_instance(entry)
    assert chosen["url"] == "https://media.spreadthesign.com/word/x.mp4"


def test_select_instance_falls_back_to_youtube_over_swf_if_no_direct_option():
    entry = {
        "instances": [
            {"url": "http://aslpro.com/x.swf"},
            {"url": "https://youtube.com/watch?v=abc"},
        ]
    }
    chosen = select_instance(entry)
    assert "youtube.com" in chosen["url"]


def test_select_instance_deprioritizes_handspeak_and_signingsavvy():
    """`handspeak` (DNS-unreachable) and `signingsavvy` (HTTP 403) are only
    picked if no other instance exists for the gloss word -- see PLAN.md's
    dry-run findings.
    """
    entry = {
        "instances": [
            {"url": "http://aslpro.com/x.swf"},
            {"url": "https://youtube.com/watch?v=abc"},
            {"url": "https://handspeak.com/word/x.mp4"},
            {"url": "https://www.signingsavvy.com/word/1234.mp4"},
        ]
    }
    chosen = select_instance(entry)
    assert "youtube.com" in chosen["url"]


def test_select_instance_picks_handspeak_or_signingsavvy_as_last_resort():
    entry = {
        "instances": [
            {"url": "https://handspeak.com/word/x.mp4"},
            {"url": "https://www.signingsavvy.com/word/1234.mp4"},
        ]
    }
    chosen = select_instance(entry)
    assert chosen is not None
    assert "handspeak" in chosen["url"] or "signingsavvy" in chosen["url"]


def test_select_instance_returns_none_for_no_instances():
    assert select_instance({"instances": []}) is None
    assert select_instance({}) is None


def test_select_instance_prefer_direct_false_returns_first():
    entry = {"instances": [{"url": "a"}, {"url": "b"}]}
    assert select_instance(entry, prefer_direct=False)["url"] == "a"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://handspeak.com/word/x.mp4", False),
    ],
)
def test_is_youtube_url(url, expected):
    assert is_youtube_url(url) is expected
