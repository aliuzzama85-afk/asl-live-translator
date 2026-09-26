"""Tests for pose_library.gloss_tokens.

The whole-string cases use real `gloss_model` (checkpoints_v2) output,
recorded while writing pipeline/STAGE1_2_PLAN.md Section 3, not invented
gloss.
"""

import pytest

from pose_library.gloss_tokens import (
    PLAYBACK_STOP_WORDS,
    gloss_to_lookup_words,
    normalize_gloss_token,
)


@pytest.mark.parametrize(
    "token, expected",
    [
        ("PHONE", "phone"),
        ("DESC-NICE", "nice"),  # descriptor prefix stripped, word kept
        ("X-I", None),  # pronoun marker: not an independent sign
        ("X-MY", None),
        ("X-", None),  # a bare marker (seen in truncated real output)
        (".", None),  # punctuation-only
        ("?", None),
        ("TODAY.", "today"),  # attached punctuation stripped
        ("BE", "be"),  # the shared rule does NOT apply the stop-list
    ],
)
def test_normalize_gloss_token(token, expected):
    assert normalize_gloss_token(token) == expected


@pytest.mark.parametrize(
    "gloss, words, dropped",
    [
        # All real checkpoints_v2 outputs.
        (
            "CAN X-YOU HELP X-I FIND X-MY PHONE",
            ["can", "help", "find", "phone"],
            ["X-YOU", "X-I", "X-MY"],
        ),
        ("WHERE BE BATHROOM", ["where", "bathroom"], ["BE"]),
        (
            "X-I NEED TO SEE DOCTOR TOMORROW MORNING",
            ["need", "see", "doctor", "tomorrow", "morning"],
            ["X-I", "TO"],
        ),
        (
            "X-MY FRIEND BE DESC-NOT COME TO PARTY DESC-TONIGHT BECAUSE X-",
            ["friend", "not", "come", "party", "tonight", "because"],
            ["X-MY", "BE", "TO", "X-"],
        ),
        ("HALF", ["half"], []),
    ],
)
def test_gloss_to_lookup_words_on_real_model_output(gloss, words, dropped):
    assert gloss_to_lookup_words(gloss) == (words, dropped)


def test_every_token_is_either_played_or_recorded_as_dropped():
    gloss = "X-I BE DESC-HAPPY TO SEE X-YOU TODAY ."
    words, dropped = gloss_to_lookup_words(gloss)
    assert len(words) + len(dropped) == len(gloss.split())
    assert dropped == ["X-I", "BE", "TO", "X-YOU", "."]


def test_repeated_words_are_kept_in_order():
    assert gloss_to_lookup_words("MORE MORE PLEASE")[0] == ["more", "more", "please"]


def test_empty_gloss_gives_nothing():
    assert gloss_to_lookup_words("   ") == ([], [])


def test_stop_list_is_the_documented_one():
    # pose_library/PLAN.md Section 1's list, plus "to" (STAGE1_2_PLAN.md
    # Section 4). A change here should be a deliberate, reviewed one.
    assert PLAYBACK_STOP_WORDS == {
        "be",
        "do",
        "at",
        "too",
        "would",
        "well",
        "and",
        "to",
    }
