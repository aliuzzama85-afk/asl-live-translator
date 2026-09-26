"""Gloss token -> pose-library lookup word: the one shared rule.

`gloss_model` emits ASL gloss in this corpus's notation: uppercase words,
`X-` pronoun markers (`X-I`, `X-MY`), `DESC-` descriptors (`DESC-NICE`), and
punctuation tokens. The pose library is keyed by plain lowercase words.
`normalize_gloss_token` is the rule that maps one onto the other. It was
extracted from `wlasl_metadata.extract_vocab_stems`, where it first lived
inline, so the pose *vocabulary* and live *playback* use exactly the same
mapping instead of two copies that could drift apart
(`pipeline/STAGE1_2_PLAN.md` Section 4).

`gloss_to_lookup_words` adds one playback-only step on top: a small stop-list
of English function words the corpus keeps as gloss tokens but ASL doesn't
sign, so they're dropped instead of fingerspelled.
"""

from __future__ import annotations

import re

# English function words the rule-generated corpus keeps as gloss tokens but
# ASL grammar doesn't sign (so fingerspelling them would be wrong, not just
# noisy). Starts with exactly the list pose_library/PLAN.md Section 1 gives
# ("be, do, at, too, would, well, and"), plus "to", seen in real model
# output (e.g. "X-I NEED TO SEE DOCTOR"). Changes only on evidence. Needs a
# check by someone who knows ASL (STAGE1_2_PLAN.md Section 4). Deliberately
# NOT applied by normalize_gloss_token itself: the pose vocabulary build
# (extract_vocab_stems) keeps these words, as it always has.
PLAYBACK_STOP_WORDS: frozenset[str] = frozenset(
    {"be", "do", "at", "too", "would", "well", "and", "to"}
)

_HAS_LETTER = re.compile(r"[A-Za-z]")


def normalize_gloss_token(token: str) -> str | None:
    """Maps one gloss token to its pose-library lookup word.

    Args:
        token: One whitespace-separated gloss token, e.g. `"DESC-NICE"`.

    Returns:
        The lowercase lookup word (`"nice"`), or `None` if the token isn't an
        independent sign: an `X-` pronoun marker (`X-MY`; pronouns are
        pointing signs, and the library has none), or pure punctuation
        (`"."`, `"?"`). `DESC-` is stripped (the word itself is a real
        adjective sign), and trailing/leading `.,?!` are removed.
    """
    if token.startswith("X-"):
        return None
    if not _HAS_LETTER.search(token):
        return None
    word = token.removeprefix("DESC-").strip(".,?!").lower()
    return word or None


def gloss_to_lookup_words(gloss: str) -> tuple[list[str], list[str]]:
    """Turns a model gloss string into the words to play, in order.

    Args:
        gloss: A whole gloss string, e.g. `"CAN X-YOU HELP X-I FIND X-MY PHONE"`.

    Returns:
        `(words, dropped)`: `words` are the lowercase lookup words to hand
        to playback (`["can", "help", "find", "phone"]`), duplicates and
        order kept. `dropped` lists every original token that was removed
        (pronoun markers, punctuation, stop-list words), in order, so nothing
        disappears without a record.
    """
    words: list[str] = []
    dropped: list[str] = []
    for token in gloss.split():
        word = normalize_gloss_token(token)
        if word is None or word in PLAYBACK_STOP_WORDS:
            dropped.append(token)
        else:
            words.append(word)
    return words, dropped
