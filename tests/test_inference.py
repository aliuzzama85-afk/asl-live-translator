"""Tests for gloss_model.inference."""

import pytest

from gloss_model import config, inference

_HAS_CHECKPOINT = config.CHECKPOINT_DIR.exists() and any(
    config.CHECKPOINT_DIR.glob("*.safetensors")
)


def test_translate_rejects_empty_input():
    with pytest.raises(inference.EmptyInputError):
        inference.translate("   ", model=None, tokenizer=None)


def test_translate_rejects_overlong_input():
    too_long = "a" * (inference.MAX_INPUT_CHARS + 1)
    with pytest.raises(inference.InputTooLongError):
        inference.translate(too_long, model=None, tokenizer=None)


def test_translate_accepts_input_at_exact_limit_without_length_error():
    at_limit = "a" * inference.MAX_INPUT_CHARS
    # Should pass the length/emptiness guards and only fail later trying to
    # call a None tokenizer -- confirms the limit is inclusive, not exclusive.
    with pytest.raises(TypeError):
        inference.translate(at_limit, model=None, tokenizer=None)


@pytest.mark.skipif(not _HAS_CHECKPOINT, reason="no fine-tuned checkpoint present yet")
def test_translate_returns_nonempty_string_with_real_checkpoint():
    model, tokenizer = inference.load_model()
    result = inference.translate("I am hungry.", model, tokenizer)
    assert isinstance(result, str)
    assert result.strip() != ""
