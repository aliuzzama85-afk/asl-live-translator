"""Tests for gloss_model.inference."""

from unittest.mock import MagicMock

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


def test_translate_moves_tokenizer_output_to_model_device():
    # Regression test for the same class of bug fixed in
    # gloss_model.evaluate.run_spot_check: "RuntimeError: Expected all
    # tensors to be on the same device, but got index is on cpu, different
    # from other tensors on cuda:0". The tokenizer output defaults to CPU
    # regardless of where the model lives, so translate() must explicitly
    # move it to model.device before model.generate(). Mocked (no GPU
    # needed) to assert BatchEncoding.to() is actually called with
    # model.device.
    fake_device = "meta"  # any device string; what matters is it's threaded through
    fake_tensors = {
        "input_ids": MagicMock(name="input_ids"),
        "attention_mask": MagicMock(name="attention_mask"),
    }

    moved_encoding = MagicMock()
    moved_encoding.keys.return_value = list(fake_tensors.keys())
    moved_encoding.__getitem__.side_effect = fake_tensors.__getitem__

    raw_encoding = MagicMock()
    raw_encoding.to.return_value = moved_encoding

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = raw_encoding
    mock_tokenizer.decode.return_value = "GLOSS OUTPUT"

    mock_model = MagicMock()
    mock_model.device = fake_device
    mock_model.generate.return_value = [MagicMock()]

    inference.translate("hello", mock_model, mock_tokenizer)

    raw_encoding.to.assert_called_once_with(fake_device)
    # generate() must be called with the *moved* encoding's tensors, not the raw one's.
    _, generate_kwargs = mock_model.generate.call_args
    assert generate_kwargs["input_ids"] is fake_tensors["input_ids"]
    assert generate_kwargs["attention_mask"] is fake_tensors["attention_mask"]
