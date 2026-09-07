"""Thin inference wrapper around a fine-tuned English -> ASL gloss checkpoint.

Not wired into `pipeline/` yet -- that integration belongs to a later stage
per CLAUDE.md's build order. This module only exposes `translate()` for that
future caller (and for this stage's own tests/spot-checks).
"""

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
)

from gloss_model import config

# Mirrors CLAUDE.md's "sanitize and length-limit all user text/audio input"
# rule. This module is the eventual callee for that live endpoint, so basic
# input guarding belongs here too, not only at the HTTP boundary.
MAX_INPUT_CHARS = 500


class EmptyInputError(ValueError):
    """Raised when the input text is empty or whitespace-only."""


class InputTooLongError(ValueError):
    """Raised when the input text exceeds `MAX_INPUT_CHARS`."""


def load_model(
    checkpoint_dir: str = str(config.CHECKPOINT_DIR),
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads a fine-tuned model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to a directory saved by `trainer.save_model`.

    Returns:
        A (model, tokenizer) tuple, with the model in eval mode.
    """
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.eval()
    return model, tokenizer


def translate(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = config.MAX_TARGET_LENGTH,
) -> str:
    """Translates one English sentence into ASL gloss order.

    Args:
        text: English input sentence.
        model: A fine-tuned T5 model (see `load_model`).
        tokenizer: The matching tokenizer.
        max_length: Max generated token length.

    Returns:
        The predicted gloss string.

    Raises:
        EmptyInputError: If `text` is empty or whitespace-only.
        InputTooLongError: If `text` exceeds `MAX_INPUT_CHARS` characters.
    """
    if not text.strip():
        raise EmptyInputError("Input text must not be empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise InputTooLongError(
            f"Input text exceeds {MAX_INPUT_CHARS} characters (got {len(text)})."
        )

    inputs = tokenizer(
        config.TASK_PREFIX + text,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_SOURCE_LENGTH,
    )
    generated = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(generated[0], skip_special_tokens=True)
