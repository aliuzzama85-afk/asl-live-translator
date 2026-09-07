"""Metrics for judging English -> ASL gloss translation quality.

Primary metrics are exact-match sequence accuracy and token-level F1, since
gloss word order matters more than n-gram overlap for the downstream pose
lookup stage. BLEU is reported alongside as a secondary, familiar reference
point only. See gloss_model/PLAN.md Section 4 for the rationale.
"""

from collections import Counter

import evaluate as hf_evaluate
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from gloss_model import config

_BLEU = None

# Hand-picked sentences NOT in ASLG-PC12, covering a spread of sentence shapes
# (simple, negation, question, compound) for eyeballing model output quality.
# See PLAN.md Section 4: metric scores alone aren't enough to judge "good enough".
SPOT_CHECK_SENTENCES = [
    "I am hungry.",
    "Where is the bathroom?",
    "She does not want coffee.",
    "Can you help me find my keys?",
    "The weather is nice today.",
    "I do not understand your question.",
    "What time does the store close?",
    "He is going to the doctor tomorrow.",
    "We should leave before it gets dark.",
    "Is this seat taken?",
    "My phone battery is almost dead.",
    "They are not coming to the party.",
    "How do I get to the train station?",
    "I have been waiting here for an hour.",
    "Please turn off the lights when you leave.",
]

# Minimal stopword list for the naive baseline -- not meant to be linguistically
# complete, just enough to distinguish "did nothing" from "did something".
_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "but",
    "for",
    "with",
    "as",
    "by",
    "that",
    "this",
    "it",
    "be",
}


def _normalize(sequence: str) -> str:
    return " ".join(sequence.strip().upper().split())


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Computes the fraction of predictions that exactly match their reference.

    Args:
        predictions: Predicted gloss strings.
        references: Reference gloss strings, same length and order as predictions.

    Returns:
        The exact-match accuracy in [0, 1], after whitespace/case normalization.
    """
    if not predictions:
        return 0.0
    matches = sum(
        1
        for pred, ref in zip(predictions, references)
        if _normalize(pred) == _normalize(ref)
    )
    return matches / len(predictions)


def token_f1(predictions: list[str], references: list[str]) -> float:
    """Computes average per-example token-multiset F1, ignoring order.

    Args:
        predictions: Predicted gloss strings.
        references: Reference gloss strings, same length and order as predictions.

    Returns:
        The mean token-level F1 across examples, in [0, 1].
    """
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = Counter(_normalize(pred).split())
        ref_tokens = Counter(_normalize(ref).split())
        overlap = sum((pred_tokens & ref_tokens).values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / sum(pred_tokens.values())
        recall = overlap / sum(ref_tokens.values())
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def bleu(predictions: list[str], references: list[str]) -> float:
    """Computes corpus-level sacreBLEU.

    Args:
        predictions: Predicted gloss strings.
        references: Reference gloss strings, same length and order as predictions.

    Returns:
        The sacreBLEU score (0-100 scale).
    """
    global _BLEU
    if _BLEU is None:
        _BLEU = hf_evaluate.load("sacrebleu")
    if not predictions:
        return 0.0
    result = _BLEU.compute(
        predictions=predictions, references=[[r] for r in references]
    )
    return result["score"]


def naive_baseline(text: str) -> str:
    """Produces a pseudo-gloss by uppercasing and dropping common stopwords.

    This is a comparison point for `evaluate.py`'s metrics, not a real
    translation strategy: a fine-tuned model should clear this baseline
    comfortably, or something is wrong with training.

    Args:
        text: English input sentence.

    Returns:
        A naive pseudo-gloss string.
    """
    tokens = [tok for tok in text.strip().split() if tok.lower() not in _STOPWORDS]
    return " ".join(tok.upper() for tok in tokens)


def decode_predictions(
    tokenizer: PreTrainedTokenizerBase,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[str], list[str]]:
    """Decodes generated token ids and label ids back into strings.

    Args:
        tokenizer: The tokenizer used during preprocessing.
        predictions: Generated token ids from `Seq2SeqTrainer.predict`. When
            eval batches are accumulated across many steps with variable
            generated lengths, the Trainer pads mismatched-length sequences
            with -100 (the label-ignore sentinel), not the tokenizer's pad id
            -- so predictions need the same -100 cleanup as labels do, or
            `batch_decode` raises OverflowError on the negative ids.
        labels: Label token ids, with -100 marking ignored positions.

    Returns:
        A tuple of (decoded predictions, decoded references).
    """
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_preds, decoded_labels


def compute_metrics_fn(tokenizer: PreTrainedTokenizerBase):
    """Builds a `compute_metrics` callback bound to a specific tokenizer.

    Args:
        tokenizer: The tokenizer used during preprocessing.

    Returns:
        A function suitable for `Seq2SeqTrainer(compute_metrics=...)`.
    """

    def compute_metrics(eval_preds) -> dict[str, float]:
        predictions, labels = eval_preds
        decoded_preds, decoded_labels = decode_predictions(
            tokenizer, predictions, labels
        )
        return {
            "exact_match": exact_match(decoded_preds, decoded_labels),
            "token_f1": token_f1(decoded_preds, decoded_labels),
            "bleu": bleu(decoded_preds, decoded_labels),
        }

    return compute_metrics


def run_spot_check(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentences: list[str] = SPOT_CHECK_SENTENCES,
    max_length: int = config.MAX_TARGET_LENGTH,
) -> list[tuple[str, str]]:
    """Generates gloss predictions for hand-picked sentences, for manual review.

    Metric scores alone can hide systematic issues (e.g. a model that faithfully
    reproduces ASLG-PC12's rule-based quirks); this is meant to be read by eye,
    not asserted on automatically.

    Args:
        model: A fine-tuned T5 model in eval mode.
        tokenizer: The matching tokenizer.
        sentences: English sentences to translate, ideally not in the training set.
        max_length: Max generated token length.

    Returns:
        A list of (input_sentence, predicted_gloss) pairs, in the input order.
    """
    inputs = tokenizer(
        [config.TASK_PREFIX + s for s in sentences],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SOURCE_LENGTH,
    ).to(model.device)
    generated = model.generate(**inputs, max_length=max_length)
    predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return list(zip(sentences, predictions))
