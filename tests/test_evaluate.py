"""Tests for gloss_model.evaluate."""

import numpy as np
import pytest
from transformers import AutoTokenizer

from gloss_model import config, evaluate


def test_exact_match_identical():
    assert evaluate.exact_match(["CAT SLEEP"], ["CAT SLEEP"]) == 1.0


def test_exact_match_ignores_case_and_whitespace():
    assert evaluate.exact_match(["  cat   sleep "], ["CAT SLEEP"]) == 1.0


def test_exact_match_disjoint():
    assert evaluate.exact_match(["CAT SLEEP"], ["DOG RUN"]) == 0.0


def test_exact_match_partial_across_examples():
    preds = ["CAT SLEEP", "DOG RUN"]
    refs = ["CAT SLEEP", "BIRD FLY"]
    assert evaluate.exact_match(preds, refs) == 0.5


def test_token_f1_identical():
    assert evaluate.token_f1(["CAT SLEEP"], ["CAT SLEEP"]) == 1.0


def test_token_f1_disjoint():
    assert evaluate.token_f1(["CAT SLEEP"], ["DOG RUN"]) == 0.0


def test_token_f1_partial_overlap():
    # pred={CAT,SLEEP}, ref={CAT,RUN}: overlap=1, precision=1/2, recall=1/2, f1=0.5
    score = evaluate.token_f1(["CAT SLEEP"], ["CAT RUN"])
    assert score == pytest.approx(0.5)


def test_naive_baseline_uppercases_and_drops_stopwords():
    assert evaluate.naive_baseline("the cat is on the mat") == "CAT MAT"


def test_naive_baseline_scores_lower_than_perfect_prediction():
    text = "the cat is sleeping on the mat"
    reference = "CAT SLEEP MAT"
    naive_pred = evaluate.naive_baseline(
        text
    )  # -> "CAT SLEEPING MAT" (no lemmatization)
    naive_score = evaluate.token_f1([naive_pred], [reference])
    perfect_score = evaluate.token_f1([reference], [reference])
    assert naive_score < perfect_score


def test_decode_predictions_handles_ignore_index():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    encoded = tokenizer(["CAT SLEEP"], padding="max_length", max_length=6)
    labels = np.array(encoded["input_ids"])
    labels_with_ignore = labels.copy()
    labels_with_ignore[labels_with_ignore == tokenizer.pad_token_id] = -100

    preds, refs = evaluate.decode_predictions(tokenizer, labels, labels_with_ignore)
    assert refs[0].strip() != ""
    assert preds[0] == refs[0]


def test_decode_predictions_handles_ignore_index_in_predictions_too():
    # Regression test: Seq2SeqTrainer pads variable-length generated sequences
    # across accumulated eval batches with -100 (not the pad token id), so
    # `predictions` can contain -100 just like `labels` can. Decoding must not
    # crash (OverflowError) when that happens.
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    encoded = tokenizer(["CAT SLEEP"], padding="max_length", max_length=6)
    clean = np.array(encoded["input_ids"])
    predictions_with_ignore = clean.copy()
    predictions_with_ignore[predictions_with_ignore == tokenizer.pad_token_id] = -100

    preds, refs = evaluate.decode_predictions(tokenizer, predictions_with_ignore, clean)
    assert preds[0].strip() != ""
    assert preds[0] == refs[0]


def test_bleu_identical_is_high():
    # sacreBLEU's default 4-gram order needs >= 4 tokens per sentence, or the
    # missing n-gram order zeroes the score even for a perfect match.
    sentence = "CAT SLEEP ON THE MAT TODAY"
    try:
        score = evaluate.bleu([sentence], [sentence])
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"sacrebleu metric unavailable (no network): {exc}")
    assert score > 90.0


def test_bleu_disjoint_is_low():
    try:
        score = evaluate.bleu(
            ["CAT SLEEP ON THE MAT TODAY"], ["DOG RUN IN THE PARK NOW"]
        )
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"sacrebleu metric unavailable (no network): {exc}")
    assert score < 10.0


def test_run_spot_check_returns_pairs_for_each_sentence():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    from transformers import T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    sentences = ["I am hungry.", "Where is the bathroom?"]
    results = evaluate.run_spot_check(
        model, tokenizer, sentences=sentences, max_length=8
    )
    assert [s for s, _ in results] == sentences
    assert all(isinstance(pred, str) for _, pred in results)
