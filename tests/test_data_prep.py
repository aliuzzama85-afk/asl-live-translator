"""Tests for gloss_model.data_prep."""

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from gloss_model import config, data_prep


def _toy_raw(rows: list[tuple[str, str]]) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": [r[0] for r in rows], "gloss": [r[1] for r in rows]}
            )
        }
    )


def test_clean_text_strips_bom_and_whitespace():
    assert data_prep.clean_text("﻿  hello world \n") == "hello world"


def test_build_splits_dedupes_exact_pairs():
    unique_rows = [(f"text {i}", f"GLOSS {i}") for i in range(100)]
    rows = unique_rows + unique_rows[:10]  # 10 exact duplicates
    raw = _toy_raw(rows)
    splits = data_prep.build_splits(raw, seed=0)
    total = sum(splits[s].num_rows for s in ("train", "validation", "test"))
    assert total == len(unique_rows)


def test_build_splits_deterministic():
    rows = [(f"text {i}", f"GLOSS {i}") for i in range(50)]
    raw = _toy_raw(rows)
    a = data_prep.build_splits(raw, seed=1)
    b = data_prep.build_splits(raw, seed=1)
    assert a["train"]["text"] == b["train"]["text"]
    assert a["test"]["text"] == b["test"]["text"]


def test_build_splits_no_overlap():
    rows = [(f"text {i}", f"GLOSS {i}") for i in range(100)]
    raw = _toy_raw(rows)
    splits = data_prep.build_splits(raw, seed=2)
    train_texts = set(splits["train"]["text"])
    val_texts = set(splits["validation"]["text"])
    test_texts = set(splits["test"]["text"])
    assert not (train_texts & val_texts)
    assert not (train_texts & test_texts)
    assert not (val_texts & test_texts)
    assert len(train_texts) + len(val_texts) + len(test_texts) == 100


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(config.MODEL_NAME)
    except OSError:
        pytest.skip("t5-small tokenizer not available (no network / not cached)")


def test_preprocess_adds_task_prefix_and_respects_max_length(tokenizer):
    ds = _toy_raw([("hello there", "HELLO THERE")])
    tokenized = data_prep.preprocess(
        ds, tokenizer, max_source_length=8, max_target_length=8
    )
    row = tokenized["train"][0]
    assert "input_ids" in tokenized["train"].column_names
    assert "labels" in tokenized["train"].column_names
    decoded_input = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
    assert decoded_input.startswith("translate English to ASL gloss")
    assert len(row["input_ids"]) <= 8
    assert len(row["labels"]) <= 8


def test_limit_dataset_caps_rows():
    ds = Dataset.from_dict({"x": list(range(10))})
    assert data_prep.limit_dataset(ds, 3).num_rows == 3


def test_limit_dataset_no_op_when_none_or_over_size():
    ds = Dataset.from_dict({"x": list(range(10))})
    assert data_prep.limit_dataset(ds, None).num_rows == 10
    assert data_prep.limit_dataset(ds, 100).num_rows == 10
