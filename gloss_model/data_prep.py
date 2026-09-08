"""Loading, cleaning, splitting, and tokenizing ASLG-PC12 for T5 fine-tuning.

See gloss_model/PLAN.md for the rationale behind each step.
"""

import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from gloss_model import config

logger = logging.getLogger(__name__)


def load_raw_dataset(dataset_name: str = config.DATASET_NAME) -> DatasetDict:
    """Loads the raw ASLG-PC12 dataset from the Hugging Face Hub.

    Args:
        dataset_name: Hugging Face Hub dataset identifier.

    Returns:
        The raw DatasetDict as returned by `datasets.load_dataset`.
    """
    return load_dataset(dataset_name)


def clean_text(value: str) -> str:
    """Strips a BOM character, surrounding whitespace, and trailing newlines.

    Args:
        value: Raw string from the `text` or `gloss` column.

    Returns:
        The cleaned string.
    """
    return value.replace("﻿", "").strip()


def _clean_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        "text": [clean_text(t) for t in batch["text"]],
        "gloss": [clean_text(g) for g in batch["gloss"]],
    }


def build_splits(
    raw: DatasetDict,
    seed: int = config.SEED,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> DatasetDict:
    """Cleans, deduplicates, and splits the dataset into train/validation/test.

    ASLG-PC12 ships a single "train" split with ~7.5% exact-duplicate
    (text, gloss) pairs. Deduplication happens before splitting so the same
    pair can't land in both train and test and inflate eval scores.

    Args:
        raw: The raw DatasetDict from `load_raw_dataset`.
        seed: Random seed for reproducible splitting.
        train_fraction: Fraction of deduplicated rows used for training.
        val_fraction: Fraction of deduplicated rows used for validation. The
            remainder (1 - train_fraction - val_fraction) is used for test.

    Returns:
        A DatasetDict with "train", "validation", and "test" splits.
    """
    dataset = raw["train"].map(_clean_batch, batched=True)

    seen: set[tuple[str, str]] = set()
    keep_indices = []
    for i, (text, gloss) in enumerate(zip(dataset["text"], dataset["gloss"])):
        key = (text, gloss)
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    deduped = dataset.select(keep_indices)
    logger.info("Deduplicated %d -> %d rows", dataset.num_rows, deduped.num_rows)

    test_fraction = 1.0 - train_fraction - val_fraction
    train_rest = deduped.train_test_split(test_size=1.0 - train_fraction, seed=seed)
    val_test = train_rest["test"].train_test_split(
        test_size=test_fraction / (val_fraction + test_fraction), seed=seed
    )

    return DatasetDict(
        {
            "train": train_rest["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )


def load_augmentation_dataset(path: str) -> Dataset:
    """Loads a supplementary (text, gloss) CSV for mixing into training data.

    Args:
        path: Path to a CSV file with "text" and "gloss" columns.

    Returns:
        A Dataset with "text" and "gloss" string columns.
    """
    dataset = load_dataset("csv", data_files=path)["train"]
    return dataset.select_columns(["text", "gloss"])


def mix_in_augmentation(
    splits: DatasetDict,
    augmentation: Dataset,
    repeat: int = 1,
    seed: int = config.SEED,
) -> DatasetDict:
    """Mixes extra (text, gloss) rows into the training split only.

    Validation and test are returned unchanged, so eval metrics stay
    comparable to a run without augmentation.

    Args:
        splits: A pre-tokenization DatasetDict with "train"/"validation"/"test".
        augmentation: Extra rows to add, with "text" and "gloss" columns.
        repeat: How many times to repeat `augmentation` before mixing in. A
            small targeted set can otherwise be diluted to near-zero effective
            weight by a much larger base training split.
        seed: Shuffle seed, so augmented rows aren't clustered at the end.

    Returns:
        A new DatasetDict with an augmented, reshuffled "train" split and the
        original "validation"/"test" splits.
    """
    augmentation = augmentation.cast(splits["train"].features)
    repeated = concatenate_datasets([augmentation] * repeat)
    merged_train = concatenate_datasets([splits["train"], repeated]).shuffle(seed=seed)
    return DatasetDict(
        {
            "train": merged_train,
            "validation": splits["validation"],
            "test": splits["test"],
        }
    )


def preprocess(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int = config.MAX_SOURCE_LENGTH,
    max_target_length: int = config.MAX_TARGET_LENGTH,
    task_prefix: str = config.TASK_PREFIX,
) -> DatasetDict:
    """Tokenizes a train/validation/test DatasetDict for T5 fine-tuning.

    Args:
        dataset: A DatasetDict with "text" (English) and "gloss" columns.
        tokenizer: The T5 tokenizer to use for both source and target.
        max_source_length: Max token length for the English input (truncated).
        max_target_length: Max token length for the gloss target (truncated).
        task_prefix: Text prepended to every source sentence to frame the task.

    Returns:
        A DatasetDict with "input_ids", "attention_mask", and "labels" columns.
    """

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        inputs = [task_prefix + t for t in batch["text"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
        labels = tokenizer(
            text_target=batch["gloss"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=next(iter(dataset.values())).column_names,
    )


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Truncates a dataset to at most `max_samples` rows, if given.

    Args:
        dataset: The dataset to limit.
        max_samples: Maximum number of rows to keep, or None for no limit.

    Returns:
        The original dataset, or a truncated selection of its first rows.
    """
    if max_samples is None or max_samples >= dataset.num_rows:
        return dataset
    return dataset.select(range(max_samples))
