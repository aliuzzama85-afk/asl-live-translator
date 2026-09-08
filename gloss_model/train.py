"""Fine-tunes T5-small on ASLG-PC12 for English -> ASL gloss translation.

Usage:
    python -m gloss_model.train --quick   # fast CPU sanity run, small subset
    python -m gloss_model.train           # full run (intended for a GPU, e.g. Kaggle)
    python -m gloss_model.train --patch --init-checkpoint <dir>
        # short continued fine-tuning pass on an existing checkpoint, mixing
        # in gloss_model/data/vocab_augmentation.csv, to close a specific
        # vocabulary gap without a full retrain -- see VOCAB_DIAGNOSIS.md.

See gloss_model/PLAN.md for the full rationale behind each step.
"""

import argparse
import logging

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
)

from gloss_model import config, data_prep, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training run.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Fast, small-scale end-to-end sanity run (CPU-friendly). "
        "Not intended to produce a good model -- see PLAN.md.",
    )
    mode.add_argument(
        "--patch",
        action="store_true",
        help="Short continued fine-tuning pass on top of --init-checkpoint "
        "(lower learning rate, few epochs), mixing in --augmentation-data. "
        "For a targeted vocabulary fix, not a full retrain -- see "
        "VOCAB_DIAGNOSIS.md.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Path to an existing fine-tuned checkpoint directory to "
        "initialize the model and tokenizer from, instead of "
        "config.MODEL_NAME (t5-small). Required in practice for --patch.",
    )
    parser.add_argument(
        "--augmentation-data",
        type=str,
        default=str(config.AUGMENTATION_DATA_PATH),
        help="CSV of extra (text,gloss) rows mixed into the TRAIN split only "
        "(validation/test are left as pure ASLG-PC12). Pass an empty string "
        "to disable.",
    )
    parser.add_argument(
        "--augmentation-repeat",
        type=int,
        default=config.PATCH_AUGMENTATION_REPEAT,
        help="How many times to repeat the augmentation rows before mixing "
        "them into the (much larger) base train split, so they aren't "
        "diluted to near-zero effective weight.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epoch count."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-device batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override the learning rate."
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the number of training examples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap the number of validation examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the fine-tuned model. Defaults to "
        "config.CHECKPOINT_DIR, or a sibling '..._patch' directory in "
        "--patch mode, so a patch run can't silently overwrite the "
        "checkpoint it's initialized from before you've reviewed it.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            str(config.CHECKPOINT_DIR) + "_patch"
            if args.patch
            else str(config.CHECKPOINT_DIR)
        )
    return args


def resolve_training_config(args: argparse.Namespace) -> config.TrainingConfig:
    """Builds a TrainingConfig from CLI args, layering overrides onto a base.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The resolved TrainingConfig for this run.
    """
    if args.quick:
        base = config.TrainingConfig.quick()
    elif args.patch:
        base = config.TrainingConfig.patch()
    else:
        base = config.TrainingConfig()
    return config.TrainingConfig(
        learning_rate=(
            args.learning_rate if args.learning_rate is not None else base.learning_rate
        ),
        weight_decay=base.weight_decay,
        num_epochs=args.epochs if args.epochs is not None else base.num_epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        max_train_samples=(
            args.max_train_samples
            if args.max_train_samples is not None
            else base.max_train_samples
        ),
        max_eval_samples=(
            args.max_eval_samples
            if args.max_eval_samples is not None
            else base.max_eval_samples
        ),
    )


def main() -> None:
    """Loads data, fine-tunes T5-small, evaluates, and saves the checkpoint."""
    args = parse_args()
    train_config = resolve_training_config(args)
    set_seed(config.SEED)

    logger.info("Loading and splitting dataset...")
    raw = data_prep.load_raw_dataset()
    splits = data_prep.build_splits(raw)
    splits["train"] = data_prep.limit_dataset(
        splits["train"], train_config.max_train_samples
    )
    splits["validation"] = data_prep.limit_dataset(
        splits["validation"], train_config.max_eval_samples
    )
    # Cap the held-out test split the same way in quick mode -- otherwise a
    # "fast sanity check" run still pays for a full, uncapped generation pass
    # over the entire test split at the end.
    splits["test"] = data_prep.limit_dataset(
        splits["test"], train_config.max_eval_samples
    )

    if args.augmentation_data:
        logger.info(
            "Mixing in augmentation data from %s (x%d)...",
            args.augmentation_data,
            args.augmentation_repeat,
        )
        augmentation = data_prep.load_augmentation_dataset(args.augmentation_data)
        splits = data_prep.mix_in_augmentation(
            splits, augmentation, repeat=args.augmentation_repeat, seed=config.SEED
        )

    logger.info(
        "train=%d validation=%d test=%d",
        splits["train"].num_rows,
        splits["validation"].num_rows,
        splits["test"].num_rows,
    )

    model_source = args.init_checkpoint or config.MODEL_NAME
    logger.info("Initializing model and tokenizer from %s", model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenized = data_prep.preprocess(splits, tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(model_source)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        weight_decay=train_config.weight_decay,
        num_train_epochs=train_config.num_epochs,
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        save_total_limit=2,
        seed=config.SEED,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=evaluate.compute_metrics_fn(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Evaluating on held-out test split...")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"], metric_key_prefix="test"
    )
    logger.info("Test metrics: %s", test_metrics)

    logger.info("Running qualitative spot check...")
    model.eval()
    for sentence, prediction in evaluate.run_spot_check(model, tokenizer):
        logger.info("  %r -> %r", sentence, prediction)

    if args.patch:
        # Distinguishes genuine vocabulary generalization from memorization
        # of the repeated (--augmentation-repeat) augmented phrasings -- see
        # VOCAB_GENERALIZATION_SENTENCES's docstring in evaluate.py. Logged
        # as its own section, not mixed into the spot check above, so the
        # comparison is easy to find and read.
        logger.info(
            "Running vocabulary generalization check (novel sentence "
            "structures, not memorization of vocab_augmentation.csv)..."
        )
        for sentence, prediction in evaluate.run_spot_check(
            model, tokenizer, sentences=evaluate.VOCAB_GENERALIZATION_SENTENCES
        ):
            logger.info("  %r -> %r", sentence, prediction)


if __name__ == "__main__":
    main()
