"""Configuration constants for the English-to-ASL-gloss translator (Stage 3).

Single source of truth for dataset identity, preprocessing parameters, and
training hyperparameters, so `data_prep.py`, `train.py`, and `evaluate.py`
stay in sync.
"""

from dataclasses import dataclass
from pathlib import Path

GLOSS_MODEL_DIR = Path(__file__).resolve().parent

# --- Dataset ---
# Verified via a real `load_dataset()` call: single "train" split, 87,710 rows,
# columns ["gloss", "text"], no empty rows. ~7.5% exact-duplicate (text, gloss)
# pairs -- deduped in `data_prep.build_splits`. See PLAN.md Section 0/1.
DATASET_NAME = "achrafothman/aslg_pc12"

# --- Reproducibility ---
SEED = 42

# --- Preprocessing ---
# T5 has no separate encoder/decoder vocab -- task framing comes from this prefix.
TASK_PREFIX = "translate English to ASL gloss: "
# Corpus token-length p99 is 21 (text) / 20 (gloss); 32 covers that with margin
# without wasting compute on the rare long tail (max observed: 59 / 54).
MAX_SOURCE_LENGTH = 32
MAX_TARGET_LENGTH = 32

# --- Splits ---
TRAIN_FRACTION = 0.90
VAL_FRACTION = 0.05
TEST_FRACTION = 0.05

# --- Model / training ---
MODEL_NAME = "t5-small"
CHECKPOINT_DIR = GLOSS_MODEL_DIR / "checkpoints"

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
BATCH_SIZE = 16

# Quick mode: a fast, small-scale end-to-end sanity run (e.g. on a CPU-only
# machine) to prove the pipeline works before a full run on a GPU (e.g. Kaggle).
# Not intended to produce a good model -- see PLAN.md Section 4.
QUICK_MAX_TRAIN_SAMPLES = 500
QUICK_MAX_EVAL_SAMPLES = 100
QUICK_NUM_EPOCHS = 1
QUICK_BATCH_SIZE = 8


@dataclass(frozen=True)
class TrainingConfig:
    """Resolved hyperparameters for one training run.

    Attributes:
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_epochs: Number of training epochs.
        batch_size: Per-device train/eval batch size.
        max_train_samples: Cap on training examples, or None to use all.
        max_eval_samples: Cap on validation examples, or None to use all.
    """

    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    @classmethod
    def quick(cls) -> "TrainingConfig":
        """Returns the small-scale config for a fast CPU pipeline sanity check."""
        return cls(
            num_epochs=QUICK_NUM_EPOCHS,
            batch_size=QUICK_BATCH_SIZE,
            max_train_samples=QUICK_MAX_TRAIN_SAMPLES,
            max_eval_samples=QUICK_MAX_EVAL_SAMPLES,
        )
