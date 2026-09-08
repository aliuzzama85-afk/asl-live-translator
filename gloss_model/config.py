"""Configuration constants for the English-to-ASL-gloss translator (Stage 3).

Single source of truth for dataset identity, preprocessing parameters, and
training hyperparameters, so `data_prep.py`, `train.py`, and `evaluate.py`
stay in sync.
"""

from dataclasses import dataclass
from pathlib import Path

GLOSS_MODEL_DIR = Path(__file__).resolve().parent

# Kaggle mounts notebook inputs (e.g. this repo, uploaded as a Dataset/Notebook
# attachment) read-only under /kaggle/input/...; only /kaggle/working is
# writable there. A checkpoint dir defaulting to somewhere under this file's
# own location would try to write into that read-only mount on Kaggle, so
# detect it and default to the writable working directory instead. See
# KAGGLE.md for the full hand-off procedure.
_KAGGLE_WORKING_DIR = Path("/kaggle/working")

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

# --- Vocabulary-gap augmentation (see gloss_model/VOCAB_DIAGNOSIS.md) ---
# Hand-written (text, gloss) pairs targeting words confirmed near-absent from
# ASLG-PC12 (e.g. "bathroom": 0 occurrences, "coffee": 2). Mixed into the
# TRAIN split only in data_prep.mix_in_augmentation -- validation/test stay
# pure ASLG-PC12 so eval metrics remain comparable across runs.
AUGMENTATION_DATA_PATH = GLOSS_MODEL_DIR / "data" / "vocab_augmentation.csv"

# --- Model / training ---
MODEL_NAME = "t5-small"
CHECKPOINT_DIR = (
    _KAGGLE_WORKING_DIR / "gloss_model_checkpoints"
    if _KAGGLE_WORKING_DIR.is_dir()
    else GLOSS_MODEL_DIR / "checkpoints"
)

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

# Patch mode: a short continued-fine-tuning pass on top of an existing
# checkpoint (--init-checkpoint), mixing in AUGMENTATION_DATA_PATH, to close a
# specific vocabulary gap without a full retrain. Lower LR than a from-scratch
# run, since the model has already converged and a targeted patch shouldn't
# risk destabilizing everything it already learned.
PATCH_NUM_EPOCHS = 2
PATCH_LEARNING_RATE = 1e-4
# The augmentation set (360 rows) is tiny next to the ~73k-row base train
# split -- under 0.5% of it unrepeated, not enough signal for a model to
# reliably pick up new vocabulary in only a couple of epochs. Repeating it
# brings it to a few percent of the mixed train set instead.
PATCH_AUGMENTATION_REPEAT = 20


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

    @classmethod
    def patch(cls) -> "TrainingConfig":
        """Returns the config for a short vocabulary-patch fine-tuning run."""
        return cls(
            learning_rate=PATCH_LEARNING_RATE,
            num_epochs=PATCH_NUM_EPOCHS,
        )
