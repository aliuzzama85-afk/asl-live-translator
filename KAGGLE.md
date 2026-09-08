# Running Stage 3 fine-tuning on Kaggle

Hand-off procedure for running the real (non-`--quick`) `gloss_model` training
run on Kaggle GPU compute. The pipeline itself was verified end-to-end locally
on CPU with `--quick` — this only covers getting the same code running there.
No training logic or hyperparameters are changed for this; see `gloss_model/PLAN.md`
for the pipeline design itself.

Sections 1-6 below cover a full run from `t5-small`. **Section 7 covers patch
runs** — continuing fine-tuning from an existing checkpoint with a small
vocabulary-augmentation dataset mixed in (see `gloss_model/VOCAB_DIAGNOSIS.md`)
— which changes the upload and run steps in a few specific ways layered on
top of the same base procedure.

## 1. What to upload

Only `gloss_model/` (all `.py` files) and `requirements.txt` are needed — not
`.venv/`, `node_modules/`, `frontend/`, `data/`, `pose_library/`, or
`gloss_model/checkpoints/` (that's local output, not input).

From the repo root, package the git-tracked source into a zip:

```bash
git archive --format=zip -o asl-live-translator-src.zip HEAD gloss_model requirements.txt
```

On kaggle.com: **Add Data** → **New Dataset** → upload `asl-live-translator-src.zip`
(Kaggle unzips it automatically) → name it (e.g. `asl-live-translator-src`) →
create. Then, in your training notebook, **Add Input** and attach that dataset.
It will be mounted read-only at `/kaggle/input/asl-live-translator-src/`.

(If you push this repo to GitHub later, `!git clone <url>` in a notebook cell
is a simpler alternative to re-uploading a zip each time.)

## 2. Notebook settings

Before running anything, in the notebook's right-hand **Settings** panel:
- **Accelerator**: GPU (e.g. T4 x2 or P100 — whichever quota you have).
- **Internet**: On. Required for two things at runtime: `load_dataset("achrafothman/aslg_pc12")`
  and `evaluate.load("sacrebleu")` both fetch from the Hugging Face Hub over the network.

## 3. Setup cell

Copy the source into the writable working directory and install the pinned
package versions (see `requirements.txt` for why these are exact pins, not
ranges — the pipeline's behavior, including a `Seq2SeqTrainer` padding bugfix,
was verified against exactly these versions):

```bash
!cp -r /kaggle/input/asl-live-translator-src/gloss_model /kaggle/working/
%cd /kaggle/working

!pip install -q "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0" \
    "evaluate==0.4.6" "sacrebleu==2.6.0"
```

Deliberately **not** `pip install -r requirements.txt` here: that file also
lists `torch` (Kaggle's preinstalled, GPU-enabled `torch` build should be left
alone — reinstalling it risks silently swapping in a mismatched or CPU-only
build) and dev/unrelated tooling (`ruff`, `black`, `pytest`, `pip-audit`,
`mediapipe`) that this training run doesn't need.

Copying into `/kaggle/working` (rather than running directly from the
read-only `/kaggle/input/...` mount) is what makes the checkpoint path
resolution in `gloss_model/config.py` land in the right place — see Section 5.

## 4. Training command

Full run, no `--quick` (that flag is for the local CPU pipeline-sanity check only):

```bash
!python -m gloss_model.train
```

To override an epoch count or batch size for this run without editing code,
`train.py` accepts `--epochs`, `--batch-size`, `--max-train-samples`, and
`--max-eval-samples`, e.g. `!python -m gloss_model.train --epochs 6`. Leave
these unset to use the defaults in `gloss_model/config.py`.

## 5. Where the output lands

`gloss_model/config.py` detects `/kaggle/working` at import time and points
`CHECKPOINT_DIR` (and therefore `train.py`'s default `--output-dir`) at:

```
/kaggle/working/gloss_model_checkpoints/
```

containing the final `model.safetensors`, tokenizer files, and (per
`save_total_limit=2`) up to two intermediate `checkpoint-<step>/`
subdirectories from `save_strategy="epoch"`.

**This only persists past the interactive session if you commit the
notebook** (Save Version → "Save & Run All"). An interactive session that's
just closed without saving a version will lose anything written to
`/kaggle/working`. After a saved version finishes, the checkpoint files are
downloadable from that version's **Output** tab, or you can add that output
as a new Kaggle Dataset to feed into a later inference notebook.

## 6. Bringing the checkpoint back locally

Download `gloss_model_checkpoints/` from the notebook's Output tab and place
it at `gloss_model/checkpoints/` in the local repo (matching the local default
path from `gloss_model/config.py`) so `gloss_model/inference.py`'s
`load_model()` finds it without needing `--output-dir`/`checkpoint_dir`
overrides.

## 7. Patch runs: continuing from an existing checkpoint

For a targeted vocabulary fix rather than a full retrain, `train.py --patch`
continues fine-tuning from an already-trained checkpoint (instead of
`t5-small`), using a lower learning rate, few epochs, and
`gloss_model/data/vocab_augmentation.csv` mixed into the training data only.
This layers a few changes on top of Sections 1-6; Section 2 (notebook
settings) and Section 5/6 (output/retrieval mechanics) are unchanged.

### 7.1 What to upload (in addition to Section 1)

`git archive ... gloss_model ...` (Section 1) now also picks up
`gloss_model/data/vocab_augmentation.csv` automatically — **but only if it's
committed to git first**; `git archive` archives committed content, not
working-tree-only files.

The checkpoint itself needs a **second** Kaggle Dataset input, since
`gloss_model/checkpoints/` is gitignored (242MB+ of weights, never part of
the source archive):

1. From the repo root: `cd gloss_model && zip -r ../checkpoint-v1.zip checkpoints && cd ..`
   (or upload the `gloss_model/checkpoints/` folder directly — Kaggle's
   dataset uploader accepts folders, not just zips).
2. On kaggle.com: **Add Data** → **New Dataset** → upload it → name it (e.g.
   `asl-gloss-checkpoint-v1`) → create.
3. In the notebook, **Add Input** and attach it alongside
   `asl-live-translator-src`. It mounts read-only at
   `/kaggle/input/asl-gloss-checkpoint-v1/`.

### 7.2 Setup cell (patch variant)

Same package installs as Section 3, plus copying the checkpoint into
`/kaggle/working` alongside the source:

```bash
!cp -r /kaggle/input/asl-live-translator-src/gloss_model /kaggle/working/
!cp -r /kaggle/input/asl-gloss-checkpoint-v1/checkpoints /kaggle/working/gloss_model/init_checkpoint
%cd /kaggle/working

!pip install -q "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0" \
    "evaluate==0.4.6" "sacrebleu==2.6.0"
```

If the second `cp` fails, run `!ls /kaggle/input/asl-gloss-checkpoint-v1/`
first — Kaggle's zip upload sometimes nests an extra folder level, so the
real path might be `.../asl-gloss-checkpoint-v1/checkpoints/checkpoints`.

### 7.3 Training command (patch variant)

```bash
!python -m gloss_model.train --patch --init-checkpoint /kaggle/working/gloss_model/init_checkpoint
```

This uses `TrainingConfig.patch()` (`gloss_model/config.py`): 2 epochs,
learning rate `1e-4` (down from `3e-4`, standard practice for continuing
fine-tuning without destabilizing what the model already learned), and mixes
in `vocab_augmentation.csv` repeated 20× (`config.PATCH_AUGMENTATION_REPEAT`)
so its 360 rows aren't diluted to near-zero effective weight against the
~73k-row base train split. Override any of these the same way as a full run,
e.g. `--epochs 3` or `--augmentation-repeat 30`.

Validation and test stay pure ASLG-PC12 (augmentation is mixed into the
**train** split only), so this run's `test_token_f1` / `test_bleu` /
`test_exact_match` are directly comparable to the original run's (0.965 /
93.7 / 0.80) — a regression there would be a red flag. The real signal for
whether the patch actually worked is the qualitative spot-check at the end:
it already includes several of the originally-failing words (`bathroom`,
`coffee`, `weather`, `keys`, `station`, `party`, `hour`, `lights`) —compare
its output against the original run's spot-check log.

### 7.4 Output location (patch variant)

`--output-dir` defaults to a **different** path when `--patch` is set —
`gloss_model_checkpoints_patch` (sibling to `gloss_model_checkpoints`, not
overwriting it) — specifically so a patch run can't silently clobber the
checkpoint it started from before you've reviewed whether it's actually
better. On Kaggle this lands at
`/kaggle/working/gloss_model_checkpoints_patch/`. Bring it back locally into
a separate folder (e.g. `gloss_model/checkpoints_patch/`), per Section 6, so
you can compare against the current `gloss_model/checkpoints/` before
deciding whether to promote it.
