# Running Stage 3 fine-tuning on Kaggle

Hand-off procedure for running the real (non-`--quick`) `gloss_model` training
run on Kaggle GPU compute. The pipeline itself was verified end-to-end locally
on CPU with `--quick` — this only covers getting the same code running there.
See `gloss_model/PLAN.md` for the pipeline design itself, and
`PROJECT_STATUS.md` Section 8 for why `MAX_SOURCE_LENGTH`/`MAX_TARGET_LENGTH`
and the default augmentation behavior changed since this file was first written.

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

Full run, no `--quick` (that flag is for the local CPU pipeline-sanity check only).
For this run specifically (the combined length + vocabulary fix, replacing
the original v1 model), pass an explicit `--output-dir` rather than relying
on the default — the default resolves to the same directory name v1 used
(`gloss_model_checkpoints`), which isn't a collision risk on Kaggle's own
fresh session but throws away the "clearly separate location" guarantee once
you go to download and compare it locally against v1:

```bash
!python -m gloss_model.train --output-dir /kaggle/working/gloss_model_checkpoints_v2
```

This bare command now does two things beyond the original design:
- Trains with `MAX_SOURCE_LENGTH=48` / `MAX_TARGET_LENGTH=80` (up from 32/32
  — the original values were validated against whitespace word counts, not
  real T5 subword tokens, and silently truncated 59% of training targets; see
  `PROJECT_STATUS.md` Section 8). Expect eval/spot-check generation to take
  noticeably longer than before — benchmarked at roughly 3x the per-sentence
  cost in the worst case (forced full-length generation on CPU); real average
  cost will be lower since most sentences still stop well short of 80 tokens.
- Mixes in `gloss_model/data/vocab_augmentation.csv` **by default**
  (`--augmentation-data`'s default), repeated 10x
  (`config.AUGMENTATION_REPEAT`) into the train split only — so this one run
  now combines the length fix and the vocabulary-gap fix from the start,
  rather than needing a separate `--patch` pass afterward (see Section 7,
  which isn't needed for this particular run). The repeat factor is lower
  than `--patch`'s 20x because this run uses more epochs (4 vs. 2) — repeat x
  epochs stays at ~40 either way, matching what the earlier patch run's
  generalization check confirmed works without memorizing the repeated rows.
- Runs the vocabulary generalization check (`evaluate.VOCAB_GENERALIZATION_SENTENCES`)
  at the end alongside the regular spot check, since that's now gated on
  augmentation actually being used, not specifically on `--patch`.

To override an epoch count, batch size, or the augmentation repeat factor for
this run without editing code, `train.py` accepts `--epochs`, `--batch-size`,
`--augmentation-repeat`, `--max-train-samples`, and `--max-eval-samples`,
e.g. `!python -m gloss_model.train --epochs 6`. Leave these unset to use the
defaults in `gloss_model/config.py`. Pass `--augmentation-data ""` to disable
mixing in the augmentation data entirely, if you ever want a "pure" run for
comparison.

## 5. Where the output lands

`gloss_model/config.py` detects `/kaggle/working` at import time and points
`CHECKPOINT_DIR` (and therefore `train.py`'s default `--output-dir`) at
`/kaggle/working/gloss_model_checkpoints/` — but for this run you passed the
explicit override from Section 4, so it lands at
`/kaggle/working/gloss_model_checkpoints_v2/` instead, containing the final
`model.safetensors`, tokenizer files, and (per
`save_total_limit=2`) up to two intermediate `checkpoint-<step>/`
subdirectories from `save_strategy="epoch"`.

**This only persists past the interactive session if you commit the
notebook** (Save Version → "Save & Run All"). An interactive session that's
just closed without saving a version will lose anything written to
`/kaggle/working`. After a saved version finishes, the checkpoint files are
downloadable from that version's **Output** tab, or you can add that output
as a new Kaggle Dataset to feed into a later inference notebook.

## 6. Bringing the checkpoint back locally

For a run using the default `--output-dir` (`gloss_model_checkpoints/`):
download it from the notebook's Output tab and place it at
`gloss_model/checkpoints/` in the local repo (matching the local default path
from `gloss_model/config.py`) so `gloss_model/inference.py`'s `load_model()`
finds it without needing `--output-dir`/`checkpoint_dir` overrides.

**For this run** (Section 4's `gloss_model_checkpoints_v2`): download it into
a **separate** local folder, e.g. `gloss_model/checkpoints_v2/` — do **not**
overwrite `gloss_model/checkpoints/`, which still holds the current v1 model
you're comparing against. Point `--init-checkpoint`/`load_model(checkpoint_dir=...)`
at `gloss_model/checkpoints_v2/` explicitly when you want to try it, until
you've decided whether to promote it to replace v1.

## 7. Patch runs: continuing from an existing checkpoint

**Not needed for the current max_length + vocabulary fix** — that's being
done as one fresh full retrain (Sections 1-6; see Section 4's notes) since
the foundational training is being redone anyway, rather than layering a
patch on top of a patch. This section stays documented for a future
situation where a targeted fix on top of an already-good checkpoint makes
more sense than a full retrain. The `checkpoint-v1.zip` /
`asl-gloss-checkpoint-v1` Kaggle Dataset uploaded for the previous patch run
isn't needed for a full retrain — no need to re-upload it for this run, but
no harm leaving it attached either.

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
the source archive). `--init-checkpoint` only *reads* from this path (it's
never written to), so unlike the source dataset it doesn't need copying into
`/kaggle/working` — it can be read straight from the read-only
`/kaggle/input` mount.

Only the **top-level** files matter — `model.safetensors`, `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`training_args.bin` (~230MB total). Skip any `checkpoint-<step>/`
subdirectories: those are `save_strategy="epoch"` intermediate saves that
each carry a full Adam `optimizer.pt` (~2x model size, meant for resuming an
*interrupted* run) — dead weight for `--init-checkpoint`, which never reads
them, and they'll roughly 7x the upload for nothing.

1. From the repo root (PowerShell): zip just the top-level files, not the
   whole `checkpoints/` tree —
   `Compress-Archive -Path (Get-ChildItem gloss_model\checkpoints -File).FullName -DestinationPath checkpoint-v1.zip`
   (on a system with `zip`: `cd gloss_model/checkpoints && zip ../../checkpoint-v1.zip *.json *.safetensors *.bin && cd ../..`).
   This produces a zip with those files at its root — no wrapping folder.
2. On kaggle.com: **Add Data** → **New Dataset** → upload `checkpoint-v1.zip`
   → name it (e.g. `asl-gloss-checkpoint-v1`) → create.
3. In the notebook, **Add Input** and attach it alongside
   `asl-live-translator-src`. Because the zip had no wrapping folder, it
   mounts read-only with the files directly at
   `/kaggle/input/asl-gloss-checkpoint-v1/` (e.g.
   `/kaggle/input/asl-gloss-checkpoint-v1/model.safetensors`) — not nested
   under an extra `checkpoints/` or `checkpoint-v1/` subfolder. If you zip it
   a different way and it does end up nested, adjust the `--init-checkpoint`
   path in Section 7.3 accordingly; run `!ls /kaggle/input/asl-gloss-checkpoint-v1/`
   to check.

### 7.2 Setup cell (patch variant)

Same package installs as Section 3, plus copying the *source* into
`/kaggle/working` as before. The checkpoint input does **not** need copying:

```bash
!cp -r /kaggle/input/asl-live-translator-src/gloss_model /kaggle/working/
%cd /kaggle/working

!pip install -q "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0" \
    "evaluate==0.4.6" "sacrebleu==2.6.0"
```

### 7.3 Training command (patch variant)

```bash
!python -m gloss_model.train --patch --init-checkpoint /kaggle/input/asl-gloss-checkpoint-v1
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
