# Running Stage 3 fine-tuning on Kaggle

Hand-off procedure for running the real (non-`--quick`) `gloss_model` training
run on Kaggle GPU compute. The pipeline itself was verified end-to-end locally
on CPU with `--quick` — this only covers getting the same code running there.
No training logic or hyperparameters are changed for this; see `gloss_model/PLAN.md`
for the pipeline design itself.

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
