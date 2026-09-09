# ASL Live Translator — Project Context

## What this is
A live speech/text → ASL sign-avatar translator. Pipeline: streaming ASR → VAD-based
phrase chunking → English-to-gloss translation → gloss-to-pose lookup → smoothed
real-time animation. This is an assistive-tech prototype for informal, everyday
communication (not a certified interpreter replacement — do not build or market it
as one).

## Architecture (5 stages)
1. **ASR (speech-to-text)**: streaming, via Deepgram or AssemblyAI. Lives in `pipeline/asr.py`.
2. **VAD chunking**: Silero VAD detects pauses, emits complete phrases. Lives in `pipeline/vad.py`.
3. **Gloss translation**: fine-tuned T5-small, English → ASL gloss order. Lives in `gloss_model/`.
   Training data: ASLG-PC12 (Hugging Face Datasets).
4. **Gloss-to-pose lookup**: MediaPipe-extracted keypoint sequences per gloss word, sourced
   from WLASL. Fingerspell fallback for out-of-vocabulary words. Lives in `pose_library/`.
5. **Rendering**: 2D skeleton animation first (Canvas/Three.js), 3D rigged avatar is a
   stretch goal (Mixamo + Three.js). Lives in `frontend/`.

Data flows through an in-memory queue so playback never blocks on translation —
the avatar always stays a phrase or two "behind" live speech, like a human interpreter.

## Reference implementations (check these before writing new logic)
- MediaPipe (google/mediapipe) — hand/pose landmarks
- WLASL (dxli94/WLASL) — isolated sign video dataset
- ASLG-PC12 — gloss/English parallel corpus (HF Datasets)
- Silero VAD (snakers4/silero-vad) — small, readable VAD model
- Hugging Face Transformers T5 fine-tuning docs — follow their canonical script structure

## Coding conventions
- Python: black + ruff formatting, type hints on all function signatures, docstrings
  (Google style) on all public functions.
- JS/TS: eslint + prettier, functional components only, no class components.
- No notebook-style scripts in the committed codebase — convert exploratory notebook
  code into proper modules with `if __name__ == "__main__":` entry points.
- Before writing a new function, search the repo for an existing one that already does
  this or something close. Extend/reuse over duplicating. Flag near-duplicate logic you
  find during other work, don't just leave it.

## Security rules (non-negotiable)
- Never hardcode API keys, tokens, or credentials. Always load via `.env` +
  `python-dotenv` / `dotenv` — and `.env` must be in `.gitignore` from commit one.
- Sanitize and length-limit all user text/audio input before it reaches any model call.
- Rate-limit the live audio/translation endpoint.
- Run `gitleaks` (or equivalent) as a pre-commit hook — no exceptions.
- Run `pip-audit` / `npm audit` before adding new dependencies.

## Testing
- Every new module in `pipeline/`, `gloss_model/`, or `pose_library/` needs a
  corresponding test in `tests/`.
- Don't move to the next pipeline stage until the current stage's tests pass.

## Build order (do not skip ahead)
1. Static gloss translator (Stage 3 only) — typed English in, gloss text out.
2. Pose library + static skeleton renderer (Stage 4 + 5), fed gloss manually.
3. Interpolation/smoothing between signs.
4. Live speech input (Stage 1 + 2) wired to the working core.

## UI/UX direction
Minimal chrome, dark mode default, feels like a live captioning overlay (think
Otter.ai's real-time transcript view) rather than a generic dashboard. Strong color
contrast and readable type sizes — this is an accessibility tool, treat that as a
hard requirement, not a nice-to-have. Show clear loading/latency state, never a
silent frozen avatar.

## Session hygiene
When you learn something during a session that should persist (a dataset quirk, a
design decision, a gotcha), update this file before ending the session.

## Known gotchas / decisions log
- **ASLG-PC12 real size**: 87,710 rows (not the ~12k sometimes cited), single `train`
  split, hub id `achrafothman/aslg_pc12`. Rule-generated, not human-translated — 1.7%
  of rows are just the English sentence case-flipped, and it has zero fingerspelling/
  classifier annotation. Treated as an acceptable quality ceiling for a first pass, not
  a blocker. Full detail: [gloss_model/PLAN.md](gloss_model/PLAN.md).
- **`Seq2SeqTrainer` + `predict_with_generate` gotcha**: when eval predictions are
  generated sequences accumulated across many batches with variable lengths, the
  Trainer pads mismatched-length predictions with `-100` (the label-ignore sentinel),
  not the tokenizer's pad id. Any code decoding `EvalPrediction.predictions` must clean
  `-100` out of *both* predictions and labels before `tokenizer.batch_decode`, or it can
  crash with `OverflowError: can't convert negative int to unsigned` on larger eval
  sets (it may not show up on a small validation split, only a bigger one). Fixed in
  `gloss_model/evaluate.py::decode_predictions`.
- **Compute split**: local dev machine is CPU-only. Real fine-tuning runs happen on
  Kaggle (GPU); local runs use `python -m gloss_model.train --quick` (small subset, 1
  epoch) purely to verify the pipeline end-to-end, not to produce a good model.
- **Kaggle checkpoint-path fix**: `gloss_model/config.py`'s `CHECKPOINT_DIR` defaulted
  to a path relative to the package's own file location, which breaks on Kaggle if this
  repo is attached as a read-only notebook input (`/kaggle/input/...`). Fixed by
  detecting `/kaggle/working` and defaulting there instead when present. See
  [KAGGLE.md](KAGGLE.md).
- **Two device-placement bugs found on the first real Kaggle GPU run**: both
  `gloss_model/evaluate.py::run_spot_check` and `gloss_model/inference.py::translate`
  tokenized inputs and called `model.generate()` without moving those inputs to
  `model.device` first — crashes with `RuntimeError: Expected all tensors to be on
  the same device` once the model is on `cuda:0` instead of CPU. Fixed in both by
  chaining `.to(model.device)` onto the tokenizer output; both have mocked regression
  tests that were verified to fail without the fix.
- **First full Kaggle GPU training run succeeded** (4 epochs, full ~81k-row train
  split): `test_token_f1=0.965`, `test_bleu=93.7`, `test_exact_match=0.80` — a strong
  first result, well above the CPU `--quick` smoke-test numbers in PLAN.md's
  "Implementation status". Stage 4 (pose library) can proceed.
- **`MAX_SOURCE_LENGTH`/`MAX_TARGET_LENGTH=32` were validated against whitespace
  word counts, never real T5 subword tokens.** The custom `DESC-`/`X-` gloss
  notation isn't in T5's vocabulary, so it fragments into several subword pieces
  per token — actual gloss-target token counts are p50/p90/p99 = 36/51/65 (not
  12/17/20 as word counts suggested), and **59% of all training targets exceed
  32 tokens and get silently truncated** by `truncation=True` in
  `data_prep.preprocess()`. True since the very first training run, not
  introduced by the vocab patch. **Fixed 2026-09-10** — raised to 48/80 and
  retrained; see the v2 entry below and `PROJECT_STATUS.md` Sections 8-9.
- **A vocabulary-gap patch (`train.py --patch`, see `gloss_model/VOCAB_DIAGNOSIS.md`)
  ran on Kaggle and improved all metrics**: `test_token_f1=0.970`,
  `test_bleu=94.7`, `test_exact_match=0.835`, with the generalization check
  confirming real vocabulary learning, not memorization of the repeated
  augmented sentences. Words outside both the corpus and the 121-word patch
  list (e.g. "apartment", "skip") still produce garbled output — expected,
  not a bug; the gap is broader than what was patched.
- **v2 retrain combined the `max_length` fix and vocabulary augmentation into
  one from-scratch run** (not another patch on top of a patch) —
  `MAX_SOURCE_LENGTH=48`/`MAX_TARGET_LENGTH=80`, `t5-small`, full 4 epochs,
  augmentation mixed in from the start. Succeeded: `test_bleu=93.9`,
  `test_token_f1=0.966`, `test_exact_match=0.812` — matching/slightly
  exceeding v1 despite being graded against far less truncated ground truth,
  a real improvement not a wash. Spot-check 15/15 clean, generalization check
  7/8 clean (including "weather" and "skip" now correct, both broken before).
  One held-out sentence ("roommate...gave them back") still stops early at
  `GIVE X-`, but confirmed via direct testing (`max_length=48` vs `80` give
  identical 33-token output) that this is the model choosing to stop, not the
  truncation bug recurring — a narrow generalization limit, not a config
  issue. **Stage 3 formally signed off 2026-09-10.**
  `gloss_model/checkpoints_v2/` is the model to use going forward, superseding
  `gloss_model/checkpoints/` (v1, kept for comparison). Full detail:
  `PROJECT_STATUS.md` Section 9. Also noted: the checkpoint's own
  `generation_config.json` still carries T5's stock `max_length: 20` default
  (harmless today since every caller passes `max_length` explicitly, but a
  footgun for any future caller that doesn't).
