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
