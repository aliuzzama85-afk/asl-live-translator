# ASL Live Translator — Project Context

## What this is
A live speech/text → ASL sign-avatar translator. Pipeline: streaming ASR → VAD-based
phrase chunking → English-to-gloss translation → gloss-to-pose lookup → smoothed
real-time animation. This is an assistive-tech prototype for informal, everyday
communication (not a certified interpreter replacement — do not build or market it
as one).

## Architecture (5 stages)
1. **ASR (speech-to-text)**: v1 uses the browser's **Web Speech API** (no key, no cost),
   so it lives in the frontend, not Python: `frontend/src/lib/asr/`, behind a small adapter
   interface (`start`/`stop` + one event stream). A streaming provider (Deepgram, AssemblyAI)
   is a future adapter plugged in at `frontend/src/lib/asr/index.js`; nothing downstream
   changes. There is **no `pipeline/asr.py`**: the browser owns the mic and no audio ever
   reaches Python. See `pipeline/STAGE1_2_PLAN.md`.
2. **Phrase chunking**: v1 uses the recognizer's own end-of-utterance ("final result")
   segmentation plus a 12-word length guard (`frontend/src/lib/phrases.js`). There is **no
   `pipeline/vad.py` / Silero VAD**: the Web Speech API never exposes the audio, and a
   second segmenter would only disagree with the recognizer's. Revisit only for a raw-audio
   provider (which ship their own endpointing anyway).
   **Live bridge**: `pipeline/gloss_server.py` (stdlib HTTP, `127.0.0.1`) serves the
   gloss model to the browser through Vite's `/api` proxy. Run both with
   `npm run dev:live` (in `frontend/`). Typed English (TRANSLATE mode, the
   default) goes through this same bridge and queue as speech; EXACT WORDS
   mode looks typed words up literally, without the model.
3. **Gloss translation**: fine-tuned T5-small, English → ASL gloss order. Lives in `gloss_model/`.
   Training data: ASLG-PC12 (Hugging Face Datasets).
4. **Gloss-to-pose lookup**: MediaPipe-extracted keypoint sequences per gloss word, sourced
   from WLASL. Fingerspell fallback for out-of-vocabulary words. Lives in `pose_library/`.
5. **Rendering**: 2D skeleton animation first (Canvas/Three.js), 3D rigged avatar is a
   stretch goal (Mixamo + Three.js). Lives in `frontend/`.

Data flows through an in-memory queue so playback never blocks on translation —
the avatar always stays a phrase or two "behind" live speech, like a human interpreter.
(Implemented in the frontend: `frontend/src/hooks/useLivePhraseQueue.js`, capped at 3
waiting phrases, dropping the oldest with a visible "fell behind" notice.)

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
- **happy-dom component tests cannot catch real CSS layout bugs** (sizing,
  overflow, `aspect-ratio`/flexbox interactions) — it has no layout engine.
  A `SkeletonCanvas.module.css` bug shipped in Stage 5's first commit and
  went unnoticed through two stages of work with a fully green suite: the
  canvas sized itself off the stage's width, overflowed vertically, and
  every sign rendered as an over-zoomed fragment. Any future rendering/
  canvas/layout work must include a manual visual check in a real browser
  (ideally at both a landscape and a phone viewport) before it's considered
  done — a green test suite alone is not sufficient. Automated visual/layout
  testing (e.g. Playwright) is a deferred follow-up; see `PROJECT_STATUS.md`
  Sections 6, 11 and 12.
- **gitleaks pre-commit hook lives in `.githooks/pre-commit`** (tracked), enabled
  per clone via `git config core.hooksPath .githooks` — `setup_env.sh`/`.ps1` do
  this. It blocks the commit if gitleaks isn't found (set `GITLEAKS_BIN` if it's
  installed but not on PATH, e.g. a shell started before a `winget install`).
  Retroactive full-history scan on 2026-09-25: 33 commits, no leaks.
- **Fingerspelling (2026-09-26): all 26 letters available.** 24 are
  converted from the **MIT-licensed
  `sid220/asl-now-fingerspelling` Hugging Face dataset**
  (`python -m pose_library.convert_hf_fingerspelling`; license record in
  `pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`).
  They're not self-recorded and not synthetic. J and Z aren't in the dataset
  as motion (every sample is one still frame), so they're self-recorded
  (`record_fingerspelling --letters jz`, then `build_fingerspelling`; the
  same commands re-record any letter). See `PROJECT_STATUS.md`
  Sections 13–14 and `pose_library/FINGERSPELLING_PLAN.md` Section 2b.
  Gotchas:
  - **Never fake J/Z motion** from still frames (holding or interpolating
    them). That would misrepresent the sign; a missing letter is skipped
    with a clear message instead.
  - The dataset has **no handedness label** and mixes hands, so don't
    average samples per landmark (that blends mirror images). The
    converter picks each letter's medoid (a real sample).
  - **Each tool only removes letter entries it owns** (by manifest
    `source`): `build_fingerspelling` never deletes a dataset letter, and
    the converter never overwrites a recorded one. A successful recording
    replaces a dataset letter on purpose.
  - The raw dataset download is a gitignored, re-fetchable cache
    (`pose_library/data/asl_now_fingerspelling/`, pinned revision in
    `config.ASL_NOW_REVISION`). Only converted letters are committed.
  - `pose_library/fingerspelling/` is **committed** (dataset-converted and
    self-recorded letters), unlike the gitignored WLASL `pose_library/data/`. The
    `.gitignore` negation `!pose_library/fingerspelling/raw/*.mp4` must stay
    *below* the global `*.mp4` rule or it silently stops applying.
  - A **missing** letter manifest means "alphabet not available" to the
    frontend (skip gracefully); never write an empty `{}` one, which would
    read as "every letter missing". `build_fingerspelling` enforces this.
  - Synthetic placeholder letters live only in
    `tests/fixtures/fingerspelling_synthetic/` (`source:
    "synthetic:placeholder"`). Never copy them into
    `pose_library/fingerspelling/`, and never present them as real
    recordings. The manual browser check points the dev server at them via
    `FINGERSPELLING_POSES_DIR`.
  - Letters and WLASL words must share `extract.py`'s `LANDMARK_NAMES`:
    `stitchTimelines`/`lerpPose` blend poses index by index across the
    boundary, and `SkeletonCanvas` takes its topology from the first unit.
- **Never put raw control bytes (e.g. NUL) in source files** — write them as
  escapes (`"\u0000"`). A raw NUL in `usePoseSequences.js` made git treat
  it as binary, so its diffs were unreviewable, from `7125800` until
  2026-09-26. Invisible characters (zero-width space, soft hyphen, BOM) are
  the same hazard; `\uXXXX` escapes written through some editing tools come
  out as the literal characters, so check new files for them.
- **`npm run lint` only linted `.js` files until 2026-09-26.** ESLint 8's
  `eslint .` skips `.jsx` without `--ext`, so every component went unlinted.
  The script is now `eslint . --ext .js,.jsx,.mjs`. When a lint command
  passes, check it actually covered the files you care about.
- **Live speech (Stage 1-2, 2026-09-26)**: `npm run dev:live` starts the gloss
  server and Vite together; `npm run dev` alone still works but live mode will
  say the translation service isn't running. Gotchas:
  - The gloss server defaults to `gloss_model/checkpoints_v2`;
    `gloss_model/config.CHECKPOINT_DIR` still points at v1, so never rely on
    that default for serving.
  - **Never import torch/transformers before the gloss server is
    listening.** That import has been measured at 4-55s on this machine (55s
    on 2026-09-27); `main()` used to do it first, so the port stayed closed
    and nothing was logged for the whole import, and the frontend reported
    "not running". Now the server binds and logs first, and the import runs
    on the model-load thread while `/api/health` answers `loading` (the
    frontend waits on `loading` with no cap). Guarded by
    `test_cli_answers_health_before_the_model_finishes_loading`. (An earlier
    note here blamed PyTorch holding the GIL. That was wrong: the GIL only
    slows health answers during the import, up to ~2.6s measured, never
    stops them.)
  - A `BaseHTTPRequestHandler` must read the request body before replying,
    or Windows resets the connection and the client never sees the JSON
    error. `gloss_server._read_body` does this first on every POST.
  - Web Speech API audio goes to the browser vendor's speech service (Google
    in Chrome); the UI discloses this before first use. Never remove that.
  - The dev-only `?asr=fake` scripted recognizer (`lib/asr/scriptedRecognizer.js`)
    runs the whole live path without a mic; it's excluded from prod builds by
    `import.meta.env.DEV`.
  - **A real microphone test still needs a person** (see `PROJECT_STATUS.md`
    Section 15).
- **Typed input has two modes (2026-09-28)**. Until then, typed words were
  always looked up literally and never reached the gloss model, a gap in the
  Stage 1-2 plan. Now:
  - **TRANSLATE** (default): `useLiveMode.submitTyped` → the same
    `waitForGlossService` gate → the same `queue.enqueueFinal` as speech, so
    typed and spoken phrases share one queue. Don't build a second
    translation path for typed text.
  - **EXACT WORDS**: the literal lookup, for testing vocabulary. It's the
    only path that calls `setSubmittedWords` directly.
  - **Never fall back to literal signing** when the gloss service is down in
    TRANSLATE mode. That would present untranslated output as a
    translation. Show the service banner (it says EXACT WORDS works without
    the service).
  - App tests that type literal words must select EXACT WORDS first
    (`searchExactWords` in `App.test.jsx`). The default mode now calls the
    gloss client.
  - Typed input exposes model-quality limits more often ("hello" → HALF,
    "taxi" → TITTLE). The gloss line shows exactly what the model produced.
    Words like "where" are fingerspelled because they aren't in the 118-word
    library, which is a vocabulary limit.
  - Full detail: `pipeline/STAGE1_2_PLAN.md` (last section),
    `PROJECT_STATUS.md` Section 17.
- **WebGL rendering can't be unit-tested here -- manual browser check
  required.** The Stage band renders a lit 3D hand with Three.js (WebGL2),
  falling back to the old 2D canvas when no WebGL2 context is available
  (`frontend/RENDERING_UPGRADE_PLAN.md`). happy-dom has no WebGL, so every
  component/App test exercises only the 2D fallback. The pure parts
  (`handGeometry.js`, the 3D camera functions in `skeletonBones.js`) are
  unit-tested, but the rendered image isn't. It's the same kind of gap as
  the CSS layout bug above: any change to `handScene.js`, the radii, the
  colors, or the camera angle must be checked by eye in a real browser. The
  first real look caught four things no test could have: arms too heavy,
  fingers merging in closed handshapes, z-fighting seams where bones meet
  joints, and all-amber joints making fingertips indistinguishable from
  knuckles. Playwright's headless Edge (`channel="msedge"`) renders WebGL2
  fine for saved screenshots, and `--disable-webgl --disable-webgl2`
  forces the 2D fallback for before/after comparisons.
