# Project Status

Last updated: 2026-09-10 (Stage 3 formally signed off — see Section 9).
This is a living status doc — update it at the end of a session with
meaningful progress. `CLAUDE.md` stays static (conventions/architecture/
security rules); this file tracks what's actually been built and what's left.

If you're a fresh Claude Code session with no memory of this project: read
this file, then `CLAUDE.md`, then `git log --oneline` — that should be enough
to pick up exactly where things stand.

---

## 1. Current state by pipeline stage

| Stage | Status | Notes |
|---|---|---|
| 1. ASR (speech-to-text) | **Not started** | `pipeline/asr.py` doesn't exist yet; `pipeline/` only has an empty `__init__.py`. |
| 2. VAD chunking | **Not started** | `pipeline/vad.py` doesn't exist yet. |
| 3. Gloss translation | **Done — signed off 2026-09-10** | See Section 9. `gloss_model/checkpoints_v2/` is the model to use going forward, superseding v1. |
| 4. Gloss-to-pose lookup | **Not started — this is the next thing to build** | `pose_library/` only has an empty `__init__.py`. |
| 5. Rendering (frontend) | **Not started (tooling only)** | `frontend/package.json` + eslint/prettier/React/Three.js configured and `npm install`-verified, but zero components written. |

### Stage 3 completion criteria — all met, signed off 2026-09-10 (see Section 9)
- All planned modules exist and work: `gloss_model/{config,data_prep,evaluate,train,inference}.py`.
- 27 tests in `tests/{test_data_prep,test_evaluate,test_inference}.py`, all passing.
- `ruff` and `black` clean on the whole `gloss_model/` + `tests/` tree.
- `pip-audit` clean.
- Pipeline verified end-to-end multiple times: locally on CPU (`--quick`,
  small-scale smoke test) and twice for real on Kaggle GPU — the original full
  run (Section 3) and the v2 retrain that combined the `max_length` fix and
  vocabulary augmentation (Section 9, which supersedes Section 3's numbers).
- Portable to Kaggle (`KAGGLE.md`), with every bug found along the way (Section 4,
  Section 9) fixed and regression-tested where applicable.
- Qualitative spot-check and vocabulary-generalization-check output **reviewed
  and signed off** — see Section 9 for the actual results (15/15 regular,
  7/8 generalization, one understood remaining case).
- The good checkpoint (v2) **is present locally** at `gloss_model/checkpoints_v2/`
  (gitignored, never committed to git, but sitting on disk in the project
  folder) — see Section 7.

---

## 2. What was actually built in Stage 3 (file by file)

- **`gloss_model/config.py`** — single source of truth: dataset id
  (`achrafothman/aslg_pc12`), seed (42), task prefix, max sequence lengths
  (48 source / 80 target, since Section 9 — see the correction in Section 8/9,
  the original 32/32 was miscalibrated), train/val/test split fractions
  (90/5/5), model name (`t5-small`), augmentation settings, and a
  `TrainingConfig` dataclass with `.quick()`/`.patch()` factories for the
  small-scale CPU sanity-check and continued-fine-tuning modes. Also resolves
  `CHECKPOINT_DIR` — defaults to
  `gloss_model/checkpoints/` locally, or `/kaggle/working/gloss_model_checkpoints/`
  when running on Kaggle (auto-detected).
- **`gloss_model/data_prep.py`** — `load_raw_dataset()` (thin wrapper over HF
  `load_dataset`), `clean_text()` (strips BOM + whitespace), `build_splits()`
  (dedupes exact `(text, gloss)` pairs, then splits 90/5/5 with a fixed seed),
  `preprocess()` (tokenizes with the T5 task prefix, respects max lengths),
  `limit_dataset()` (caps row count for quick-mode runs).
- **`gloss_model/evaluate.py`** — `exact_match()` and `token_f1()` (primary
  metrics), `bleu()` (secondary, via `evaluate`/`sacrebleu`), `naive_baseline()`
  (uppercase + stopword-drop, a comparison floor), `decode_predictions()` (cleans
  `-100` out of both predictions and labels before decoding — see Section 4),
  `compute_metrics_fn()` (builds the `Seq2SeqTrainer.compute_metrics` callback),
  and `run_spot_check()` (generates gloss for 15 hand-picked English sentences,
  for manual eyeballing — not asserted on automatically).
- **`gloss_model/train.py`** — CLI entry point (`if __name__ == "__main__":`).
  `--quick` flag for the small/fast CPU sanity run; `--epochs`/`--batch-size`/
  `--max-train-samples`/`--max-eval-samples`/`--output-dir` overrides for a real
  run. Builds a `Seq2SeqTrainer`, trains, saves the checkpoint, evaluates on the
  held-out test split, then runs and logs the qualitative spot check.
- **`gloss_model/inference.py`** — `load_model()` (loads a fine-tuned checkpoint
  + tokenizer) and `translate(text, model, tokenizer)` (the function later stages
  will call). Enforces input sanitization per CLAUDE.md's security rules:
  rejects empty input (`EmptyInputError`) and input over `MAX_INPUT_CHARS=500`
  (`InputTooLongError`).
- **`gloss_model/PLAN.md`** — the full design doc: dataset inspection findings,
  preprocessing rationale, hyperparameter choices, evaluation design, known
  limitations, and an implementation log. Read this for the "why" behind any of
  the above; this status file only summarizes.
- **`tests/test_data_prep.py`**, **`tests/test_evaluate.py`**,
  **`tests/test_inference.py`** — 27 tests total. Notably include two mocked
  regression tests (no GPU required) for the device-placement bugs in Section 4,
  each verified to fail without its fix and pass with it.
- **`KAGGLE.md`** — hand-off steps for running the real training pass on Kaggle
  (what to upload, notebook settings, setup cell, training command, where output
  lands, how to bring the checkpoint back). See Section 7 for the exact naming
  convention used.

---

## 3. Key results and metrics (v1 — superseded by Section 9's v2 numbers)

Kept as historical record of the first real training run. **v2
(`gloss_model/checkpoints_v2/`) is the model to use going forward** — see
Section 9.

**Local CPU `--quick` smoke test** (500 train / 100 val / 100 test rows, 1 epoch,
batch size 8) — purpose was only to prove the pipeline runs end-to-end, not to
produce a good model: `test_token_f1=0.313`, `test_bleu=4.99`, `test_exact_match=0.0`.
Spot-check output was gibberish/repetitive, as expected for that budget.

**Real Kaggle GPU run** (full ~81k-row train split, 4 epochs — the actual
fine-tuning pass):

| Metric | Value |
|---|---|
| `test_token_f1` | **0.965** |
| `test_bleu` | **93.7** |
| `test_exact_match` | **0.80** |

**What this means**: token-F1 of 0.965 means the model is getting almost all the
right gloss words; exact-match of 0.80 means 80% of test sentences produce the
*exact* reference gloss string (whitespace/case-normalized). This is a strong
first result — well above the "naive baseline clears comfortably, token-F1
0.7+" bar PLAN.md set as a loose first-pass target.

**Is this good enough to move on?** **Yes — formally confirmed, see Section 9.**
The known ceiling from PLAN.md still applies:
ASLG-PC12 is rule-generated (not human-translated), so even a model that
matches the test set well is learning that corpus's own rule-based
gloss conventions, not necessarily gloss a fluent signer would consider
natural. The metrics say the model learned the corpus; they don't by
themselves say the corpus is a great target. The qualitative spot-check
(15 sentences, see `gloss_model/evaluate.py::SPOT_CHECK_SENTENCES`) is the
check meant to catch that, and it hasn't been read through yet by a human.

---

## 4. Known bugs found and fixed this session (brief)

1. **`OverflowError: can't convert negative int to unsigned`** in
   `evaluate.py::decode_predictions`, hit during the local CPU test-split eval.
   `Seq2SeqTrainer` pads variable-length generated sequences across accumulated
   eval batches with `-100` (the label-ignore sentinel), not the tokenizer's pad
   id — so `predictions` can carry `-100` just like `labels` does. Only `labels`
   was being cleaned. Mattered because it silently depends on eval-batch count/
   sequence-length variance — it passed on a small validation split and only
   crashed on the larger test split, so it's exactly the kind of bug that looks
   fine until you scale up (which is what happened again with the next two bugs,
   on Kaggle). Fixed by applying the same `-100`-cleanup to predictions too.
2. **Kaggle checkpoint-path assumption**: `config.py`'s `CHECKPOINT_DIR` defaulted
   to a path relative to the `gloss_model` package's own file location. On
   Kaggle, if this repo is attached as a notebook input, that mounts read-only
   under `/kaggle/input/...` — so the default would try to write a checkpoint
   into read-only storage. Mattered because it would have silently failed (or
   required remembering a `--output-dir` flag) on the very first real training
   run. Fixed by detecting `/kaggle/working` and defaulting there when present.
3. **Two device-placement bugs**, found on the actual Kaggle GPU run (after
   training and test-eval had already succeeded): `evaluate.py::run_spot_check`
   and `inference.py::translate` both tokenized input and called
   `model.generate()` without moving the tokenized tensors to `model.device`
   first. Crashed with `RuntimeError: Expected all tensors to be on the same
   device` once the model was on `cuda:0` instead of CPU. Mattered because
   the pipeline had only ever been run on CPU before this, so nothing in local
   testing could have caught it. Fixed identically in both places (`.to(model.device)`
   chained onto the tokenizer output); both have mocked regression tests
   (no GPU needed) that were verified to fail without the fix and pass with it.

Full detail on bug #1 and the dataset findings is in `gloss_model/PLAN.md`; all
three are also logged in `CLAUDE.md`'s "Known gotchas / decisions log".

---

## 5. Open decisions / unresolved questions

1. ~~Is Stage 3 quality sufficient to move on to Stage 4?~~ **Resolved
   2026-09-10, see Section 9** — reviewed and signed off.
2. ~~The good checkpoint isn't in the repo yet.~~ **Resolved** — v1 was
   downloaded to `gloss_model/checkpoints/` earlier, and v2 (the current model)
   is at `gloss_model/checkpoints_v2/`. Both are gitignored (never committed —
   242MB+ each), present on disk only.
3. **Licensing of ASLG-PC12** (flagged in PLAN.md, never resolved): not checked
   against the HF dataset page or the original achrafothman.net release. Not
   blocking local prototyping, but should happen before any public release.
4. **README.md limitation note**: PLAN.md documents the rule-generated-corpus
   quality ceiling, but you asked for that to also land in the top-level README
   "eventually" (not yet — this hasn't been done).
5. **Stage 4 approach itself is entirely undiscussed** — no decisions yet on
   MediaPipe extraction specifics, how much of WLASL to pull, or the
   fingerspelling-fallback design, beyond what CLAUDE.md's `ml-engineer` agent
   description already states as responsibilities.

---

## 6. Exact next steps, in order

Stage 3 is done (Section 9) — **Stage 4 (pose-to-gloss lookup library) is the
immediate next step**, per CLAUDE.md's build order and its `ml-engineer` agent
responsibilities:

1. **Set up WLASL access**: figure out how to obtain the WLASL video dataset
   (dxli94/WLASL on GitHub — check its actual current download process/license
   before assuming anything, the same way ASLG-PC12's provenance turned out to
   need verifying rather than trusting secondhand descriptions).
2. **Extract keypoints with MediaPipe**: use MediaPipe Hands (and Holistic if
   facial/body context turns out to matter) to pull pose sequences from WLASL
   videos. `mediapipe` is already in `requirements.txt` and installed/verified
   in the local venv (Stage 3's environment setup), so no new environment work
   should be needed here.
3. **Store as lightweight JSON, not raw video**, in `pose_library/data/`
   (already gitignored — same "don't commit large binaries" pattern as the
   gloss checkpoints).
4. **Build the lookup interface**: `get_pose_sequence(gloss_word: str) -> PoseSequence | None`.
5. **Implement the fingerspelling fallback for out-of-vocabulary gloss words
   before considering Stage 4 done** — this matters more than usual here,
   since `gloss_model/PLAN.md` Section 1 confirmed ASLG-PC12 has **zero**
   fingerspelling/classifier annotation, so gloss output from Stage 3 will
   essentially never itself contain fingerspelling markers — the fallback in
   Stage 4 has to detect OOV gloss words on its own and carry real load, not
   handle a rare edge case.
6. **Write tests in `tests/` alongside**, per CLAUDE.md's testing rule — new
   modules in `pose_library/` need corresponding tests before moving on, same
   as Stage 3.

Lower priority, whenever convenient (not blocking Stage 4):
- Add the README.md limitation note about ASLG-PC12 being rule-generated,
  Europarl-register text (Section 5.4/`VOCAB_DIAGNOSIS.md`).
- Check ASLG-PC12's actual license terms (Section 5.3) before any public release.

---

## 7. Environment / setup facts for a fresh session

- **Python**: 3.13.0, in a local venv at `.venv/` (create via `setup_env.ps1` /
  `setup_env.sh`). Local `torch` is CPU-only (`torch==2.14.0+cpu`,
  `cuda available: False`) — this machine has no GPU.
- **Node**: v22.11.0, npm 10.9.0. `frontend/` has `package.json` with
  React 18, Three.js, Vite 6.4.3, ESLint 8 + prettier configured and installed
  (0 `npm audit` vulnerabilities as of last check) — but no components written yet.
- **Checkpoint location**: **`gloss_model/checkpoints_v2/` is the current
  model to use** (gitignored, never committed — 242MB+ of model weights;
  present on disk only). `gloss_model/checkpoints/` holds v1 (the original
  full run, pre-`max_length`-fix, pre-vocabulary-augmentation) — kept for
  comparison, not deleted, but superseded; see Section 9.
  `gloss_model/config.py::CHECKPOINT_DIR` resolves automatically: the
  package-relative `gloss_model/checkpoints/` path locally (i.e. v1's
  location — pass `--init-checkpoint`/`checkpoint_dir=` explicitly to use v2),
  or `/kaggle/working/gloss_model_checkpoints/` on Kaggle (detected via that
  path's existence).
- **Kaggle**: no remote git repo configured for this project (`git remote -v`
  is empty) — the hand-off method used was zipping git-tracked source
  (`git archive --format=zip -o asl-live-translator-src.zip HEAD gloss_model requirements.txt`)
  and uploading it as a Kaggle Dataset named `asl-live-translator-src`, attached
  to the training notebook as an input (mounted read-only at
  `/kaggle/input/asl-live-translator-src/`). Full steps, including notebook
  settings and the exact setup/training commands, are in `KAGGLE.md`. No Kaggle
  account details, notebook name, or dataset version are recorded anywhere in
  this repo — if you need to find the notebook again, that's on kaggle.com under
  your own account, not discoverable from this repo.
- **Credentials**: none required for Stage 3 as built. `evaluate.load("sacrebleu")`
  and `load_dataset(...)` make unauthenticated HF Hub requests (you'll see a rate-limit
  warning, not an error) — setting an `HF_TOKEN` env var is optional, not required.
  `.env.example` at the repo root lists variables for *later* stages
  (`DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY` for Stage 1) — none of those are
  needed yet for Stage 3 or the upcoming Stage 4.
- **Dependency pins**: `requirements.txt` pins `transformers==5.16.1`,
  `datasets==5.0.1`, `accelerate==1.14.0`, `evaluate==0.4.6`, `sacrebleu==2.6.0`
  exactly (not `>=`) because pipeline behavior was verified against these exact
  versions (see bug #1 in Section 4). `torch` is intentionally left as `>=2.2`
  with a comment not to force-reinstall it on Kaggle (would risk clobbering
  Kaggle's preinstalled CUDA-enabled build).
- **Git**: local repo only, no remote. Latest commit at the time of writing is
  `4750dd0` — run `git log --oneline` for the current head.

---

## 8. Vocabulary patch results and generation-length finding (2026-09-09)

Since Section 5/6 above were written, the vocabulary-gap patch (see
`gloss_model/VOCAB_DIAGNOSIS.md`) was built, reviewed, and run on Kaggle via
`train.py --patch`. **It succeeded and improved on the original run**:
`test_token_f1=0.970` (was 0.965), `test_bleu=94.7` (was 93.7),
`test_exact_match=0.835` (was 0.80). The generalization check
(`evaluate.VOCAB_GENERALIZATION_SENTENCES`) showed real vocabulary
generalization on most held-out novel-structure sentences, not memorization
of the repeated augmented phrasings. This doc's Sections 1/3/5/6 above
predate this and are now stale on that point — not rewritten here, since
that wasn't asked for this session; treat Section 8 as the current word on
the patch/checkpoint status until someone reconciles them.

Two things came out of reviewing that run's output:

1. **A real, corpus-wide `max_length` miscalibration, bigger than one bad
   spot-check line.** Two generalization-check outputs looked broken: one
   garbled ("Skipping..." → "SPIPP...") and one cut off mid-sentence
   ("...gave them back." → "...GIVE X-"). Investigated by actually tokenizing
   with the real T5 tokenizer rather than assuming:
   - The cutoff one is a genuine `max_length=32` truncation. A plausible
     correct gloss for that sentence tokenizes to **40 subword tokens** —
     `MAX_TARGET_LENGTH=32` (`gloss_model/config.py`) cuts it off exactly
     where the bad output stops.
   - This isn't specific to that one sentence. `PLAN.md`'s original p99
     figures (21 text / 20 gloss) that justified `MAX_SOURCE_LENGTH`/
     `MAX_TARGET_LENGTH=32` were computed by **whitespace word-splitting**,
     never checked against actual T5 subword tokenization. Doing that check
     properly on the real training split: source tokens (with task prefix)
     have p50/p90/p99/max = 26/32/40/76, **9.75% over 32**; gloss target
     tokens have p50/p90/p99/max = **36/51/65/129, with 59.20% of all
     training targets exceeding 32 tokens** and being silently truncated by
     `truncation=True` in `data_prep.preprocess()`. This has been true since
     the very first training run (the CPU `--quick` smoke test and the
     original full Kaggle run both trained under this same silent
     truncation) — it isn't something the patch introduced.
   - Root cause: the custom `DESC-`/`X-` gloss notation isn't in T5's
     pretrained vocabulary, so SentencePiece fragments each occurrence into
     several subword pieces (`DESC-NEVER`, `DESC-BACK`, etc. each cost 3+
     tokens, not 1) — a sentence with several such tokens blows well past a
     word-count-based length estimate.
   - The other bad output ("SPIPP") is **not** length-related — truncation
     only cuts off the end of a sequence, and this garbling is at the start.
     See point 2 below; it's the same category of issue as "apartment."
   - **No code or config changed for this** — the user asked only to
     investigate, not fix. Whoever picks this up next should decide whether
     to raise `MAX_SOURCE_LENGTH`/`MAX_TARGET_LENGTH` (e.g. to 48 or 64,
     re-checked against real subword-token percentiles, not word counts) and
     retrain, given how much of the corpus's supervision signal has been
     silently cut short.
2. **"apartment" → "APPEAL" and "skip"/"skipping" → "SPIPP" are known,
   expected vocabulary gaps, not bugs.** Both words are essentially absent
   from the real corpus (`apartment`=2, `apartments`=1, `skip`=1,
   `skipping`=0, `skipped`=0 occurrences) and were never part of the 121-word
   augmentation list. This confirms the vocabulary gap documented in
   `VOCAB_DIAGNOSIS.md` is broader than the specific words patched — expected
   and already understood, not a new problem. No action needed right now.

---

## 9. v2 retrain: Stage 3 formally signed off (2026-09-10)

Following Section 8's `max_length` finding, `MAX_SOURCE_LENGTH`/
`MAX_TARGET_LENGTH` were raised to 48/80 (from 32/32) and combined with the
vocabulary augmentation into **one from-scratch retrain** (`t5-small`, full 4
epochs, `vocab_augmentation.csv` mixed in from the start) rather than another
`--patch` layered on top of v1 — see the chat session for the latency
benchmark (forced full-length generation, ~144ms/sentence worst case on CPU
at 80 tokens vs. ~50ms at 32, well inside the pipeline's ~1-2s lag budget)
that justified the length increase before spending the GPU time.

**The v2 run succeeded.** Test metrics:

| Metric | v1 | v2 |
|---|---|---|
| `test_bleu` | 93.7 | **93.9** |
| `test_token_f1` | 0.965 | **0.966** |
| `test_exact_match` | 0.80 | **0.812** |

v2's numbers essentially match or slightly exceed v1's — despite being graded
against **far less truncated ground truth** (recall from Section 8: 59% of
v1's targets were silently cut at 32 tokens, inflating v1's apparent accuracy
on longer sentences since both reference and generation were truncated at the
same point). Matching v1 under a fairer, harder evaluation is a real quality
improvement, not a wash.

**Qualitative review** (the actual basis for sign-off, per PLAN.md's own
evaluation design — metrics alone were never treated as sufficient):

- **Regular spot-check: 15/15 clean.** No truncation, no garbling anywhere,
  including "weather" (`WEATHER BE DESC-NICE TODAY.` — no more "WEST"
  hallucination).
- **Vocabulary generalization check: 7/8 clean**, including two sentences
  that were broken pre-patch: "weather" (`X-WE CANCEL TRIP BECAUSE
  WEATHER.`) and "skip" (`SKIP BREAKFAST DESC-ALWAYS MAKE X-I DESC-HUNGRY BY
  NOON` — "SKIP" renders correctly now, was garbled "SPIPP"/"SIPP" before).
- **One remaining case, understood, not a config bug**: "My roommate
  borrowed my keys and never gave them back." still stops early at
  `GIVE X-`. Confirmed directly (not assumed) that this is *not* the
  `max_length` truncation recurring: `max_length=48` and `max_length=80`
  produce the **identical 33-token output** — if the ceiling were still the
  bottleneck, raising it further would have let generation continue. It
  didn't. The model itself is choosing to emit EOS early on this specific
  input. That's a narrow generalization limit on one held-out sentence, not
  the systemic issue Section 8 diagnosed and this retrain fixed.
- **"apartment" → "APPEAL" persists, unchanged** — still the same
  known/expected out-of-scope vocabulary gap from Section 8, not new.

**Latent footgun noted, not fixed**: `gloss_model/checkpoints_v2/generation_config.json`
still carries T5's stock `"max_length": 20` default — training/saving never
updates it to match `config.MAX_TARGET_LENGTH`. Harmless today because
`train.py`, `inference.py`, and `evaluate.py` all pass `max_length` explicitly
to `generate()` rather than relying on the checkpoint's baked-in default, but
any future caller that doesn't pass it explicitly would silently get
20-token generations. Worth fixing (e.g. setting `model.generation_config.max_length`
before `save_model()`) whenever `gloss_model/train.py` is touched again, but
not blocking anything right now.

**Sign-off**: Stage 3 is done. `gloss_model/checkpoints_v2/` is the model to
use going forward, superseding v1 (`gloss_model/checkpoints/`, kept for
comparison, not deleted). Per CLAUDE.md's build-order rule, **Stage 4 (pose
library) is next** — see Section 6.
