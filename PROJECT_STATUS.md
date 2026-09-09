# Project Status

Last updated: 2026-09-08 (end of the session that implemented Stage 3).
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
| 3. Gloss translation | **Implementation + infra complete; final quality sign-off pending** | See criteria below and Section 5. |
| 4. Gloss-to-pose lookup | **Not started** | `pose_library/` only has an empty `__init__.py`. This is the next stage to build. |
| 5. Rendering (frontend) | **Not started (tooling only)** | `frontend/package.json` + eslint/prettier/React/Three.js configured and `npm install`-verified, but zero components written. |

### Stage 3 completion criteria (what "done" means here, and what's still open)
Met:
- All planned modules exist and work: `gloss_model/{config,data_prep,evaluate,train,inference}.py`.
- 27 tests in `tests/{test_data_prep,test_evaluate,test_inference}.py`, all passing.
- `ruff` and `black` clean on the whole `gloss_model/` + `tests/` tree.
- `pip-audit` clean.
- Pipeline verified end-to-end twice: once locally on CPU (`--quick`, small-scale
  smoke test) and once for real on Kaggle GPU (full ~81k-row train split, 4 epochs)
  — see Section 3 for the real numbers.
- Portable to Kaggle (`KAGGLE.md`), with the checkpoint-path and device-placement
  bugs found on that run fixed and regression-tested (Section 4).

**Not yet met** — this is what's blocking calling Stage 3 fully "done":
- **You have not yet reviewed the qualitative spot-check output** (15 hand-picked
  sentences, see `gloss_model/evaluate.py::SPOT_CHECK_SENTENCES`) from the real
  Kaggle-trained model. The metrics are strong (Section 3), but PLAN.md's own
  evaluation design says metric scores alone shouldn't be the sole "good enough"
  bar — see Section 5.
- The real, good Kaggle checkpoint has **not been downloaded back into the repo
  yet**. See Section 7 — this matters a lot for what a fresh session should do.

---

## 2. What was actually built in Stage 3 (file by file)

- **`gloss_model/config.py`** — single source of truth: dataset id
  (`achrafothman/aslg_pc12`), seed (42), task prefix, max sequence lengths (32/32),
  train/val/test split fractions (90/5/5), model name (`t5-small`), hyperparameter
  defaults, and a `TrainingConfig` dataclass with a `.quick()` factory for the
  small-scale CPU sanity-check mode. Also resolves `CHECKPOINT_DIR` — defaults to
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

## 3. Key results and metrics

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

**Is this good enough to move on?** Likely yes, but **not yet formally
confirmed** — see Section 5. The known ceiling from PLAN.md still applies:
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

## 5. Open decisions / unresolved questions (nothing here has your sign-off yet)

1. **Is Stage 3 quality sufficient to move on to Stage 4?** The metrics
   (Section 3) look strong, but you haven't reviewed the qualitative spot-check
   output yet, and PLAN.md's evaluation design explicitly treats that read-through
   as necessary, not optional, before calling a result "good enough" — metrics
   alone can hide a model that's faithfully reproducing ASLG-PC12's rule-based
   quirks rather than producing gloss a signer would find natural. **Next
   session's first job, if this hasn't been discussed since, is to surface the
   spot-check output for your review** (rerun `run_spot_check` against the real
   checkpoint once it's local — see Section 6 step 1 — or pull the log output
   from the Kaggle notebook if you still have it).
2. **The good checkpoint isn't in the repo yet.** `gloss_model/checkpoints/`
   locally still holds the *weak* CPU `--quick` model (500 examples/1 epoch),
   not the strong Kaggle-trained one. Nobody has decided/executed how the real
   checkpoint gets from Kaggle's Output tab into this repo's expected location
   (or into some other storage, e.g. a Kaggle Model/Dataset, if 242MB+ in git is
   undesirable — that tradeoff hasn't been discussed either).
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

1. **Get a sign-off on Stage 3 quality** before writing any Stage 4 code (per
   CLAUDE.md's build-order rule: "don't move to the next pipeline stage until
   the current stage's tests pass" — tests passing is necessary but PLAN.md's
   own bar also wants the spot-check read). Concretely: bring the real Kaggle
   checkpoint's spot-check output (or the checkpoint itself, see step 2) back
   for review, and get an explicit "yes, good enough" or "no, needs more
   training/data work" from the user.
2. **Bring the real Kaggle checkpoint back into the repo** (or decide it
   shouldn't live in git — that's an open question, see Section 5.2):
   download `gloss_model_checkpoints/` from the Kaggle notebook's Output tab,
   place it at `gloss_model/checkpoints/` locally (overwriting the current weak
   `--quick` one), per `KAGGLE.md` Section 6. After that,
   `tests/test_inference.py::test_translate_returns_nonempty_string_with_real_checkpoint`
   will exercise the real model instead of the quick-run one.
3. Only after (1) and (2): **start Stage 4** (pose-to-gloss lookup library).
   Per CLAUDE.md's `ml-engineer` agent responsibilities: use MediaPipe Hands
   (and Holistic if needed) to extract keypoints from WLASL videos, store
   sequences as lightweight JSON (not raw video) in `pose_library/data/`
   (already gitignored), build `get_pose_sequence(gloss_word: str) -> PoseSequence | None`,
   and implement the fingerspelling fallback for OOV words before considering
   Stage 4 done — this last part matters more than usual here, since Section 1
   of `gloss_model/PLAN.md` confirmed ASLG-PC12 has **zero** fingerspelling/
   classifier annotation, so the fallback will carry real load, not be a rare
   edge case. Write tests in `tests/` alongside, per CLAUDE.md's testing rule.
4. Lower priority, whenever convenient: add the README.md limitation note
   (Section 5.4), and check ASLG-PC12's license (Section 5.3).

---

## 7. Environment / setup facts for a fresh session

- **Python**: 3.13.0, in a local venv at `.venv/` (create via `setup_env.ps1` /
  `setup_env.sh`). Local `torch` is CPU-only (`torch==2.14.0+cpu`,
  `cuda available: False`) — this machine has no GPU.
- **Node**: v22.11.0, npm 10.9.0. `frontend/` has `package.json` with
  React 18, Three.js, Vite 6.4.3, ESLint 8 + prettier configured and installed
  (0 `npm audit` vulnerabilities as of last check) — but no components written yet.
- **Checkpoint location**: `gloss_model/checkpoints/` (gitignored, never
  committed — 242MB+ of model weights). **As of this writing, that directory
  holds the weak CPU `--quick` model, not the real Kaggle-trained one** — see
  Section 5.2. `gloss_model/config.py::CHECKPOINT_DIR` resolves automatically:
  the package-relative path locally, `/kaggle/working/gloss_model_checkpoints/`
  on Kaggle (detected via that path's existence).
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
