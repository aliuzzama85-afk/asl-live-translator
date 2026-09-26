# Project Status

Last updated: 2026-09-26 (fingerspelling: **24 of 26 letters now real**,
from the MIT-licensed `sid220/asl-now-fingerspelling` dataset; **J and Z
still missing** — see Section 14, which updates Section 13). Build-order step 3,
multi-word playback with cross-sign interpolation, done 2026-09-25 — see
Section 12.
Stage 5 formally signed off 2026-09-22 — see Section 11; Stage 4 also
complete — see Section 10.
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
| 4. Gloss-to-pose lookup | **Done — see Section 10** | 118-word pose library built from WLASL + MediaPipe, 61 tests passing. Fingerspelling fallback: built, 24/26 letters from the MIT-licensed asl-now dataset, J and Z missing (Sections 13–14). The WLASL/C-UDA licensing question was resolved 2026-09-26 (local/dev-only indefinitely) — see Section 5. |
| 5. Rendering (frontend) | **Done — signed off 2026-09-22, see Section 11** | Single-word skeleton renderer, feature-complete per `frontend/PLAN.md`'s v1 scope, 25/25 tests passing (`a024993`). Multi-word playback with cross-sign interpolation (build-order step 3) built on top of this — see Section 12. |

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
3. **Licensing of ASLG-PC12 — checked 2026-09-25, license identified but a
   deeper provenance question is now open, not resolved.**
   `gloss_model/config.py:25` sources the dataset as
   `achrafothman/aslg_pc12` via HF `datasets.load_dataset()`.
   - **The HF dataset page's own card is internally inconsistent**: its
     structured metadata (`README.md` YAML frontmatter, driving the sidebar
     badge) declares `license: cc-by-nc-4.0`, but the card's own prose
     "Licensing Information" section says "More Information Needed" — same
     for "Curation Rationale," "Source Data," "Dataset Curators." The card
     was added to the Hub by a third party (`@AmitMY`), not the original
     authors.
   - **Checked the authoritative primary source directly**:
     `achrafothman.net/site/asl-smt/` (Dr. Achraf Othman's own release page,
     linked as the HF card's "Homepage") states verbatim: *"English-ASL
     Gloss Parallel Corpus 2012: ASLG-PC12 by Dr. Achraf Othman is licensed
     under Attribution-NonCommercial 4.0 International."* This corroborates
     the HF tag from the primary source (CC BY-NC 4.0, no ShareAlike),
     despite the empty card section.
   - **What CC BY-NC 4.0 actually says** (read from the [legal
     code](https://creativecommons.org/licenses/by-nc/4.0/legalcode), not a
     summary): attribution is required (Section 3(a)); the license grants
     rights "for NonCommercial purposes only" (Section 2(a)(1)), covering
     *both* the original material and "Adapted Material" (material derived
     from/based on it). Whether a model fine-tuned on this data legally
     counts as "Adapted Material" under copyright law is an unsettled,
     interpretive question — not resolved here, not a judgment this doc
     makes. If it does count, the license's own text restricts sharing it to
     NonCommercial purposes, same as the source data. No research-only
     carve-out or public-domain claim exists anywhere in either source — a
     standard CC BY-NC 4.0 grant, nothing narrower or broader.
   - **The `VOCAB_DIAGNOSIS.md` Europarl finding changes the picture, and is
     not resolved by the above.** `VOCAB_DIAGNOSIS.md`'s addendum
     established that this specific 87,710-row dataset's English text reads
     as Europarl (EU Parliament proceedings), not Gutenberg literary text as
     Othman & Jemni's own paper claims its source is. Checked Europarl's own
     terms directly at `statmt.org/europarl/`: *"We are not aware of any
     copyright restrictions of the material"* — a disclaimer of awareness,
     not an affirmative open-license grant, and a materially weaker basis
     than a real license. Othman's CC BY-NC 4.0 claim is over *his own
     compiled/annotated corpus*, premised (per his paper) on the underlying
     English text coming from Gutenberg. If the actual underlying text is
     Europarl proceedings instead, that's a different, unverified chain of
     title: it's unclear he held clean rights to apply his own license over
     text that — per this project's own independent finding — doesn't match
     what he described it as. Neither the CC BY-NC 4.0 grant nor Europarl's
     "not aware of restrictions" statement closes this gap.
   - **Net effect**: the license *label* is now identified with reasonable
     confidence (CC BY-NC 4.0, checked at the primary source, not just
     inferred from the HF tag) — but the provenance mismatch this project
     independently discovered raises a real, unresolved question about
     whether that label was validly applied to begin with. **Not fully
     resolved; needs the user's explicit sign-off before any public release**
     — not blocking local prototyping.
4. ~~README.md limitation note~~ **Resolved 2026-09-22** — added to
   README.md (rule-generated, Europarl-register-corpus quality ceiling).
5. ~~Stage 4 approach itself is entirely undiscussed~~ **Resolved** — Stage 4
   was designed (`pose_library/PLAN.md`) and built; see Section 10.
6. ~~WLASL/C-UDA licensing ambiguity (Stage 4)~~ **Resolved 2026-09-26** —
   see the resolution note at the end of this item. Original analysis, kept
   for the record: `pose_library/PLAN.md` Section 1 flags a genuine ambiguity: whether an
   extracted keypoint sequence counts as a C-UDA "Result" (unrestricted) or
   still "Data" in modified form (restricted) isn't settled by the license
   text's own de-minimis test, and WLASL's own README separately states "no
   commercial usage is allowed" in plain terms. Current policy is local-only/
   gitignored/never redistributed, but this needs the user's explicit
   sign-off before any public release or commercial pivot — flagging again,
   not resolved here. See Section 10.

   **Resolved 2026-09-26** — decision made to keep this project
   local/dev-only indefinitely, never serving or redistributing
   `pose_library/data/` contents publicly. This usage is squarely
   "Computational Use" per C-UDA and within WLASL's own academic-use
   framing. If this decision is ever revisited (e.g. a public demo), the
   unresolved questions documented above (Data vs Result classification)
   and in `pose_library/PLAN.md` (unverified third-party video-source
   terms, Section 1 and its open questions) would need to be revisited
   before proceeding.
7. **Fingerspelling alphabet — 24 of 26 letters available (2026-09-26);
   J and Z missing.** Infrastructure built (Section 13). 24 letters are
   converted from the MIT-licensed `sid220/asl-now-fingerspelling` dataset
   (Section 14). J and Z aren't in it as motion, so they need recording
   (`record_fingerspelling --letters jz`). Words containing them are skipped
   with "NO FINGERSPELLING FOR "J"" until then.
8. ~~Multi-word / cross-sign interpolation — not started.~~ **Resolved
   2026-09-25** — designed (`frontend/MULTIWORD_PLAN.md`, `f4fdc82`) and
   built (`7125800`), 40/40 tests passing. See Section 12.
9. **Stage 1/2 (live ASR + VAD) — not started**, per CLAUDE.md's build order
   (step 4, after step 3 above).

---

## 6. Exact next steps, in order

Stage 4 (Section 10), Stage 5 (Section 11), and build-order step 3
(multi-word playback with cross-sign interpolation, Section 12) are all
done. Per CLAUDE.md's build order, the next official step is **step 4:
live speech input** (Stage 1/2 — streaming ASR + Silero VAD) wired to the
now-working core. Nothing about this has been designed yet:

1. **Design Stage 1/2**: streaming ASR (Deepgram or AssemblyAI,
   `pipeline/asr.py`) and Silero VAD-based phrase chunking
   (`pipeline/vad.py`), per `CLAUDE.md`'s architecture section — including
   the in-memory queue so playback never blocks on translation, and how a
   live phrase's gloss output feeds the multi-word sequence
   `frontend/MULTIWORD_PLAN.md`/Section 12 already built the playback side
   for (that side takes an already-glossed word sequence; this step is
   about producing one live, not consuming it).
2. **Write tests alongside**, per CLAUDE.md's testing rule.

Loose ends, not blocking step 4 but worth closing out when convenient:
- **"Phantom static hand" rendering characteristic** (Section 12): a
  one-handed word's unused hand renders as a static cluster at the
  projected origin instead of being invisible, confirmed pre-existing in
  Stage 5's single-word playback too, not just multi-word transitions. Not
  fixed; a real fix (skip-drawing a zero-sentinel hand, or fading it) is
  future work if it turns out to matter visually in practice.
- **Record J and Z** (Section 5 item 7, Section 14): the only data still
  missing for fingerspelling, and a manual step. Run
  `python -m pose_library.record_fingerspelling --letters jz`, then
  `python -m pose_library.build_fingerspelling`, review in the browser, and
  commit `pose_library/fingerspelling/`. Separately, the feature still needs
  a check by someone who knows ASL fingerspelling (Section 14).
- ~~WLASL/C-UDA licensing decision~~ **Resolved 2026-09-26** (Section 5
  item 6): local/dev-only indefinitely, `pose_library/data/` never served
  or redistributed publicly.
- **ASLG-PC12 licensing sign-off** (Section 5 item 3): license identified
  (CC BY-NC 4.0, checked 2026-09-25) but the Europarl-provenance question it
  raised needs the user's explicit sign-off before any public release.
- **Automated layout/visual regression testing (e.g. Playwright)** — not
  yet added; manual browser checks are the only current safeguard against
  this class of bug (see the Section 11 amendment).

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
- **Kaggle**: at the time of the Kaggle runs no remote git repo was
  configured (one exists now — see **Git** below), so the hand-off method used
  was zipping git-tracked source
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
- **Git**: pushed to GitHub at
  `https://github.com/aliuzzama85-afk/asl-live-translator` (`origin`); local
  `main` tracks and matches `origin/main`. Latest commit at the time of
  writing is `5b9c753` — run `git log --oneline` for the current head.

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

---

## 10. Stage 4: pose library — built and complete (2026-09-11)

What was built (`pose_library/`), file by file:

- **`config.py`** — paths/constants (`POSES_DIR`, `MANIFEST_PATH`, etc.).
- **`wlasl_metadata.py`** — parses `WLASL_v0.3.json`; `build_gloss_index()`,
  `extract_vocab_stems()` (strips `DESC-`/`X-` prefixes and pronoun markers so
  gloss tokens can be compared against `gloss_model`'s augmentation
  vocabulary), `select_instance()` (prefers a direct `.mp4` URL over YouTube/
  `.swf`/gated sources, per `pose_library/PLAN.md` Section 1's real
  source-diversity findings across WLASL's 19 hosting domains).
- **`build_target_vocab.py`** — cross-references `gloss_model`'s
  `vocab_augmentation.csv` against the real WLASL gloss list, splits into
  matched/unmatched, writes `pose_library/data/target_vocab.json`.
- **`download.py`** — downloads and trims WLASL video instances.
- **`extract.py`** — MediaPipe Holistic extraction to a 48-landmark
  `PoseSequence` per word (21 left-hand + 21 right-hand + 6 pose-subset
  shoulder/elbow/wrist points); drops (doesn't zero-fill or interpolate)
  frames where neither hand is detected.
- **`types.py`** — the frozen `PoseSequence` dataclass (`to_dict`/`from_dict`/
  `as_array`).
- **`manifest.py`** — `compute_quality_flags()` (the `low_confidence`/
  `quality_notes` heuristic: frame count, retention %, and mid-clip-gap
  thresholds picked from the real distribution across the first full build,
  not round guessed numbers), `build_entry()`, `load_manifest()`/
  `save_manifest()`.
- **`lookup.py`** — `get_pose_sequence(gloss_word) -> PoseSequence | None`.
- **`build_library.py`** — the end-to-end pipeline (download → extract →
  manifest) actually run to produce the real library on disk.
- **`PLAN.md`** — the full design doc: WLASL access/licensing research (read
  directly from the real `WLASL_v0.3.json` and `C-UDA-1.0.pdf`, not assumed),
  the MediaPipe pipeline design, storage format, lookup interface, and the
  fingerspelling-fallback design (not yet implemented — see below).

**Real library on disk** (gitignored, `pose_library/data/`, produced by
`build_library.py`; verified directly against the real manifest this
session, not assumed):

- **118 words.**
- **14/118 (~12%) flagged `low_confidence: true`** by `compute_quality_flags()`
  — not a rare edge case; budgeted for accordingly in Stage 5's UI
  (Section 11).

**Tests**: 61 tests across `tests/test_pose_download.py`,
`test_pose_extract.py`, `test_pose_lookup.py`, `test_pose_manifest.py`,
`test_pose_types.py`, `test_wlasl_metadata.py`, `test_build_library.py`, and
`test_build_target_vocab.py` — all passing; `ruff`/`black` clean on
`pose_library/` (all verified this session, not assumed from an earlier run).

**Not built**:

- **Fingerspelling fallback** — *superseded 2026-09-26, see Section 13.*
  (Original note, kept for the record: `pose_library/PLAN.md` Section 5's
  `get_fingerspelling_sequence()`/`resolve_gloss_word()` sketch was never
  built; it was redesigned rather than implemented as sketched, and the
  resolution now lives in the frontend — see
  `pose_library/FINGERSPELLING_PLAN.md` Section 4.)

**Resolved 2026-09-26 — kept local/dev-only indefinitely (see Section 5
item 6 for the decision note).** Original flag, kept for the record:
`pose_library/PLAN.md` Section 1 flags a genuine, still-unresolved
licensing ambiguity around WLASL-derived pose data under the Computational
Use of Data Agreement (C-UDA): whether an extracted keypoint sequence counts
as a "Result" (Section 2.2/5.5, unrestricted) or is still "Data" in modified
form (Section 5.2, restricted) turns on the license's own de-minimis test,
which the license text doesn't settle either way — and WLASL's own README
separately, plainly states "no commercial usage is allowed." Current policy
(per PLAN.md's recommendation, never formally revisited by the user): treat
all WLASL-derived pose data as local-only, gitignored, never redistributed,
never sold. That policy is now the user's explicit, standing decision
(2026-09-26); revisit the ambiguity above before any public release or
commercial pivot.

---

## 11. Stage 5: rendering — signed off (2026-09-22)

Initial implementation landed 2026-09-11 (`3744355`, single-word skeleton
playback; `2c9b6f5`, design tokens/app scaffolding). Refined and closed out
across two later sessions:

- **`b5138da`** (2026-09-21) — soft-follow content-fit camera
  (`computeContentBounds`/`computeFitTransform`/`projectPoint` in
  `skeletonBones.js`): hand landmarks only cover ~12-17% of the raw
  normalized frame, so the original direct `x * canvasSize` mapping left
  signs too small to read; this re-fits per frame to real hand content,
  eased via exponential smoothing, with a `prefers-reduced-motion`
  instant-snap fallback. Documented in `frontend/PLAN.md`'s "Camera fit"
  subsection (`fa93b3e`).
- **`a024993`** (2026-09-22) — added a DOM test environment (happy-dom +
  `@testing-library/react`; `vite.config.js`'s `test.environment` had been
  `"node"`, so no component could be rendered at all) and the integration
  test `frontend/PLAN.md` Section 6 explicitly requires: real words queried
  live from the actual 118-word manifest (`about`, clean; `phone`,
  `low_confidence` at 11/68 frames kept; plus a word confirmed absent from
  the manifest for the OOV case), asserting on real rendered DOM output
  (text, ARIA roles, canvas presence), not pixels or snapshots.
- **`28a516d`** (2026-09-22) — doc hygiene: `frontend/PLAN.md`'s
  low-confidence section said `quality_notes` should be available "on
  hover/expand"; the shipped `CaptionBand` banner shows it always-visible
  instead, so the doc was corrected to match the real behavior.

**What was built** (`frontend/`, file by file): `App.jsx` (owns word-lookup
state, wires `usePoseSequence` → `reconstructTimeline` → `SkeletonCanvas`),
`hooks/usePoseSequence.js` (fetches the manifest + pose JSON, returns the
`ok`/`low_confidence`/`not_found`/`error` discriminated union
`frontend/PLAN.md` Section 5 specifies), `hooks/usePrefersReducedMotion.js`,
`components/StatusStrip.jsx`/`CaptionBand.jsx`/`SkeletonCanvas.jsx`,
`lib/skeletonBones.js` (bone topology + camera fit), `lib/reconstructTimeline.js`
(within-word gap reconstruction: short gaps linearly interpolated, long gaps
held and dimmed to 50%, per `PLAN.md` Section 4), plus a Vite dev-server
middleware (`vite.config.js`) serving `pose_library/data/poses/*.json` over
HTTP as `PLAN.md`'s documented stopgap.

**Stage 5 completion criteria — all met, signed off 2026-09-22:**

- Feature-complete against `frontend/PLAN.md`'s full v1 scope: single-word
  lookup and playback, play/pause/loop controls with frame-accurate timing
  from each word's real `fps`, within-word interpolation/gap-smoothing,
  low-confidence and OOV/not-found UI states, loading/latency state, and the
  full accessibility pass (contrast, type size, color-independent status,
  keyboard/focus, ARIA live regions, `prefers-reduced-motion` handling) —
  verified line-by-line against the actual code, not assumed, in a dedicated
  gap-analysis pass this session; the one gap it found (no integration test)
  is exactly what `a024993` closes.
- **25/25 frontend tests passing** (`skeletonBones.test.js`,
  `reconstructTimeline.test.js`, and the new `App.test.jsx` integration
  test) — `a024993`.
- `frontend/PLAN.md` itself kept in sync with the shipped implementation
  (`fa93b3e`, `28a516d`) rather than left stale.

**Known gaps, explicitly not blocking sign-off** (out of Stage 5's v1 scope
per `PLAN.md` Section 6, or deferred to a later stage):

- Fingerspelling rendering — no data exists yet (Section 10 above).
- Multi-word sentence playback / cross-word transition blending — this is
  CLAUDE.md's build-order **step 3** ("interpolation/smoothing between
  signs"), a distinct and still-unstarted step after the Stage 4/5 work
  ("fed gloss manually", step 2) this section covers. **This, not Stage 1/2
  live speech, is the next build-order step** — see Section 6.
- 3D rigged avatar / Three.js — explicit v2 stretch goal; `three` stays
  unused by v1 code.
- No live latency measurement against a real pipeline (none exists yet) —
  the status strip's latency readout is honestly scoped to this stage's own
  fetch-to-first-frame time only, never a simulated pipeline number.

**Sign-off**: Stage 5 is done. Per CLAUDE.md's build-order rule, the next
step was multi-word/cross-sign interpolation (build-order step 3) — now
done, see Section 12.

**Amendment 2026-09-25**: a layout bug (`SkeletonCanvas.module.css`)
present since this stage's first commit (`3744355`) caused the rendered
canvas to overflow its container and appear over-zoomed on real viewports;
never caught by the automated test suite since happy-dom has no layout
engine. Found and fixed during build-order step 3's manual browser testing.
See Section 12.

---

## 12. Build-order step 3: multi-word playback with cross-sign interpolation — done (2026-09-25)

Per `CLAUDE.md`'s build order, step 3 ("interpolation/smoothing between
signs") is now complete: a short, already-glossed sequence of words (typed
space-separated into the same search input single-word mode used) plays
back as one continuous skeleton animation with smoothed transitions between
signs, instead of one word at a time. Designed in
`frontend/MULTIWORD_PLAN.md` (`f4fdc82`, two clarifications added and
reviewed before implementation) and built in `7125800`. This is a pure
playback/animation-stitching feature — no ASL grammar/reordering, no live
ASR/VAD (still build-order step 4, next, see Section 6).

**What was built** (`frontend/src/`), file by file:

- **`hooks/usePoseSequences.js`** (new) — batch version of
  `usePoseSequence`: fetches `/poses/manifest.json` **once** per submitted
  sequence (not once per word) plus every word's own `/poses/<word>.json`,
  all dispatched simultaneously; returns one discriminated-union result per
  word plus a sequence-level `idle`/`loading`/`ready`/`error` status. A
  manifest-fetch failure surfaces as a sequence-level `error`, distinct
  from any one word's own `not_found`/`error`.
- **`hooks/usePoseSequence.js`** — refactored, not behaviorally changed:
  the shared fetch-and-classify logic was extracted into an exported
  `fetchWordPoseResult(word, manifest, sequenceFetch)`, used by both this
  hook and `usePoseSequences`, while this hook's own manifest + word-JSON
  requests are still dispatched simultaneously exactly as before the
  extraction (no added latency, no behavior change).
- **`lib/stitchTimelines.js`** (new) — resamples each word's own
  `reconstructTimeline()` output (unchanged) onto a shared fixed 30fps
  grid, then inserts 200ms of `lerpPose`-interpolated transition frames at
  each word boundary, reusing the exact same zero-point-aware lerp
  `reconstructTimeline.js` already had (now exported for this reuse).
  Output is the identical `Timeline` shape the single-word renderer already
  consumed, so `SkeletonCanvas.jsx` needed no changes to its core playback
  loop. Transition frames are attributed to the word they lead *into*, so
  the returned `wordBoundaries` spans are contiguous and gapless across the
  whole merged timeline.
- **`components/SkeletonCanvas.jsx`** — the one change: an optional
  `onFrameChange(frameIndex)` callback, fired every tick, so `App.jsx` can
  map the currently-playing frame to a word via `wordBoundaries` without
  `SkeletonCanvas` itself needing any concept of "words."
- **`App.jsx`** — generalized to always treat its input as a sequence (a
  single typed word is simply a sequence of length 1); filters `not_found`
  words out before calling `stitchTimelines` (so a `not_found` word never
  needs its own transition handling — the words on either side just become
  adjacent and get one ordinary transition); a per-word or manifest-level
  `error` still aborts the whole sequence, while `not_found` words are
  skipped and listed visibly, not aborting.
- **`components/CaptionBand.jsx`** — multi-word search input
  (space-separated, capped at 20 words per `CLAUDE.md`'s input
  length-limiting rule, truncation surfaced visibly not silently), a
  word-progress row (active/low-confidence/skipped styling per word), and
  skipped/low-confidence/truncated banners generalized from the original
  single-word banners.
- **`components/StatusStrip.jsx`** — added a composite
  "READY — N/M WORDS (K SKIPPED)" label, shown whenever there's more than
  one word or a skip; a single clean or low-confidence word keeps its
  original bare label unchanged.

**Tests**: 40/40 passing across 5 files — the original 11
`reconstructTimeline.test.js` and 12 `skeletonBones.test.js` tests
unchanged; the 3 original `App.test.jsx` single-word tests still pass under
the generalized implementation (real regression coverage, not rewritten);
2 new `App.test.jsx` tests (a real mid-sequence `not_found` word against
the actual manifest, and an all-missing fallback); 7 new
`stitchTimelines.test.js` tests; 6 new `usePoseSequences.test.js` tests.
`eslint .` clean (0 errors/warnings — one real `react-hooks/exhaustive-deps`
warning was found and fixed, a misplaced `eslint-disable-next-line`
comment), `prettier --check` clean on all new/changed files.

**A real bug found and fixed during implementation, not anticipated in the
plan**: `StatusStrip`'s word-count/skip-count label was initially gated on
`state === "ready"` only, silently dropping that info whenever the sequence
included a `low_confidence` word (since that state took precedence over
`ready`). Fixed by decoupling the composite label text from the dot's
semantic color/state.

**Known limitation, documented not fixed — the "phantom static hand"
visual characteristic**: confirmed by reading `lerpPose` and
`SkeletonCanvas.jsx`'s draw loop directly (see `MULTIWORD_PLAN.md`'s "Known
gotchas"), not assumed. A hand that's the `(0,0,0)` "not detected" sentinel
for an entire word (a one-handed sign's unused hand) is **not**
skip-drawn by the renderer — only the 6 pose-subset landmarks get that
treatment, never hand landmarks. So that hand's joints/bones render as a
static, motionless cluster at the projected origin coordinate for as long
as it stays zero, then pop in abruptly (no fade) the instant playback
reaches a frame where that hand has real tracked data. **This is not a new
bug introduced by multi-word stitching** — the identical phantom-point
rendering already happens throughout a one-handed word's own single-word
playback today (Stage 5, already shipped, `3744355`/`b5138da`) — stitching
only means the same characteristic is also present, unchanged, during the
inserted transition frames. Not fixed in this pass; a real fix
(skip-drawing a zero-sentinel hand entirely, or fading it in/out) is future
work if it turns out to matter visually in practice, not blocking anything.

**Two camera/rendering bugs found in manual browser testing after
`7125800`, fixed together** (45/45 tests passing after the fix):

1. **A pre-existing Stage 5 layout bug, not a multi-word regression.**
   Every playback — single-word "bathroom" alone included — rendered as a
   few oversized bone segments filling the stage. Confirmed pre-existing by
   reproducing it on the clean committed code (working-tree changes
   stashed). Per-frame logging showed the camera math was correct (hand
   bounds ~0.07×0.18, camera span settling ~0.24, i.e. a whole hand at ~75%
   of canvas height); the canvas itself was the problem — 1167px square
   inside a 498px-tall stage at a 1280×720 viewport. `.stage`'s row flexbox
   (`align-items: center`) never gave `.stageWrapper` a definite height, so
   the bezel's `height: 100%`/`max-height: 100%` resolved to `auto` and
   `aspect-ratio: 1 / 1` sized the square off the stage's *width*,
   overflowing vertically on any landscape viewport and cropping the canvas
   to its middle slice. Fixed in `SkeletonCanvas.module.css`
   (`align-self: stretch` + `container-type: size` on `.stageWrapper`, a
   `min(100cqw, 100cqh)` square on `.bezel`); verified at 1280×720 and
   375×812. See the Section 11 amendment and `CLAUDE.md`'s gotchas log for
   the process lesson (happy-dom can't catch layout bugs).
2. **`stepCamera()` in `skeletonBones.js`**: a cross-word transition
   between two signs that use *different* hands (e.g. `about`, left-only →
   `bathroom`, right-only) zeroes *both* hands for the whole transition via
   `lerpPose`'s zero-endpoint guard, so `computeContentBounds` fell back to
   the full `[0,1]` frame and the camera zoomed out and back at every such
   boundary. `stepCamera` now holds the camera steady on a frame with no
   real hand points — confirmed in-browser on "about bathroom doctor
   angry" (transition frames 68–74 and 167–173 hold exactly). Its effect
   was invisible until fix 1, which dwarfed it.

Also observed, not a bug: "angry" zooms out to a near-full-frame shot
(camera span ~1.0) on frames where both hands are genuinely tracked far
apart (x ≈ 0.16–0.89) — the fit correctly includes both hands.

**Explicitly out of scope for this pass**, per `MULTIWORD_PLAN.md` Section
6: ASL grammar/gloss reordering (a separate NLP problem — words are
stitched in the order given, verbatim), live ASR/VAD wiring (build-order
step 4, next), fingerspelling for skipped words (still not built, see
Section 5 item 7), variable/linguistically-informed transition duration
(one fixed 200ms for every boundary), non-manual/facial coarticulation (no
such data exists), scrubbing/seeking, mid-playback sequence editing, and
cross-submission caching.

**Sign-off**: build-order step 3 is done. Per `CLAUDE.md`'s build-order
rule, the next step is **step 4: live speech input** (Stage 1/2, ASR +
VAD), wired to this now-working core — not yet started, see Section 6.

---

## 13. Fingerspelling alphabet — infrastructure complete, letters NOT recorded (2026-09-26)

> **Superseded in part by Section 14 (same day)**: 24 of the 26 letters now
> come from the MIT-licensed `sid220/asl-now-fingerspelling` dataset, so the
> "none of the 26 letters recorded" status and the "record all 26" steps
> below no longer apply. Only J and Z still need recording. Kept below as
> written, as the record of the infrastructure pass.

**Honest status first: infrastructure-complete, data-incomplete.** Everything
needed to spell an out-of-library word letter by letter is built, tested,
and verified in a real browser, but **none of the 26 real letters has been
recorded**. That takes a person signing on camera, which can't be automated.
Every test and the manual browser check ran on **synthetic placeholder
letters** (geometric stand-in hands, not ASL), which prove the plumbing and
nothing about handshape quality. Until the letters are recorded, the app
behaves as before for OOV words, with honest copy: "FINGERSPELLING ALPHABET
NOT RECORDED YET".

Design: `pose_library/FINGERSPELLING_PLAN.md` (`8942f3d`). Built in
`ffe1ed0` (recorder), `f6ed0b5` (extraction/manifest build), `767a794`
(frontend integration, UI, synthetic fixtures).

**What was built**, file by file:

- **`pose_library/record_fingerspelling.py`** — webcam recorder
  (`python -m pose_library.record_fingerspelling`). One OpenCV window,
  keyboard-driven: SPACE records (3s countdown, then 2.0s for static letters
  and 2.5s for J/Z), R re-records, N/P next/previous, Q quits; `--letters jz`
  targets specific letters, and a session resumes at the first unrecorded
  letter. Takes are 640×480 (matching the WLASL clips the pipeline was
  validated on), written atomically at the **measured** fps (webcams
  misreport it, and `extract.py` trusts the file header), saved unmirrored,
  then immediately checked with the real Holistic extraction ("hand detected
  in N/M frames").
- **`pose_library/fingerspelling.py`** — pure logic: letter rules, file
  layout, and segment selection. A static letter keeps its stillest
  gap-free 0.4s window (`LETTER_HOLD_SECONDS`); J/Z keep the span where the
  hand actually moves; the non-signing hand is zero-filled so a resting
  hand can't pull the camera out. Motion thresholds come from real WLASL
  per-frame displacement, not guesses.
- **`pose_library/build_fingerspelling.py`** — `python -m
  pose_library.build_fingerspelling`: runs every recorded letter through the
  **unchanged** `extract.py`, writes `pose_library/fingerspelling/poses/
  <letter>.json` and `manifest.json`. Expected failures are reported rather
  than raised, and a failed rebuild removes the stale JSON and entry.
- **`pose_library/manifest.py`** — `build_letter_entry()`: WLASL's entry
  shape wherever the fields apply (the timeline fields keep their exact
  meaning, so `reconstructTimeline` consumes letters unchanged); WLASL ids
  dropped; letter-specific quality flags.
- **`frontend/src/lib/fingerspelling.js`** — `spellWord`, `lettersNeeded`,
  `planPlayback`: `not_found` word → letters → playback units shaped exactly
  like word results. `App.jsx` calls the **same** `usePoseSequences` hook a
  second time (new optional `basePath: "/fingerspelling"`) and feeds every
  unit to the **same** `stitchTimelines` call. There is no second rendering path.
- **UI** (`CaptionBand.jsx`/`.module.css`, `StatusStrip.jsx`) — a spelled
  word renders in gloss notation (`J-A-D-E`, amber, dotted underline); the
  playing word highlights the letter being signed; the accessible name is
  "FINGERSPELLED: JADE"; the strip reads "READY — 3/3 WORDS (1 FINGERSPELLED)".
- **`frontend/vite.config.js`** — serves `/fingerspelling/*` from
  `pose_library/fingerspelling/poses/`; `FINGERSPELLING_POSES_DIR` overrides
  it (used only for the synthetic browser check).
- **`tests/fixtures/fingerspelling_synthetic/`** + generator
  `tests/fixtures/make_synthetic_fingerspelling.py` — six SYNTHETIC
  placeholder letters (a–e, j) in the real file shapes, each labeled
  `source: "synthetic:placeholder"`. Never inside `pose_library/fingerspelling/`.
- **`.gitignore`** — negates the global `*.mp4` rule for
  `pose_library/fingerspelling/raw/*.mp4`: the recordings are project-owned
  and **are committed**, unlike WLASL data.

**Tests**: **138/138 Python** (88 before, +50: `test_fingerspelling.py`,
`test_record_fingerspelling.py`, `test_build_fingerspelling.py`, letter-entry
tests in `test_pose_manifest.py`) and **75/75 frontend** (45 before, +30:
`fingerspelling.test.js`, `usePoseSequences` `basePath` tests, and 5 new
`App.test.jsx` integration tests that spell words end to end on synthetic
letters, including one asserting the highlighted letter advances C → A → B
during real rAF playback). `ruff`, `black`, `eslint`, `prettier` clean.

**Verified beyond unit tests** (per `CLAUDE.md`'s happy-dom lesson):
- **Real MediaPipe, no mocks**: `build_alphabet` run on two WLASL clips
  standing in for recordings (scratch directory only, nothing committed)
  produced a 12-frame static hold and a 36-frame motion span with honest
  interior drops. Both were correctly flagged low-confidence, since WLASL
  signers enter and leave the frame.
- **Real browser, synthetic letters**: "about jade phone" played ABOUT → J →
  A → D → E → PHONE; at 375×812 there was no overflow, and a word with
  unrecorded letters ("quiz") was skipped with the missing letters named.
- **Real browser, real (empty) alphabet**: "about cab" plays ABOUT and skips
  CAB as "FINGERSPELLING ALPHABET NOT RECORDED YET". That's the state today.

**Bugs found and fixed during the build, not anticipated in the plan**:
- Running the build with no recordings wrote an empty `{}` manifest, which
  the frontend would have read as "alphabet exists, every letter missing"
  instead of "not recorded yet". The build now never writes an empty
  manifest and removes one that becomes empty. Regression-tested.
- `usePoseSequences.js` contained a raw NUL byte (its `words.join` separator)
  since `7125800`, so git treated the file as binary and none of its diffs
  were reviewable. It's now the `"\u0000"` escape: same value, text file.

**Sign-off criteria**:
- **Infrastructure — met 2026-09-26**: everything above.
- **Feature — NOT met, blocked on recording**: all 26 letters recorded and
  built, `manifest.json` has 26 entries with nothing unexpectedly
  `low_confidence`, and several real spelled words (at least one with J or
  Z) confirmed readable **by someone who knows ASL fingerspelling**.

**Known limitations** (see the plan's Section 6 for the full out-of-scope
list): digits and punctuation aren't spelled; letters use the generic 200ms
word transition; 0.4s per letter is a readability guess, not user-tested;
`DESC-`/`X-` gloss prefixes aren't stripped before spelling; the
phantom-hand characteristic (Section 12) applies to the zero-filled
non-signing hand; one take per letter, one signer.

**The exact next manual step** (the only thing left for fingerspelling):
1. From the repo root with the venv active:
   `python -m pose_library.record_fingerspelling`. Sign each letter when
   prompted and re-record any take the window reports as LOW.
2. `python -m pose_library.build_fingerspelling`. It should report 26/26
   letters built.
3. Restart the dev server (`npm run dev` in `frontend/`), spell a few
   words (e.g. `jazz`, `quick`), and check that they read correctly.
4. Commit `pose_library/fingerspelling/` (raw videos, poses, manifest).

---

## 14. Fingerspelling data: 24 letters from the asl-now dataset; J and Z still missing (2026-09-26)

**Status, exactly: 24 of 26 letters have real, usable data. 2 (J, Z) still
need recording.** The 24 are converted from
[`sid220/asl-now-fingerspelling`](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
(Hugging Face, MIT, Sidney Trzepacz, revision `9b3c96ae`). They're **not**
self-recorded and **not** synthetic placeholders. License record:
`pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`.
Every decision, with the evidence behind it, is in
`pose_library/FINGERSPELLING_PLAN.md` Section 2b.

**Dataset inspection (all 2,122 files read, not sampled):**
- 26 letter folders, 53–155 samples each. **Every file is a single frame**
  of 21 `{x, y, z}` points, exactly as the card describes.
- **J and Z: no motion data.** They have 93 and 155 files, all single still
  frames like every other letter. They are **not converted**; no motion was
  faked.
- No handedness label, and mixed hands (2D palm test ~60/40 per letter; a
  3D chirality measure is inconsistent across letters, so not trustworthy).
  Recorded as `signing_hand: "unlabeled"`.
- 41 of 1,874 static-letter samples have a landmark outside the image and are
  excluded; every letter keeps at least 48 usable samples.
- License: MIT per the card's metadata and prose. The HF repo has no LICENSE
  file, so the copyright + permission notice is taken verbatim from the
  author's own project repo (github.com/Sid220/asl-now, which links the
  dataset). Both sources are quoted separately in the license record.

**What was built**, file by file:
- **`pose_library/convert_hf_fingerspelling.py`**
  (`python -m pose_library.convert_hf_fingerspelling`): downloads the pinned
  revision into the gitignored `pose_library/data/asl_now_fingerspelling/`,
  validates and filters samples, picks each letter's **medoid** (lowest
  median wrist-anchored, scale-normalized shape distance; a centroid would
  blend mirror images), and writes the same pose JSON + manifest format a
  recorded letter gets: 12-frame (0.4s) hold, landmark values unaltered.
- **`pose_library/manifest.py`**: `build_dataset_letter_entry` (`source`,
  `source_url`, `license`, `dataset_sample`), sharing one `_letter_entry`
  core with `build_letter_entry`.
- **`pose_library/build_fingerspelling.py`**: a real bug fixed. A failed or
  absent recording used to delete the letter's entry whatever its source, so
  `--letters a` with no recording of A would have deleted the dataset's A. It
  now only removes entries it owns. A recording still replaces a dataset
  letter on success, and the converter never overwrites a recorded one.
- **`frontend/`**: the missing-alphabet copy is now "FINGERSPELLING ALPHABET
  NOT AVAILABLE" ("not recorded yet" stopped being true). No other frontend
  change: dataset letters flow through the unchanged pipeline.
- **Committed data**: `pose_library/fingerspelling/poses/` (24 letters +
  manifest, ~600KB); 11 real samples as test fixtures
  (`tests/fixtures/asl_now_sample/`); `requirements.txt` pins
  `huggingface_hub==1.30.0` (already transitive; `pip-audit` clean).

**Tests**: **159/159 Python** (138 + 21 new in
`test_convert_hf_fingerspelling.py`, on real dataset samples: format
validation, the in-frame filter, normalization, medoid selection (matches an
independent computation, file-order independent, never picks an outlier),
the J/Z decision, manifest provenance, unaltered values, merge precedence,
and the ownership fix, which fails against the pre-fix code). **77/77
frontend** (75 + 2 App tests on the real committed letters). `ruff`,
`black`, `eslint`, `prettier`, and `gitleaks` all clean. The synthetic
plumbing tests are unchanged.

**Verified in a real browser, with the real converted letters:**
- Held single letters on screen: **B, L, V, Y are clearly legible and
  correct** handshapes. **A and O are correct but harder to read.**
  Curled-finger letters are inherently ambiguous in a 2D stick skeleton with
  no depth; that's a rendering limit, not a data problem, since the landmark
  values are the dataset's own.
- "about black help": the stitched timeline is ABOUT (frames 0–68) → B, L,
  A, C, K (18 frames each: 6 transition + 12 hold) → HELP (159–221), 7.37s,
  with every letter carrying its `hf:sid220/asl-now-fingerspelling:…`
  source. Live playback showed the words and letters in that order, but the
  browser pane throttled animation frames during the session, so some letters
  were skipped over on screen. That was environmental (0–1 frames/s), not an
  app issue.
- "about jazz": ABOUT plays; JAZZ is skipped with `NO FINGERSPELLING FOR
  "J", "Z"`.

**Known limitations**: J and Z missing; one sample per letter from a mix of
participants, so letters may differ slightly in hand size and apparent hand
across a word; handedness unlabeled; held poses have no micro-motion; not
yet checked by someone who knows ASL fingerspelling.

**The exact next manual step** (J and Z only):
1. From the repo root with the venv active:
   `python -m pose_library.record_fingerspelling --letters jz`. Draw each
   letter's motion once when recording starts, and press R to re-take if the
   window reports LOW.
2. `python -m pose_library.build_fingerspelling`. It should report J and Z
   OK and "Alphabet: 26/26 letters built." The 24 dataset letters are left
   alone.
3. Restart `npm run dev` in `frontend/` and spell `jazz`.
4. Commit `pose_library/fingerspelling/` (two raw clips, two poses, manifest).
