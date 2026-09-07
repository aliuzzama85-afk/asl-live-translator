# Stage 3 Plan: T5-small Fine-Tuning for English → ASL Gloss

Status: **implemented and verified end-to-end on a small CPU sanity run.**
`config.py`, `data_prep.py`, `evaluate.py`, `train.py`, and `inference.py` exist under
`gloss_model/`, with tests in `tests/`. A full-scale training run on real GPU compute
(Kaggle) is still pending — see "Implementation status" near the end of this file.
Scope: standalone text-to-text translation (typed English in, gloss text out). No ASR,
VAD, pose lookup, or rendering involved. Per the project build order, this must be
working and tested before Stage 4 work begins.

---

## 0. Dataset identity (confirmed via a real `load_dataset()` call)

- **Hub ID**: `achrafothman/aslg_pc12` — confirmed correct. `datasets.load_dataset("achrafothman/aslg_pc12")`
  resolves and loads without error.
- **Fields**: two string columns, `gloss` and `text` (English). Confirmed via
  `dataset.features`.
- **Size**: **87,710 rows** — notably larger than the ~12,000 estimated from secondary
  sources during planning (that number was wrong; this is the real, directly-queried count).
- **Splits**: ships a single `train` split only, as expected — no pre-defined
  validation/test split. Confirmed via `dataset.keys()`.

---

## 1. Dataset loading & inspection

### Loading
```python
from datasets import load_dataset

raw = load_dataset("achrafothman/aslg_pc12")
```
This belongs in `gloss_model/data_prep.py`, behind a small `load_raw_dataset()`
function — not inline in the training script — so inspection and training share one
source of truth for how the data enters the pipeline.

### Inspection results (confirmed by directly querying the loaded dataset)
- **Row count**: 87,710 total rows, single `train` split (see Section 0).
- **Column names**: `gloss`, `text` — both `Value('string')`.
- **Empty rows**: none. `empty text: 0, empty gloss: 0`.
- **Encoding artifacts**: a BOM character (`﻿`) appears in exactly 2 rows —
  negligible in volume, but `clean_text()` strips it (and surrounding whitespace/
  trailing `\n`, which appears on every row) regardless.
- **Gloss casing convention**: overwhelmingly consistent uppercase. In a 20,000-row
  sample, only 152 of 198,037 alpha gloss tokens (~0.08%) were not fully uppercase —
  not worth special-casing.
- **English/gloss alignment / rule-based artifacts**: **1,466 rows (1.7%)** have
  `gloss == uppercase(text)` exactly — i.e. the "gloss" is just the English sentence
  case-flipped, with no real reordering or simplification. This confirms the rule-based
  generation produces a meaningful fraction of trivial (non-)translations. See "Known
  limitations" below.
- **Duplicates**: **6,585 exact-duplicate `(text, gloss)` rows (7.5%)** — deduped before
  splitting in `data_prep.build_splits` (87,710 → 81,091 unique rows... 81,125 was the
  planning-time estimate from a slightly different dedup pass; 81,091 is what the
  shipped `build_splits` produces, a difference of 34 rows likely due to the exact
  whitespace-cleaning order — not investigated further since both numbers agree it's
  ~7.5% duplication). Only 3 distinct `text` values map to more than one distinct
  `gloss` — negligible one-to-many ambiguity.
- **Length distribution** (whitespace token count): text p50/p90/p99 = 14/19/21,
  max 59. Gloss p50/p90/p99 = 12/17/20, max 54. Drove the `MAX_SOURCE_LENGTH` /
  `MAX_TARGET_LENGTH = 32` choice in Section 2.
- **Fingerspelling / classifier / non-manual markers**: **zero** rows matched a
  fingerspelling-like pattern (`X-X-X`) or a `CL:` classifier prefix, across the full
  87,710-row corpus. Confirms the literature's characterization of ASLG-PC12 as "flat"
  gloss with essentially no such annotation — Stage 4's fingerspelling fallback will be
  doing real work for proper nouns/numbers/OOV words, not a rare edge case.

---

## 2. Preprocessing

### Source/target format for T5
T5 is a text-to-text model with no dedicated encoder/decoder vocab split — task
framing comes entirely from a text prefix on the input. Use:

```
input:  "translate English to ASL gloss: {english_sentence}"
target: "{gloss_sequence}"
```

Using an explicit, human-readable task prefix (rather than a bare sentence) follows
the T5 paper's convention and the HF fine-tuning example scripts referenced in
CLAUDE.md, and leaves room to later add other task prefixes to the same model/dataset
(e.g. a debug "translate ASL gloss to English" direction) without a schema change.

### Tokenization
- Use the T5-small tokenizer (`T5TokenizerFast.from_pretrained("t5-small")`) for both
  source and target — no custom vocabulary. If gloss tokens are single all-caps words
  this should tokenize fine with the default SentencePiece vocab; if inspection turns
  up unusual tokens (hyphenated fingerspelling markers, `CL:` prefixes) we may need to
  check how the tokenizer splits those specifically (SentencePiece will subword-split
  anything not in vocab, which is *fine* for T5 but worth eyeballing so we're not
  silently exploding a single gloss token into 5 subword pieces).
- Tokenize target text using the tokenizer's target-mode context (`tokenizer(text_target=...)`
  in current `transformers` versions) rather than the older `as_target_tokenizer()`
  context manager, since CLAUDE.md's reference-conventions instruction points at
  current HF example scripts.

### Max sequence length
- Pick a max length empirically from the length-distribution check in Section 1
  (e.g. the 99th-percentile token count, rounded up to a multiple of 8), rather than
  defaulting to 128 or 512 without justification. ASLG-PC12 sentences are corpus
  sentences (news/formal text), so I'd expect source lengths mostly under ~40-60
  tokens — but this is a guess to be replaced by the actual histogram.
- Use the same max length for source and target as a starting point (gloss is
  typically similar length to or shorter than English, since function words often
  drop out); revisit only if the histogram shows a real asymmetry.
- Truncate rather than filter out longer examples initially, but log how many rows
  get truncated — if it's a meaningful fraction, that's a sign the chosen max length
  is too aggressive.

### Train/val/test split
- If the HF dataset ships only a single `train` split (per Section 0's caveat), split
  it ourselves with a fixed seed: **90/5/5** train/val/test. Rationale: ~12k rows means
  5% (~600 rows) is enough for a stable validation signal and a meaningful held-out
  test set, without starving the training set on a corpus this small.
- Split at the row level with `datasets.Dataset.train_test_split(test_size=..., seed=SEED)`
  called twice (train vs. rest, then rest split again) rather than a manual index
  slice, so it's reproducible and uses the library's own randomization.
- **Dedupe before splitting**, not after — if exact-duplicate pairs exist (per Section
  1), the same pair could otherwise end up in both train and test, silently inflating
  eval scores.

### Gloss-specific token handling
- Do **not** invent custom special tokens speculatively. Only add tokens to the
  tokenizer (`tokenizer.add_tokens([...])`) if inspection in Section 1 actually finds
  gloss-specific markers (fingerspelling, classifiers, non-manual markers) that get
  destructively subword-split in a way that looks harmful (e.g. a marker that appears
  in hundreds of rows getting split into 4+ pieces). Adding tokens means resizing the
  model's embedding matrix (`model.resize_token_embeddings(len(tokenizer))`), which is
  cheap but should be a deliberate, logged decision tied to a specific inspection
  finding — not done preemptively.

### Reproducibility
- One `SEED` constant (e.g. `42`) defined once in `gloss_model/config.py` (see
  Section 3) and threaded through: dataset splitting, model init (`set_seed(SEED)`
  from `transformers`), and the data loader's shuffling.
- Log the exact dataset hub ID + revision/commit hash actually loaded (HF datasets
  exposes this) alongside the seed, so a training run can be reconstructed later even
  if the hub dataset changes upstream.

---

## 3. Fine-tuning script structure

### Module layout under `gloss_model/`
```
gloss_model/
  __init__.py            # already exists (empty)
  config.py              # dataclass/constants: seed, model name, max lengths,
                          # hyperparameters, paths — single source of truth
  data_prep.py           # load_raw_dataset(), inspect_dataset(), preprocess(),
                          # build_splits() — all pure functions, no training logic
  train.py               # training entry point, if __name__ == "__main__":
  evaluate.py            # metric computation + qualitative spot-check harness,
                          # importable by both train.py (for eval-during-training)
                          # and tests/
  inference.py           # thin wrapper: load a fine-tuned checkpoint, expose
                          # translate(english_text: str) -> str for later pipeline
                          # integration (not wired into pipeline/ yet — that's a
                          # later stage's job per CLAUDE.md's build order)
  checkpoints/            # gitignored — trained model artifacts land here
```
This mirrors the standard HF `transformers` example-script shape (separate data
prep, a `Trainer`-based train script, and an eval module) while keeping each file
small enough to unit-test in isolation, per CLAUDE.md's testing requirement.

### Training approach
Use `transformers.Seq2SeqTrainer` + `Seq2SeqTrainingArguments` rather than a hand-
rolled training loop — this is the canonical HF T5 fine-tuning structure CLAUDE.md
asks us to follow, and it gives us checkpointing, eval-during-training, and logging
for free instead of reimplementing them.

`train.py` shape (structure, not full implementation):
```python
def main() -> None:
    args = parse_args()  # or load from config.py; keep CLI overrides minimal
    set_seed(config.SEED)
    raw = data_prep.load_raw_dataset()
    tokenized = data_prep.preprocess(raw, tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=...,
        compute_metrics=evaluate.compute_metrics,
    )
    trainer.train()
    trainer.save_model(config.CHECKPOINT_DIR)

if __name__ == "__main__":
    main()
```

### Starting hyperparameters and rationale
| Hyperparameter | Starting value | Why |
|---|---|---|
| Base model | `t5-small` (60M params) | Fixed by project scope; small enough to fine-tune on CPU in a pinch, though see risk note below. |
| Learning rate | `3e-4` (with AdamW) | Standard starting point for fine-tuning T5-small on a downstream seq2seq task in HF examples; T5's own pretraining used Adafactor with a different schedule, but AdamW + a few e-4 LR is the well-trodden fine-tuning path and easier to reason about for a first pass. |
| Batch size | 16 (effective, via gradient accumulation if GPU memory is tight) | Corpus is small (~12k rows); a batch size in the 8-32 range gives enough gradient signal per step without needing a huge memory budget. |
| Epochs | 3-5, chosen by early stopping on validation loss/metric rather than a fixed number | With ~10-11k training rows this is enough exposure to converge without needing dozens of epochs; going higher risks overfitting a corpus this size and this repetitive (rule-based generation tends to have repeated sentence templates). |
| Optimizer | AdamW (Trainer default) | Simplicity and wide precedent for fine-tuning; no reason to reach for Adafactor at this model size. |
| Max source/target length | TBD from Section 1's histogram | See above — not guessed in advance. |
| Weight decay | 0.01 (Trainer default) | Standard, no strong reason to deviate for a first pass. |
| LR schedule | Linear warmup + decay (Trainer default) | Standard; revisit only if training curves look unstable. |

These are explicitly "reasonable first-pass defaults, not tuned" — the plan is to get
an end-to-end run working and evaluated, then treat hyperparameter adjustment as a
second pass driven by what the eval numbers and qualitative spot-checks actually show.

### Coding conventions applied
- Every function in `data_prep.py`, `train.py`, `evaluate.py`, `inference.py` gets
  type hints on its signature and a Google-style docstring, per CLAUDE.md.
- `config.py` as a `@dataclass(frozen=True)` (or module-level constants with type
  annotations) rather than a loose dict, so hyperparameters are discoverable and
  type-checked.
- Formatting via `black` + `ruff`, matching the rest of the repo; no notebook-style
  top-to-bottom scripts — everything callable and testable as functions.
- No hardcoded secrets — this stage doesn't need any API keys (local HF dataset +
  local model), but if we ever push the fine-tuned model to the HF Hub, the token
  goes through `.env` / `python-dotenv` like everything else per CLAUDE.md's security
  rules.

---

## 4. Evaluation

### Metric choice: not BLEU alone
BLEU is the default reach-for-it metric for text-to-text HF examples, but it's a
mediocre fit here on its own:
- BLEU rewards n-gram overlap and is somewhat tolerant of local reordering, but ASL
  gloss's whole point is a *specific* word-order transformation (e.g. topic-comment
  structure, time-first ordering) — a BLEU score can look fine on outputs that get the
  vocabulary right but the *order* wrong in ways that matter a lot for a gloss
  consumer (the pose lookup stage cares about sequence order, not bag-of-words
  overlap).
- Gloss vocabularies are much smaller/closed compared to open English generation, so
  exact-match and token-level metrics are more informative than they'd be for general
  MT.

Plan: report a **combination**, not BLEU alone:
- **Exact-match sequence accuracy** (predicted gloss string == reference gloss string,
  after whitespace/case normalization) — strict, but a real "did we nail it" signal
  and easy to sanity-check by hand.
- **Token-level F1** (precision/recall over the multiset of gloss tokens, ignoring
  order) — captures "got the right words" independent of ordering, useful to
  distinguish a vocabulary problem from an ordering problem.
- **BLEU (sacreBLEU)** reported alongside as a familiar reference point for comparing
  across training runs/checkpoints, but explicitly *not* the metric used to decide
  "is this good enough" on its own.
- Optionally, a simple **word-order/permutation distance** (e.g. average token-level
  edit distance or Kendall-tau-style ordering score between predicted and reference
  gloss, for pairs sharing the same token multiset) if exact-match turns out too
  strict to be useful — flagged here as a nice-to-have, not a commitment.

### "Good enough" bar for a first pass
- Numeric target is deliberately loose given we haven't seen real eval numbers yet:
  treat a first pass as viable if exact-match accuracy is clearly above a trivial
  baseline (e.g. above copying the English input unchanged, and above a naive
  "uppercase + drop stopwords" heuristic baseline that should also get implemented as
  a comparison point in `evaluate.py`) and token-F1 is comfortably high (rough
  aspiration: 0.7+) even where exact-match is low.
- Equally important: a **qualitative spot-check** on ~15-20 hand-picked English
  sentences that were *not* in the training set (a few simple/short, a few with
  negation, a few with questions, a few longer/compound sentences) — read the
  gloss output and judge by eye whether it's plausible ASL gloss, not just whether
  it scores well. This matters because ASLG-PC12 is rule-generated and may have its
  own systematic quirks (per Section 1) that a model can learn to reproduce faithfully
  while still being a poor fit for genuinely natural gloss.
- Treat the first fine-tuning run as a baseline to compare against, not a
  pass/fail gate — the real bar-setting happens after seeing actual numbers and
  actual sample outputs.

### Test coverage for `tests/`
Per CLAUDE.md, every new module here needs a corresponding test. Planned test files
(describing *what* they check, not implementations):
- `tests/test_data_prep.py`
  - `load_raw_dataset()` returns a dataset with the expected column names and a
    non-zero row count (can run against a tiny cached fixture or a few real rows —
    avoid re-downloading the full dataset on every test run).
  - `preprocess()` produces the expected input string shape (task prefix present,
    matches `"translate English to ASL gloss: "` + source) and that tokenized
    outputs respect the configured max length (no silently-untruncated overlong
    sequences).
  - `build_splits()` is deterministic given a fixed seed (same split indices across
    two calls) and produces splits that don't overlap (no row appears in more than
    one of train/val/test).
  - A dedup check: feeding in a dataset with a known injected duplicate pair results
    in only one copy surviving into the split dataset.
- `tests/test_evaluate.py`
  - `compute_metrics()` returns expected values on hand-constructed
    prediction/reference pairs with a known answer (e.g. identical strings ->
    exact-match 1.0 and F1 1.0; completely disjoint token sets -> F1 0.0).
  - The naive baseline function (uppercase/stopword-drop heuristic) runs without
    error and produces a lower score than a "perfect" prediction on a toy example.
- `tests/test_inference.py`
  - `translate()` (once a checkpoint exists) returns a non-empty string for a
    non-empty input, and raises/handles gracefully on empty or absurdly long input
    (ties into CLAUDE.md's "sanitize and length-limit all user text/audio input"
    security rule — even though that rule is stated for the live endpoint, this
    module is the eventual callee, so basic input guarding belongs here too).
  - Can be marked to skip gracefully (`pytest.mark.skipif`) if no checkpoint is
    present yet, so the test suite doesn't hard-fail before training has ever run,
    while still running fully in CI/local envs once a checkpoint exists.

Per CLAUDE.md's build-order rule ("don't move to the next pipeline stage until the
current stage's tests pass"), Stage 4 work should not start until these pass against
a real, if modest, fine-tuned checkpoint — not just against mocked-out logic.

---

## Known limitations

- **ASLG-PC12 is rule-generated, not human-translated** (Othman & Jemni 2012), and
  inspection confirms this concretely: 1,466 rows (1.7%) have `gloss` that is just
  `text` case-flipped with no real transformation, and the corpus has **zero**
  fingerspelling-marker or classifier (`CL:`) annotations across all 87,710 rows. The
  gloss a model trained on this data produces will reflect these rule-based artifacts
  (fairly literal reordering, thin non-manual/classifier annotation) rather than gloss
  produced by fluent signers. Decision (approved): treat this as an acceptable ceiling
  for a first working pipeline for an informal, everyday-communication assistive tool —
  not a blocker — but it should also be noted in the top-level README once Stage 3 is
  presented as a working feature, so users understand the translation quality ceiling.
- **No official validation/test split**: scores are only as good as our own 90/5/5
  split (Section 2), not comparable to any published benchmark number for this dataset.
- **Licensing**: not yet confirmed on the HF dataset page / achrafothman.net. Not a
  blocker for local prototyping, but needs a check before any public release or
  redistribution.

## Open questions / risks — resolved

1. ~~Dataset identifier confidence~~ — **resolved**: `achrafothman/aslg_pc12` confirmed
   correct via a real `load_dataset()` call (Section 0).
2. ~~Single split only~~ — **confirmed as expected**: one `train` split, split
   ourselves 90/5/5 with a fixed seed (Section 2).
3. ~~Rule-generated corpus quality ceiling~~ — **acknowledged, not blocking** (see
   "Known limitations" above). Decision made by the user.
4. ~~Compute budget~~ — **resolved**: this machine is CPU-only (no CUDA device
   detected). Decision: local runs use `--quick` mode (small subset, 1 epoch) purely to
   verify the pipeline end-to-end; the real fine-tuning run happens on Kaggle (GPU).
   `train.py` is a plain script (`if __name__ == "__main__":`), so it runs unmodified
   in a Kaggle notebook cell or as `!python -m gloss_model.train`.
5. **Fingerspelling/classifier/non-manual coverage — confirmed absent** (Section 1):
   zero matches in the full corpus. Stage 4's fingerspelling fallback will carry real
   load for proper nouns/numbers/OOV words, not a rare edge case.

---

## Implementation status

Implemented: `config.py`, `data_prep.py` (load/clean/dedupe/split/tokenize),
`evaluate.py` (exact-match, token-F1, BLEU, naive baseline, qualitative spot-check
harness), `train.py` (CLI with `--quick` mode), `inference.py` (with input
sanitization: empty-input and length-limit guards, matching CLAUDE.md's security
rules). Tests in `tests/test_data_prep.py`, `tests/test_evaluate.py`,
`tests/test_inference.py` — 25 tests, all passing (`black`/`ruff` clean).

**Verified end-to-end on a CPU quick run** (`python -m gloss_model.train --quick`:
500 train / 100 validation / 100 test rows, 1 epoch, batch size 8):
- Pipeline completes without error: load → dedupe (87,710 → 81,091) → split →
  tokenize → train → per-epoch eval → save checkpoint → held-out test eval →
  qualitative spot-check.
- Validation: `exact_match=0.0, token_f1=0.303, bleu=3.92`. Test: `exact_match=0.0,
  token_f1=0.313, bleu=4.99`. These numbers are expected to be weak — 500 examples and
  1 epoch is a pipeline smoke test, not a real training budget — and the qualitative
  spot-check output is correspondingly rough (e.g. repetition artifacts like `"X-I AM
  X-I X-I AM X-I..."`), consistent with an undertrained T5 rather than a pipeline bug.
- **Bug found and fixed during this run**: the real test-split evaluation (larger,
  more eval batches than validation) crashed with `OverflowError: can't convert
  negative int to unsigned` inside `decode_predictions`. Root cause: `Seq2SeqTrainer`
  pads variable-length generated sequences across accumulated eval batches with `-100`
  (the label-ignore sentinel), not the tokenizer's pad id — so `predictions` can carry
  `-100` just like `labels` does, and the original code only cleaned `-100` out of
  `labels`. Fixed in `evaluate.decode_predictions` by applying the same `-100` →
  pad-token replacement to `predictions`; added a regression test
  (`test_decode_predictions_handles_ignore_index_in_predictions_too`) that would have
  caught this. Worth remembering for any future eval code that decodes
  `Seq2SeqTrainer`-generated predictions.
- Also capped the held-out test split by `max_eval_samples` in `--quick` mode
  (`train.py`) — otherwise a "fast sanity check" run still paid for an uncapped
  ~15-minute generation pass over the full 4,055-row test split.

**Not yet done**: a real fine-tuning run at full scale (full ~81k-row train split,
more epochs) on GPU compute (Kaggle, per the user's decision). The `--quick`-verified
pipeline should run as-is there; only `--epochs`/`--batch-size`/dropping `--quick`
should be needed. Stage 4 (pose library) work should wait until a real run's
metrics + spot-checks have been reviewed, per CLAUDE.md's build-order rule.
