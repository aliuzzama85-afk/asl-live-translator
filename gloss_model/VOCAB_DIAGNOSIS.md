# Vocabulary Coverage Diagnosis

Diagnostic only — no code or training changes made. Triggered by three spot-check
misses against the real Kaggle checkpoint: "bathroom" → "BEDROOM", "coffee" →
"CAffe", "weather" → "WEST" (see the spot-check review in this session).

## Method

Built the exact same training split used for the real Kaggle run —
`data_prep.build_splits(data_prep.load_raw_dataset())`, seed 42, 90/5/5 —
giving **72,981 training rows**. Tokenized the English `text` column only
(not gloss): lowercased, `[a-zA-Z']+` regex (strips punctuation, keeps
contractions), counted every token across all training rows.

- Total word tokens: **893,807**
- Total unique word types: **19,849**

## 1. The three failing words

| Word | Count in training split |
|---|---|
| `bathroom` | **0** |
| `coffee` | **2** |
| `weather` | **12** |

`bathroom` does not appear even once, in any form. Checked near-synonyms too,
since a model could plausibly have learned the *concept* under a different
surface word: `toilet`=1, `restroom`=0, `washroom`=0, `lavatory`=0,
`bathrooms`=0. The concept, not just this exact word, is essentially absent
from training data.

## 2. Broader 18-word sample (your list of 20 had two repeats — `station` and
`party` — so 18 unique words were checked)

| Word | Count | Rank (of 19,849) | Frequency position |
|---|---|---|---|
| `question` | 970 | 105 | top 0.6% — very common |
| `tomorrow` | 164 | 670 | top 3.4% — common |
| `party` | 143 | 740 | top 3.8% — common |
| `hour` | 27 | 2,700 | top 14% |
| `seat` | 20 | 3,322 | top 17% |
| `weather` | 12 | 4,503 | top 23% |
| `station` | 13 | 4,269 | top 22% |
| `phone` | 13 | 4,168 | top 21% |
| `doctor` | 8 | 5,347 | top 27% |
| `dark` | 8 | 5,608 | top 28% |
| `store` | 6 | 6,118 | top 31% |
| `tired` | 3 | 9,603 | top 48% (rank-wise — see caveat below) |
| `hungry` | 3 | 9,679 | top 49% (rank-wise — see caveat below) |
| `coffee` | 2 | 11,298 | rank-wise mid-table (see caveat below) |
| `keys` | 1 | 14,230 | bottom 28% by rank |
| `lights` | 1 | 19,336 | bottom 3% by rank |
| `battery` | 1 | 18,706 | bottom 6% by rank |
| `bathroom` | 0 | — | absent entirely |

**Caveat on rank/percentile for low counts**: this vocabulary is heavily
Zipfian — roughly 8,500 of the 19,849 unique words (~43%) occur only once.
That flattens the rank-percentile for anything with count 1-3: `coffee` at a
raw count of 2 lands at rank 11,298, which *sounds* like unremarkable
"mid-table," but that's an artifact of thousands of tied-at-the-bottom
singleton words, not evidence that `coffee` is reasonably well-represented.
**Raw count is the more meaningful signal here than rank/percentile** — 2
occurrences across 72,981 sentences is not enough for a model to learn a
reliable mapping regardless of where that lands in a rank ordering.

## 3. Is this narrow or systemic?

Systemic, not narrow. Of the 18 sampled everyday words:
- **7 have single-digit counts or zero** (`bathroom`, `keys`, `lights`,
  `battery`, `coffee`, `hungry`, `tired`) — essentially unlearnable.
- **6 more are in single-to-low-double digits** (`doctor`, `dark`, `store`,
  `weather`, `station`, `phone`) — thin enough that reliable generalization
  is unlikely.
- Only **3 are genuinely well-represented** (`question`=970, `tomorrow`=164,
  `party`=143) — and per the root-cause finding below, that's not a
  coincidence of good general coverage; it's the same register bias showing
  up from the other direction (see below).

## Root cause: corpus register, not vocabulary size

The unique-vocabulary count (19,849 word types over ~73k sentences) isn't
small in absolute terms — the problem is *what* that vocabulary consists of.
The 30 most frequent words in the training split are dominated by formal,
procedural, institutional language:

| Word | Count |
|---|---|
| `european` | 5,380 |
| `commission` | 2,382 |
| `parliament` | 1,892 |
| `member` | 1,746 |
| `council` | 1,275 |
| `vote` | 1,358 |
| `directive` | 489 |
| `committee` | 341 |
| `minister` | 181 |

This confirms empirically what `PLAN.md` already noted about ASLG-PC12's
provenance (Europarl-style EU parliamentary proceedings, not conversational
text): the corpus's dominant register is formal political/legislative
discourse. `question` (970) and `tomorrow` (164) being well-represented
isn't evidence of good everyday-conversation coverage — in this corpus
"question" overwhelmingly means a parliamentary question, and "tomorrow" is
scheduling language for votes/sessions, not "see you tomorrow" small talk.
`party` (143) very plausibly skews toward "political party," not "birthday
party." The words that scored *well* here did so for the wrong reason.

Household, body-need, and casual-social vocabulary (`bathroom`, `coffee`,
`hungry`, `tired`, `keys`, `lights`, `battery`) sits at the bottom of the
frequency distribution because that register barely exists in EU
parliamentary proceedings at all — not because the corpus is small or the
model failed to learn what was there.

## Summary

- The three specific failures are consistent with the data, not surprising
  outliers: `bathroom` (0 occurrences), `coffee` (2), `weather` (12) are all
  at or near the bottom of what a model can be expected to learn.
- This generalizes: most everyday-conversation vocabulary sampled is
  similarly underrepresented or absent.
- Root cause is **domain/register mismatch** between the training corpus
  (formal EU parliamentary text) and the assistive tool's actual target use
  case (informal everyday communication, per `CLAUDE.md`) — not a narrow gap
  fixable by patching in a handful of missing words, and not a training-run
  or hyperparameter problem (the strong aggregate metrics — token-F1 0.965,
  exact-match 0.80 — reflect the model learning this corpus well; the corpus
  itself just doesn't cover the vocabulary this tool needs).

No fix is proposed here — this file is diagnostic only, per the request that
led to it. Decision on how to address this (e.g. supplementary
conversational training data, a vocabulary-augmentation pass, fingerspelling
fallback reliance, or accepting the limitation for a first release) is
still open.

---

## Addendum: dataset provenance and an alternative-corpus check

Diagnostic only, as above — no code or training changes. Two follow-up
investigations, prompted by the register mismatch found above.

### Does `achrafothman/aslg_pc12` (the 87,710-row HF dataset we're using) actually match the ASLG-PC12 paper?

**No.** Read the original paper directly (Othman & Jemni, 2012,
["English-ASL Gloss Parallel Corpus 2012: ASLG-PC12"](https://www.sign-lang.uni-hamburg.de/lrec/pub/12019.pdf),
5th Workshop on the Representation and Processing of Sign Languages). Its
data-source section (3.4, "Collecting data from Gutenberg") is explicit and
exclusive:

> "We start collecting only English data from Gutenberg Project toward
> transform it to ASL gloss. Gutenberg Project (Lebert, 2008) offers over
> 38K free ebooks..."

The word "Europarl," "parliament," or any EU institutional source appears
**nowhere** in the paper — not in the methodology, not in the "Background"
survey of related sign-language corpora, not in the references. The paper
describes a corpus of **over one hundred million sentence pairs** (Table 1:
five parts totaling ~1.59 billion English tokens / ~77 million sentences),
built by running 800 expert-validated transformation rules over Gutenberg
ebook text.

What we actually loaded and sampled (see raw examples below) reads
unmistakably as **Europarl** — the well-known European Parliament plenary
proceedings corpus widely used in machine-translation research — not
Gutenberg literary/book text:

```
"provision of audiovisual media services codified version"
"you should raise the matter again when mr pöttering is in the chair ."
"in writing . I voted in favour of this report even though our amendment was not adopted ."
"the policy of isolation of gaza has failed , radicalising the population who have been its first victim ."
"in other words , yes , europe must show solidarity with greece ."
```

("Mr Pöttering" is Hans-Gert Pöttering, a real MEP and former European
Parliament President; "in the chair," "voted in favour of this report," and
"our amendment was not adopted" are all standard Europarl procedural
phrasing.) This isn't a one-off — it's consistent across a random 15-row
sample from the actual loaded training split, and it matches the earlier
finding that the corpus's top-30 most frequent words are dominated by
`european`, `commission`, `parliament`, `council`, `directive`, `committee`.

This isn't just our own read: the **Hugging Face dataset card for
`achrafothman/aslg_pc12` independently states** "The content appears
derived from European Parliament proceedings, with entries documenting
parliamentary sessions, votes, and procedural items" — while still citing
the Othman & Jemni paper as its source. The card doesn't reconcile that
contradiction; most of its own sections (curation rationale, source
description) are marked "[More Information Needed]."

**Conclusion**: the specific 87,710-row dataset at
`achrafothman/aslg_pc12` on the Hugging Face Hub does not match its own
cited paper's description of its source text. It is very likely Europarl
English text run through the same or a similar gloss-generation rule
pipeline, or a different corpus that got mislabeled/conflated with the
ASLG-PC12 name at some point in its release history — the paper does
mention a planned "third release" with more data, which could be where a
mismatch was introduced, but public documentation doesn't confirm this, and
we could not find a corrected or alternative HF/Kaggle mirror that actually
matches the Gutenberg description within the scope of this check. This
doesn't change any earlier finding in this file — the vocabulary gaps
measured above are real and accurate for the data actually used — but it
reframes the explanation: the gap isn't "everyday words are naturally rare
in a large literary corpus," it's "the corpus we're using is European
political discourse, not literature, despite being labeled and cited as a
Gutenberg-sourced resource."

### How2Sign as an alternative/supplementary corpus

Checked the paper ([arXiv:2008.08143](https://arxiv.org/abs/2008.08143),
Duarte et al., CVPR 2021) and the live project site/download tooling
([how2sign.github.io](https://how2sign.github.io/),
[how2sign/how2sign-data](https://github.com/how2sign/how2sign-data)).

**Size and composition**: 35,191 sentence-level clips total, 79.12 hours of
video (69.62h "Green Screen" studio + 2.96h "Panoptic" 3D-capture studio).
Green Screen split: 31,128 train / 1,741 validation / 2,322 test. Panoptic
studio: 642 validation / 940 test (no dedicated train subset in that arm —
it's a smaller high-fidelity 3D-pose complement, not a standalone training
set). Average 17 words and 5.4 seconds per clip.

**Register**: sourced from "How2," an existing corpus of instructional
"how-to" YouTube videos (cooking, DIY, crafts, etc.). This is a genuinely
different register from Europarl — concrete, procedural, everyday-object
vocabulary — but it's worth being precise that "instructional tutorial
narration" isn't identical to casual conversational chit-chat either. It's
a much better match for this project's "informal everyday communication"
target than parliamentary debate, but not a perfect one.

**Gloss annotation**: real, human-produced — ASL linguists annotated all
35,191 clips using ELAN, at a cited cost of roughly **1 hour of annotation
per 90 seconds of video** (so ~3,165 person-hours for the full corpus — not
something we could feasibly redo ourselves at scale if the released gloss
files turned out to be unavailable). This confirms the premise: it's a
real, non-rule-generated gloss corpus, not synthetic like ASLG-PC12.

**Modalities available**: video (multiview RGB), 2D pose keypoints
(OpenPose, all clips), 3D pose (Panoptic studio subset only), English text
(original + a manually re-aligned version), gloss annotations, depth data
(Green Screen only), original speech audio.

**License**: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
— the same license class as ASLG-PC12's own CC-BY-NC-4.0, so switching or
supplementing wouldn't add a new licensing constraint beyond what's already
accepted for this project.

**Access — the important caveat**: video, 2D pose keypoints, and English
text are directly downloadable today via Google Drive links from the
project site, with no registration or agreement step evident. **Gloss
annotations are not.** Two independent, current sources confirm this isn't
just missing documentation:
1. The live download page's modality section states other modalities are
   "under construction. We will be releasing the other modalities soon!"
2. The official `download_how2sign.sh` script in the `how2sign-data` repo
   has explicit placeholder/TODO comments for `# Gloss annotations` and
   `# Panoptic Studio data` — they're not wired up, not just undocumented.

So despite the paper describing gloss coverage for all 35,191 clips, there
is currently **no direct download** for it. The realistic access path today
is emailing the dataset maintainer (contact listed in the `how2sign-data`
repo: amanda.duarte@upc.edu) to ask whether gloss files can be obtained by
request — with no confirmed availability or turnaround time. This is a real
open risk for this option, not a formality to route around.

**What this means, without proposing a decision**: pose/keypoint data
(relevant to Stage 4 per the original question) and English text are
concretely available right now. Gloss text — the piece that would matter
most for a Stage-3-style fine-tuning task — currently requires reaching out
to the authors and is not guaranteed. Any plan built around How2Sign gloss
data should treat that as an open dependency, not a settled fact.

### Summary

- The corpus we trained on doesn't match what its own citation says it is —
  it's Europarl-register text, not Gutenberg literary text — which is the
  root cause of the vocabulary gaps diagnosed above, not a fixable
  preprocessing oversight on our side.
- How2Sign is a real, appropriately-sized, human-annotated, better-register
  (though not perfectly conversational) gloss corpus — but its gloss
  annotations specifically are not yet publicly downloadable, only its
  video/pose/text modalities are. Getting the gloss data would require
  direct contact with the maintainers and isn't guaranteed.
- No dataset-switch or fix decision is made here — this is diagnostic input
  for that decision, which is still open.
