# Stage 4 Plan: Gloss-to-Pose Lookup Library

Status: **planning only — no code, no downloads, no package installs**. This document
exists so the user can review WLASL's licensing/access reality (Section 1) and the
overall design before any implementation time is spent, per explicit instruction.
Nothing in `pose_library/` beyond this file and the pre-existing empty `__init__.py`
should exist yet.

Scope note: per `CLAUDE.md`'s **build order**, "Stage 4" here means step 2 of the
5-step build order ("Pose library + static skeleton renderer, fed gloss manually") —
not yet wired to live speech/ASR (build-order steps 3-4) or to the fine-tuned T5 model's
live output (that wiring is a later integration step once both Stage 3 and this stage
are independently proven). It corresponds to Stage 4 in the 5-stage *architecture*
numbering (gloss-to-pose lookup). Stage 3 (`gloss_model/`) is signed off — see
`PROJECT_STATUS.md` Section 9 and `CLAUDE.md`'s gotchas log — so this plan treats
Stage 3's outputs (ASL gloss tokens, uppercase, with `X-`/`DESC-` prefixes, per
`gloss_model/data_prep.py`) as the input contract for whatever this stage builds,
without assuming any change to Stage 3.

---

## 1. WLASL access reality

### What was actually checked
Not going from memory alone — pulled the real files from `github.com/dxli94/WLASL`
(`master` branch) during this planning session:
- `start_kit/README.md` (full text)
- `start_kit/WLASL_v0.3.json` (11,932,637 bytes, downloaded and parsed directly)
- `start_kit/C-UDA-1.0.pdf` (downloaded and read in full)
- MediaPipe's current (`developers.google.com/edge/mediapipe`) Hand Landmarker and
  Holistic Landmarker docs, for Section 2

### Distribution format: metadata index + external links, not hosted video
WLASL does **not** ship video files in the repo or via a single bulk download. What
you get is:
- `WLASL_v0.3.json` — a **metadata-only** JSON file (2,000 gloss entries, confirmed by
  direct parse). Each gloss entry has an `instances` list; each instance carries
  `gloss`, `bbox`, `fps`, `frame_start`, `frame_end`, `instance_id`, `signer_id`,
  `source`, `split`, `url`, `variation_id`, `video_id` — but no video bytes.
- `start_kit/video_downloader.py` — downloads the actual videos from each instance's
  `url` using `yt-dlp` (preferred) or `youtube-dl`.
- `start_kit/preprocess.py` — trims each downloaded video to the gloss-relevant
  segment using `frame_start`/`frame_end`/`bbox`.
- `start_kit/find_missing.py` + a Google Form — for videos whose source URL has gone
  dead (a real, acknowledged problem: the README says links "may find the downloaded
  videos incomplete" over time), you can request the maintainer's own cached copies.
  The README's own wording ("I got more occupied, some delays may be expected... If
  urgent, drop me an email") makes clear this is an informal, best-effort channel with
  **no guaranteed turnaround** — not something to plan a schedule around.

**Confirmed dataset scale** (via direct parse of `WLASL_v0.3.json`): 2,000 glosses,
21,083 total video instances, split `train`=14,289 / `val`=3,916 / `test`=2,878. Every
gloss has at least 6 instances (min 6, max 40, median 10) — so there's no single-point-
of-failure instance per word, which helps with the dead-link problem in principle.

**Confirmed source diversity** (from the same parse — this matters a lot for licensing,
see below): instances come from 19 distinct `source` values, not just YouTube:

| source | count | source | count |
|---|---|---|---|
| signingsavvy | 2,668 | asldeafined | 1,833 |
| handspeak | 2,211 | aslu | 1,827 |
| aslsearch | 1,875 | aslsignbank | 1,071 |
| asl5200 | 1,561 | asllex | 814 |
| aslpro | 1,736 | startasl | 623 |
| spreadthesign | 1,584 | northtexas | 295 |
| lillybauer | 269 | aslbrick | 218 |
| nabboud | 200 | elementalasl | 161 |
| valencia-asl | 133 | scott | 36 |

Spot-checking a sample instance (`coffee`) shows the actual mix: some `youtube.com`
URLs (needs `yt-dlp`), but also direct file URLs on a dozen unrelated third-party
domains — `aslpro.com/.../coffee.swf` (Flash — obsolete format, may not download or
play cleanly with modern tooling), `aslsearch.com/.../coffee.mp4`,
`media.spreadthesign.com/...mp4`, `handspeak.com/.../coffee.mp4`,
`signingsavvy.com/.../7924.mp4`, `s3-us-west-1.amazonaws.com/files.startasl.com/...`,
`aslsignbank.haskins.yale.edu/.../protected_media/...` (URL literally contains
"protected" — may be access-gated), `signschool`'s Azure blob storage, etc. This is a
real engineering cost, not a one-line `wget` loop: expect meaningfully-sized YouTube
downloader breakage, dead links needing the missing-video request flow, at least one
obsolete format (`.swf`), and at least one URL that looks auth-walled.

### Licensing — read the actual C-UDA text, not just the README's summary
The dataset is covered by the **Computational Use of Data Agreement v1.0 (C-UDA)**,
full text read directly from `C-UDA-1.0.pdf` in this session. Key clauses, quoted:
- **2.1**: "You agree that you will use the Data solely for **Computational Use**."
  (Section 5.1 defines Computational Use as "activities necessary to enable the use of
  Data... for analysis by a computer.")
- **2.2 / 5.5**: "The C-UDA does not impose any restriction with respect to the use,
  modification, or distribution of **Results**." A "Result" is "anything you develop
  or improve from your use of Data that does not include more than a **de minimis**
  portion of the Data" — and explicitly, "**Artificial intelligence models trained on
  Data** (and which do not include more than a de minimis portion of Data) are
  Results."
- **3.1**: Redistribution of the Data itself is allowed, but only if you (3.1.1) keep
  all attribution/credit info attached, and (3.1.2) bind every downstream recipient to
  the same C-UDA terms.
- **5.2**: "Data" is explicitly defined to include the material "in **modified or
  unmodified form**" — i.e., the C-UDA doesn't stop applying just because you
  transformed the original file.

**This creates a genuinely open legal question this plan flags rather than resolves**:
is a MediaPipe-extracted keypoint sequence (per Section 2/3 below) a "Result" (Sec 2.2,
unrestricted) or still "Data" in "modified form" (Sec 5.2, still bound by 2.1/3.1)? The
"de minimis" test in 5.5 is the crux — a full per-frame landmark trajectory extracted
from one specific video plausibly still substantially derives from and represents that
video's content, which reads more like "modified Data" than a de-minimis-based
"Result" to me. **I'm not a lawyer and this isn't a legal opinion** — flagging this as
a real ambiguity to resolve (or route around, see recommendation below) rather than
asserting an answer either way.

**Separately, and more concretely restrictive**: the WLASL `README.md`'s own
**Disclaimer** section (not part of the formal C-UDA legal text, but the Data
Provider's own stated terms) says, verbatim: *"All the WLASL data is intended for
academic and computational use only. **No commercial usage is allowed.**"* This is
plain and unambiguous, and stricter than the formal C-UDA document's own wording. For
a project explicitly framed as "an assistive-tech prototype" today (per `CLAUDE.md`'s
opening line) this isn't a blocker *yet*, but it is a **real, explicit constraint that
must be revisited before any commercial pivot, paid product, or public
redistribution** — not fine print to note once and forget.

**A second, compounding licensing layer that's easy to miss**: WLASL itself doesn't
own or host the underlying videos — it's an index into ~19 independent third-party
sites' content (see source table above). The C-UDA is WLASL's own license for the
*compiled dataset/metadata it produced*; it cannot sublicense rights WLASL doesn't
hold in, e.g., SigningSavvy's or Spread The Sign's own copyrighted video content.
Several of those sites are commercial ASL-instruction businesses that plausibly have
their own restrictive terms of use for their video content independent of WLASL's
C-UDA. This wasn't independently re-verified site-by-site in this session (out of
scope for a planning pass) — flagging it explicitly as a **thing to spot-check before
bulk-downloading from any single source domain**, not something WLASL's license
automatically clears.

**Recommendation given all of the above**: treat any WLASL-derived pose data as a
**local, non-redistributed development asset** — never committed to git (this is
already true today: `.gitignore` has `pose_library/data/` — confirmed by reading the
file directly), never shipped in a public release build, never sold or offered as part
of a commercial product, and accompanied by a small per-file attribution record (see
Section 3) so Section 3.1.1's attribution obligation is met if this policy is ever
revisited. This is not a blocker for the currently-scoped prototype work, but it is a
decision the user should consciously sign off on given the ambiguity above, not
something to build past silently.

### A realistic vocabulary subset — checked against the real gloss list, not assumed
Read `gloss_model/data/vocab_augmentation.csv` directly (360 hand-authored
English/gloss sentence pairs) and extracted every distinct content token from the
`gloss` column, stripping the `X-`/`DESC-` grammatical prefixes and pronoun markers
(`X-MY`, `X-YOU`, etc.) that aren't independent signs: **231 unique word-stems** (not
the ~121 figure associated with the narrower vocabulary-*patch* list used for T5
fine-tuning specifically — see the flag below; this 231 is a direct, reproducible
count off the actual file, done fresh for this plan).

Cross-referenced all 231 against the real, directly-parsed `WLASL_v0.3.json` gloss
list (2,000 glosses) with exact case-insensitive string match:

- **199 / 231 (86%) matched directly** — e.g. `bathroom`, `coffee`, `weather`,
  `doctor`, `question`, `dark`, `hungry`, `tired`, `party`, `tomorrow` (the exact words
  `VOCAB_DIAGNOSIS.md` flagged as rare-or-absent in ASLG-PC12's *training text* are
  present as **pose vocabulary** in WLASL — a different corpus with a different,
  much better-matched register for this use case).
- A handful of the 32 apparent misses are **phrasing, not absence**: `thank` alone
  isn't a WLASL gloss, but `thank you` is; `wake` alone isn't, but `wake up` is.
  Confirmed by direct substring check.
- After that adjustment, roughly **~28-30 genuine gaps** remain, split into two kinds:
  - **English function words gloss doesn't need signs for anyway**: `be`, `do`, `at`,
    `too`, `would`, `well`, `and` — these are exactly the kind of word ASL grammar
    drops or expresses non-manually, not missing pose-library content.
  - **Real content-word gaps worth noting**: `airport`, `hotel`, `pharmacy`,
    `station`, `park`, `parking`, `seat`, `hand`, `leg`, `menu`, `reservation`,
    `receipt`, `cash`, `confusion`, `negotiable`, `embarrassed`, `cloudy`, `windy`,
    `worried`, `allergic`, `anymore`, `fresh`, `nearest`. `hand` and `leg` being absent
    from a 2,000-word ASL vocabulary is genuinely surprising and wasn't investigated
    further here (possibly present under a different exact string, a compound gloss,
    or a variant not in this top-2000 subset) — flagged as worth a closer look at
    implementation time, not a settled fact.
- **Flagged discrepancy, not silently resolved**: the task brief's "~121 hand-authored
  augmentation words" almost certainly refers to the narrower vocabulary-*patch* list
  used specifically to fix T5 vocabulary gaps (`gloss_model/VOCAB_DIAGNOSIS.md`
  mentions a "121-word patch list" in `PROJECT_STATUS.md`), which is a curated
  subset of, not identical to, the full 231-word set actually present across all 360
  `vocab_augmentation.csv` gloss sentences. This plan uses the larger, directly-counted
  231-word set as the more complete and directly verifiable source of truth for pose
  vocabulary planning, since pose lookup needs every content word that can appear in
  gloss output, not just the words a text-generation vocabulary patch specifically
  targeted.

**Bottom line**: a WLASL-sourced subset covering this project's actual hand-curated
everyday vocabulary is realistic and well-supported by the real data — no need to
touch anywhere near the full 2,000-gloss/21,083-instance dataset. See Section 6 for the
concrete recommended subset size.

---

## 2. MediaPipe extraction pipeline

### Recommendation: MediaPipe **Holistic Landmarker**, not Hands alone
Checked MediaPipe's **current** Tasks API docs directly (`developers.google.com/edge/
mediapipe/solutions/vision/...`, not the old, now-404 `google.github.io/mediapipe/
solutions/holistic.html` legacy Solutions page — confirmed the legacy URL 404s and the
new Task-based Holistic Landmarker page is live today, so this is an actively
maintained current API, not a deprecated one to avoid).

- **Hand Landmarker** alone outputs: handedness (left/right), hand landmarks in image
  coordinates, hand landmarks in world coordinates — **21 landmarks per hand**
  (confirmed from the live docs page).
- **Holistic Landmarker** outputs a combined **553 landmarks**: 33 pose landmarks
  (body), 478 face landmarks (including 10 iris points), and 21 landmarks per hand
  (42 total) — plus optional face blendshape scores and an optional segmentation mask.
  All three land in the same coordinate frame from one pipeline call.

**Why Holistic despite only needing hands today**: `CLAUDE.md`'s own architecture
notes flag that ASL grammar carries real meaning in **non-manual markers** — eyebrow
raises for yes/no questions, head tilts, mouth morphemes, body lean for role-shifting —
and Stage 3's own `PLAN.md` independently confirms (Section 1) that ASLG-PC12 has
**zero** annotation for any of this, meaning any such expressiveness would have to be
sourced from the *pose data* layer, not the gloss text layer, if it's ever added.
Extracting **only hand keypoints today but storing them inside a schema shaped like a
Holistic output** (Section 3) costs relatively little extra compute per video now and
avoids a full re-extraction pass over the whole library later if/when facial or body
context becomes a real feature requirement. Recommendation: **run Holistic, but only
persist the hand + (lightweight) pose landmark subsets for the current milestone**,
leaving face landmarks/blendshapes as an easy add-on field later rather than a
from-scratch re-processing job. This is a compute-vs-future-flexibility trade, not a
free lunch — Holistic is a heavier model than Hands alone, so this should be
re-evaluated if per-video processing time becomes a bottleneck at the vocabulary sizes
in Section 6 (a few hundred short clips is unlikely to be a real bottleneck either way).

### Pipeline shape (video in, keypoints out)
1. Downloaded/trimmed WLASL clip (per Section 1) → decode frames (e.g. via
   `cv2.VideoCapture` or MediaPipe's own video-frame utilities).
2. Run Holistic Landmarker per frame (`running_mode="VIDEO"` per the Tasks API, which
   is the documented mode for decoded video frames vs. live streams).
3. Extract, per frame: hand landmarks (21 x,y,z per hand, both normalized-image and
   world coordinates — recommend keeping normalized-image coordinates as the primary
   stored representation since that's what a 2D skeleton renderer, Stage 5, consumes
   directly) and a minimal pose subset (e.g. shoulders/wrists/elbows only, not all 33,
   to keep file size down — full field-selection decision belongs in implementation,
   not this plan).
4. Drop frames where MediaPipe fails to detect a hand (occlusion, motion blur) rather
   than inventing interpolated values at this stage — flag as a known-gap frame in the
   stored metadata (Section 3) so a later smoothing/interpolation stage (build-order
   step 3, out of scope here) has an honest signal instead of silently-wrong zeros.
5. Serialize the resulting per-frame landmark list + metadata to JSON (Section 3).

---

## 3. Storage format

### Recommendation: one JSON file per gloss word, not one big indexed file
`pose_library/data/` — **confirmed already gitignored** (`.gitignore` line
`pose_library/data/`, read directly) — is where this lives, consistent with the
project's existing "Data (large, don't commit)" gitignore convention already applied
to `data/raw/` and `gloss_model/checkpoints*/`.

Recommended layout:
```
pose_library/
  data/                          # gitignored — WLASL-derived, license-sensitive
    poses/
      bathroom.json
      coffee.json
      weather.json
      ...
    manifest.json                # small index: word -> filename, source, license tag
  fingerspelling/                # NOT gitignored — self-recorded, project-owned
    alphabet.json                # all 26 letters, one small file (see Section 5)
  lookup.py                      # get_pose_sequence(), fallback composition
  extract.py                     # MediaPipe extraction pipeline (Section 2)
  types.py                       # PoseSequence dataclass (Section 4)
  __init__.py
```

**One-file-per-word over one big file, justified against the three criteria asked
for**:
- **Lookup speed**: at the vocabulary sizes this stage actually needs (Section 6:
  ~150-250 words, not 2,000+), a filesystem `open(f"{word}.json")` is effectively O(1)
  and needs no in-memory index of the whole library loaded up front — useful since the
  live pipeline (build-order step 4, later) shouldn't need to hold every pose sequence
  in memory just to answer one lookup.
- **Adding new words later**: dropping in one new file has zero risk of touching or
  re-serializing unrelated words, no file-lock/merge contention if two people or
  processes add different words at the same time, and no risk of a partially-written
  giant file corrupting the whole library on a crash mid-write.
- **Diffability/git-friendliness**: moot for the WLASL-derived `poses/` directory today
  since it's gitignored — but **not** moot for `fingerspelling/`, which this plan
  recommends keeping *out* of `.gitignore` since it's small, fixed-size (26 letters),
  and fully project-owned (self-recorded, no WLASL license question at all — see
  Section 5). One file per gloss word means a future edit to one WLASL-sourced pose
  (e.g. re-extracting `coffee` with a better source clip) would show as a clean
  single-file diff if this policy is ever revisited, rather than an opaque diff inside
  one monolithic blob.
- A small `manifest.json` (word → filename, WLASL `source`/`video_id`/`instance_id`,
  and a license tag) is worth keeping alongside the per-word files specifically to
  satisfy the C-UDA's Section 3.1.1 attribution-retention requirement (Section 1) if
  this data is ever handled by anyone other than the original extractor — cheap
  insurance, not required for local-only use today.

---

## 4. Lookup interface

### Return type: a small typed `PoseSequence`, not a bare list/array
```python
@dataclass(frozen=True)
class PoseSequence:
    """A single sign's extracted keypoint sequence.

    Attributes:
        gloss: The gloss word this sequence represents (uppercase, matching
            gloss_model's output convention).
        fps: Frame rate the sequence was extracted/should be replayed at.
        landmark_names: Ordered names/indices for each landmark column, so
            frame data is self-describing rather than a bare positional array.
        frames: One entry per frame; each frame is a list of
            (x, y, z) tuples (normalized image coordinates), one per
            landmark_names entry, in the same order.
        source: Attribution/provenance string (e.g. WLASL video_id + source
            site, or "fingerspelling:self-recorded"), kept for the C-UDA
            attribution requirement noted in Section 1/3.
    """

    gloss: str
    fps: float
    landmark_names: tuple[str, ...]
    frames: list[list[tuple[float, float, float]]]
    source: str
```
A numpy array of shape `(num_frames, num_landmarks, 3)` is the natural in-memory form
once loaded (and worth exposing as a `.as_array()` convenience method for the
renderer), but the *stored/returned* type should stay a small dataclass wrapping plain
Python lists — keeps the JSON on disk human-diffable (relevant for the
`fingerspelling/alphabet.json` file, which this plan recommends actually committing;
see Section 3) and avoids forcing numpy as a hard dependency of anything that merely
wants to inspect metadata (e.g. `source`) without paying for array construction.

### Behavior on a miss: return `None`, don't raise
```python
def get_pose_sequence(gloss_word: str) -> PoseSequence | None:
    """Look up a single gloss word's pose sequence.

    Args:
        gloss_word: A single ASL gloss token (case-insensitive; internally
            normalized to lowercase to match stored filenames).

    Returns:
        The word's PoseSequence if present in the library, otherwise None.
        A miss is an expected, common outcome (see Stage 3's confirmed
        finding that ASLG-PC12 has zero fingerspelling coverage and this
        library's vocabulary is a small curated subset, not full ASL) — not
        an error condition — so callers should treat None as "try the
        fingerspelling fallback," not as a bug signal.
    """
```
**Recommendation: `None`, not an exception, and not a bare boolean/sentinel.** Reasons,
argued from how `pipeline/` will actually consume this (per `CLAUDE.md`'s explicit
"never a silent frozen avatar" requirement):
- OOV gloss words are an **expected, frequent** runtime case for this project
  specifically (confirmed twice over: Stage 3's `PLAN.md` found zero fingerspelling
  markers in the training corpus, and `VOCAB_DIAGNOSIS.md` found the training corpus's
  register barely overlaps this tool's target vocabulary at all) — treating every miss
  as an exception would mean try/except-wrapping nearly every real call site, which is
  a sign the miss belongs in the return type, not the control-flow-via-exceptions path.
  Reserve raising for genuinely invalid input (e.g. an empty string, or a non-string) —
  that's a caller bug, not a vocabulary gap, and should fail loudly and immediately.
- A live translation pipeline processing a streamed phrase word-by-word needs to keep
  going even when one word in the middle is OOV — an uncaught/re-raised exception on
  one word would either crash the whole phrase's rendering or force every caller to
  wrap every lookup in its own try/except, duplicating the same handling everywhere.
  `None` lets the caller compose cleanly: `get_pose_sequence(w) or
  fingerspell(w)`-shaped logic in one place (Section 5), which is also where a "now
  fingerspelling..." loading-state cue (again, per `CLAUDE.md`'s UI/UX requirement to
  "show clear loading/latency state, never a silent frozen avatar") naturally belongs.

---

## 5. Fingerspelling fallback

> **Superseded 2026-09-26** by [`FINGERSPELLING_PLAN.md`](FINGERSPELLING_PLAN.md).
> The data-source decision below (self-record all 26 letters) no longer
> holds: 24 letters now come from the MIT-licensed `sid220/asl-now-fingerspelling`
> dataset, and only J and Z (motion letters the dataset lacks) are
> self-recorded (`FINGERSPELLING_PLAN.md` Section 2b). The
> `get_fingerspelling_sequence()`/`resolve_gloss_word()` interface sketched
> here was never built. It was redesigned so the resolution lives in the
> frontend next to `stitchTimelines` (see that doc's Section 4). The storage
> layout in Section 3 (`fingerspelling/alphabet.json`) is also superseded:
> it's now one JSON per letter plus a manifest, the same shape as the word
> library. Kept below as the original record.

### Trigger condition
Purely a `get_pose_sequence()` miss (`None` return) on a whole gloss word — **not**
driven by any special marker in the gloss text itself. This is a deliberate
consequence of Stage 3's own finding: `gloss_model/PLAN.md` Section 1 confirmed **zero**
rows in the full 87,710-row ASLG-PC12 corpus matched a fingerspelling-like pattern
(`X-X-X`) — the model was never trained to emit an explicit fingerspelling marker, so
Stage 4 can't rely on one existing in its input. Fingerspelling is entirely this
stage's own decision, made at lookup time.

### Data source: self-record the 26-letter alphabet, don't source it from WLASL
Checked directly whether WLASL's own single-letter gloss entries could cover this:
`WLASL_v0.3.json` does contain 21 single-letter glosses —
`a, b, d, e, f, g, h, i, j, k, m, n, o, p, q, r, s, t, u, v, w` — but **not**
`c, l, x, y, z` (confirmed by direct query; 5 of 26 letters are simply absent from this
2,000-gloss subset).

**Recommendation: self-record all 26 letters with this project's own MediaPipe Hands
capture, rather than using WLASL's partial letter set.** Reasoning:
1. It's incomplete on its own (5 letters missing) — some second source would be needed
   regardless.
2. The letters that *are* present still carry the exact same source-site licensing
   questions raised in Section 1 (they're drawn from the same 19-source pool), for a
   component that's small enough to fully own instead.
3. Fingerspelling handshapes are standardized/iconic — unlike whole-word signs, where
   dataset scale and signer diversity genuinely matter for coverage and robustness,
   26 letters is small enough that **one clean, project-owned recording session**
   plausibly suffices for a first pass, and sidesteps the Section 1 licensing question
   entirely for this component (self-recorded content the project fully owns can be
   committed to git outright — see Section 3's recommendation to keep
   `fingerspelling/` out of `.gitignore`, unlike the WLASL-derived `poses/`
   directory).
4. A public "ASL Alphabet" image dataset (e.g. a Kaggle-hosted one) is a known
   fallback option if self-recording proves impractical — **flagged with explicit
   uncertainty**: this wasn't independently checked for exact licensing terms in this
   session, and it's static images (handshape only, no motion), not keypoint
   sequences, so it would still need an offline MediaPipe extraction pass plus a
   license check before use. Not recommended as the first option given self-recording
   is simpler and legally cleaner for this specific, small, standardized component.

**Important handshape detail to carry into the capture plan**: standard ASL
fingerspelling has **two letters that require motion, not a static handshape** — `J`
and `Z` are each drawn as a small trajectory (a hook/loop for J, a "Z" stroke shape for
Z), not held still like the other 24 letters. `PoseSequence.frames` already supports
multi-frame sequences (Section 4), so this isn't a schema change, but it does mean the
capture session needs to record J and Z as a short clip each, not a single still frame
like the other 24 — worth stating explicitly now so it isn't discovered as a surprise
mid-recording.

### Integration with `get_pose_sequence`'s interface
Keep fingerspelling as a separate function plus one small composing wrapper, rather
than baking letter-splitting into `get_pose_sequence` itself:
```python
def get_fingerspelling_sequence(letter: str) -> PoseSequence:
    """Look up a single letter's fingerspelling handshape.

    Args:
        letter: A single a-z character (case-insensitive).

    Returns:
        The letter's PoseSequence. Always present for a valid a-z input,
        since the full 26-letter alphabet is a fixed, self-owned, always-
        complete asset (unlike the open-ended word library) — raises
        ValueError for anything outside a-z rather than returning None,
        since an invalid letter here is a caller bug, not an expected gap.
    """


def resolve_gloss_word(gloss_word: str) -> PoseSequence | list[PoseSequence]:
    """Resolve one gloss word to pose data, falling back to fingerspelling.

    Args:
        gloss_word: A single ASL gloss token.

    Returns:
        A single PoseSequence if gloss_word is in the pose library, or a
        list of per-letter PoseSequences (one per character) if it must be
        fingerspelled. Callers (the eventual renderer) must handle both
        shapes — this is a deliberate part of the contract, documented here
        so Stage 5's renderer interface is designed for it from the start
        rather than retrofitted after the fact.
    """
```
Keeping `get_pose_sequence` itself simple (single word → `PoseSequence | None`, per
Section 4) and layering the fallback as its own composed function keeps the "is this
word in the library at all" question testable in isolation from the "how do we spell
it out" question — two different pieces of logic with two different data sources and
two different test suites, per `CLAUDE.md`'s testing requirement.

### Scope note: this will be exercised often, not rarely
Worth restating from Section 1/Stage 3's own findings rather than assuming
fingerspelling is a rare edge case: ASLG-PC12 has zero native fingerspelling
annotation, its vocabulary register is a poor match for this tool's everyday-
communication target (`VOCAB_DIAGNOSIS.md`'s addendum), and even the curated 231-word
`vocab_augmentation.csv` list only has 86% direct WLASL coverage. Proper nouns,
numbers, and any content word outside both the training corpus and whatever subset of
WLASL gets processed (Section 6) will hit this path. Budget real testing time for it,
not a token unit test.

---

## 6. Realistic first-pass scope

### Recommendation: ~150-250 words, not "as much of WLASL as possible"
Concretely: process **one video instance per target gloss word** (not all 6-40
available instances per word) for a target list built from the 199 directly-confirmed
WLASL matches against `vocab_augmentation.csv`'s 231-word vocabulary (Section 1), plus
the 26-letter fingerspelling alphabet (Section 5, self-recorded, separate track) — a
total processing job on the order of **~200 short video downloads/extractions**, not
2,000+.

### Why not more
- **This vocabulary is the one part of the project already known to match the target
  register.** `VOCAB_DIAGNOSIS.md` established, with real frequency data, that
  ASLG-PC12 (Stage 3's training corpus) is dominated by EU-parliamentary-style formal
  discourse, not the "informal, everyday communication" register `CLAUDE.md` actually
  targets — and that `vocab_augmentation.csv` was specifically hand-authored to cover
  that everyday-communication gap. It's the most direct, already-validated proxy this
  project has for "what gloss vocabulary will this tool actually need to render,"
  which makes it the right first target for pose data specifically — a better basis
  for scoping than "cover as many WLASL glosses as possible" or "cover whatever
  ASLG-PC12 happens to produce most often" (which Section 1 already showed skews
  toward political/procedural vocabulary this tool doesn't need).
- **Per-instance selection over exhaustive collection**: downloading every available
  instance of every word multiplies engineering time (more dead links, more `.swf`/
  auth-walled edge cases per Section 1) for a benefit — signer/dialect variation
  robustness — that matters for a production sign-recognition model, not for a
  first-pass lookup library whose whole job (per the build order) is to prove Stage 4
  + 5 work end-to-end. One clean instance per word, preferring a durable direct `.mp4`
  source over `.swf`/YouTube/auth-gated URLs where the metadata offers a choice,
  is enough to validate the full path: download → extract → store → look up → render.
- **Build-order discipline**: `CLAUDE.md` explicitly says "don't move to the next
  pipeline stage until the current stage's tests pass," and separately that Stage 4 +
  5's job is to get a *static* skeleton renderer working off *manually*-fed gloss
  before touching live speech at all. A 2,000-word download-and-process effort is real
  engineering investment with license risk (Section 1) attached, spent before the
  renderer and the fingerspelling fallback have even been proven out once — exactly
  the kind of over-building-ahead-of-validation the staged build order is designed to
  prevent. Scale up the vocabulary later, once Stage 5's renderer is real and the
  fallback path (which will legitimately carry a lot of the load per Section 5) has
  been exercised against real gaps, not hypothetical ones.
- **Explicit non-goal**: 100% vocabulary coverage before moving on. An OOV rate that
  exercises the fingerspelling fallback during Stage 4+5 testing is a **feature of the
  test plan** (it's the fallback path's own integration test), not a shortfall to
  eliminate first.

---

## Known gotchas / open questions (carried into implementation)

1. **WLASL licensing ambiguity (Section 1) is a real open decision, not resolved
   here.** Whether extracted keypoints count as C-UDA "Data" (still restricted) or a
   "Result" (unrestricted) under Section 5.5's de-minimis test is genuinely unclear
   from the C-UDA text alone, and the README's own plain "no commercial usage"
   disclaimer is a real constraint on this project's future, not just today's
   prototype scope. Recommendation stands: local-only, gitignored, unmarketed,
   revisit before any public/commercial step — but this needs the user's explicit
   sign-off before any WLASL video is actually downloaded, not just a plan-level note.
2. **Third-party source-site terms (19 distinct hosting domains) were not individually
   audited in this session.** Spot-check the specific domains a real subset ends up
   pulling from (Section 1's table) before bulk-downloading, particularly the
   commercial ASL-instruction sites in that list.
3. **`hand` and `leg` being absent from WLASL's 2,000-word gloss list is unexplained**
   and worth a closer look at implementation time — possibly present under a different
   string, a compound gloss, or genuinely outside the top-2,000 subset.
4. **The Kaggle-hosted ASL-alphabet-style datasets mentioned in Section 5 were not
   licensing-verified in this session** — flagged as an unverified fallback option
   only, not a recommended primary source.
5. **`.swf` (Flash) source videos (Section 1, `aslpro` source, 1,736 instances) may not
   download or decode cleanly with current tooling** — worth a small feasibility check
   before counting on any `aslpro`-sourced instance for the target word list.
