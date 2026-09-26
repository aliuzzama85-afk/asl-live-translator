# Fingerspelling Alphabet Plan (v1): A–Z Fallback

Scope: when a gloss word isn't in the 118-word WLASL pose library, spell it out
letter by letter with a 26-letter ASL fingerspelling alphabet, instead of
skipping it with "NOT FOUND". The letters now come from the MIT-licensed
`sid220/asl-now-fingerspelling` dataset (24 letters, Section 2b), with
self-recording (Section 1) for J and Z, the two letters the dataset can't
provide. This supersedes `pose_library/PLAN.md`
Section 5's earlier sketch (`get_fingerspelling_sequence()` /
`resolve_gloss_word()` — **never built**, and designed before
`stitchTimelines()` existed; see Section 4 for why it's redesigned rather than
preserved).

Baseline checked directly before writing this (not assumed): `CLAUDE.md`,
`PROJECT_STATUS.md`, `pose_library/PLAN.md`, `frontend/MULTIWORD_PLAN.md`,
`pose_library/{extract,download,manifest,build_library,lookup,types,config}.py`,
`frontend/src/{App.jsx,lib/stitchTimelines.js,lib/reconstructTimeline.js,
hooks/usePoseSequences.js,hooks/usePoseSequence.js,components/CaptionBand.jsx}`,
`frontend/vite.config.js`, `.gitignore`, and real pose JSON/manifest data on
disk (details in Section 2).

**Data status up front, stated plainly (updated 2026-09-26)**: **all 26
letters have real data.** 24 are converted from the MIT-licensed
[`sid220/asl-now-fingerspelling`](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
dataset (Section 2b). **J and Z are self-recorded** with the recorder in
Section 1, because the dataset has no motion data for them and they're motion
letters (Section 2c). This doc was first written when *no* letters existed
and the plan was to self-record all 26; Sections 1 and 2 describe that
recording path, which was used for J and Z and remains valid for re-recording
any letter. Synthetic placeholder letters under
`tests/fixtures/fingerspelling_synthetic/` still exist for plumbing tests only.

---

## 1. Recording process

### Format: match what the existing pipeline was validated on
`extract.py` itself is format-agnostic: it decodes any file `cv2.VideoCapture`
can open and reads `fps` from the file's own container metadata
(`cap.get(cv2.CAP_PROP_FPS)`), with no resolution assumption anywhere. So
nothing *requires* a specific format — but the pipeline has only ever been
validated against WLASL trimmed clips, and those are (checked directly with
OpenCV on `pose_library/data/trimmed_videos/about.mp4`) **640×480, ~30fps**.
Decision: record at **640×480, 30fps requested**, written as `.mp4` with the
`mp4v` codec.

- **Why `mp4v`, not H.264**: OpenCV's `VideoWriter` H.264 (`avc1`) path needs
  a separately-installed OpenH264 DLL on Windows (this project's dev machine)
  and silently fails to open without it; `mp4v` works out of the box with
  the pinned `opencv-python` wheel and is exactly what
  `tests/test_pose_extract.py` already uses to write synthetic test videos.
  Slightly larger files, not a real concern at 26 short clips.
- **fps is measured, not trusted**: webcams frequently report a nominal 30fps
  via `CAP_PROP_FPS` while actually delivering fewer frames (low light, USB
  bandwidth). Since `extract.py` trusts the file header's fps for timing, a
  wrong header would make playback run too fast or too slow. The recorder
  therefore buffers frames with their capture timestamps and writes the file
  with the **measured** fps (`(n - 1) / elapsed`), not the camera's claim.
- **Mirroring**: the live preview is mirrored (selfie view — what people
  expect when watching themselves), but frames are **saved unmirrored**, the
  same orientation as WLASL's camera-facing-signer clips, so the renderer
  shows letters the same way it shows words.

### Session flow (`python -m pose_library.record_fingerspelling`)
One OpenCV window, keyboard-driven, no code-writing needed:
1. Shows the live camera with the current target letter large on screen.
2. **SPACE** starts a 3-second countdown (form and hold the handshape during
   it), then records for a fixed duration: **2.0s for the 24 static letters,
   2.5s for J and Z** (the two letters that are a motion, not a held shape —
   already flagged in `pose_library/PLAN.md` Section 5).
3. After capture it writes `pose_library/fingerspelling/raw/<letter>.mp4`
   (atomically — a temp file then rename, so an interrupted write never
   leaves a truncated video in place of a good one), then runs the real
   Holistic extraction on it and reports **"hand detected in N/M frames"**,
   so a bad take is caught immediately instead of at build time.
4. **R** re-records the same letter (overwrites), **N** / **P** move to the
   next / previous letter, **Q** or **ESC** quits. Starting a session resumes
   at the first letter with no recording yet; `--letters jz` targets specific
   letters for re-recording.

Signing guidance shown on screen and in Section 7: sign with one hand
(fingerspelling is one-handed), keep the other hand down and out of frame,
face the camera, hold static letters steady, and for J/Z start the motion
when recording begins.

---

## 2. Data format — justified against real pose JSON, not assumed

### Pose JSON: identical `PoseSequence` shape, unchanged
Checked real files (`about.json`, `come.json`): `{gloss, fps, landmark_names
(48), frames (N × 48 × [x,y,z]), source}` — the `PoseSequence.to_dict()`
shape. Letters reuse it **verbatim**: `extract_pose_sequence()` runs
unchanged (same 48 landmarks — 21 left hand, 21 right hand, 6 pose-subset
points), `gloss` is the uppercase letter (`"A"`), `source` is
`"fingerspelling:self-recorded"`. This is the load-bearing decision: because
the shape is identical, `fetchWordPoseResult` → `reconstructTimeline` →
`stitchTimelines` → `SkeletonCanvas` consume a letter with **zero changes**
to any of them. There is no second rendering path.

### What a letter's pose JSON actually needs to contain
A raw 2s capture of a *static* handshape is ~60 frames of the same pose.
Playing all of it per letter would take ~2.2s per letter (with the 200ms
transition) — a five-letter word would take 11s, far slower than any
readable fingerspelling. What a static letter actually needs is **a short,
real, stable hold**:

- **Static letters (24)**: the build keeps only the **most stable
  gap-free window of `LETTER_HOLD_SECONDS = 0.4s`** (12 frames at 30fps) —
  the run of consecutive kept frames with the least total hand motion. Real
  frames, not a synthesized or duplicated pose, so the hand keeps its natural
  micro-motion instead of looking frozen. Measured against real WLASL data to
  pick thresholds (not guessed): a near-still hand moves ~0.001–0.005
  normalized units/frame (p10–p50 across `about`/`bathroom`/`doctor`), a
  moving one ~0.02–0.05 (p90). With the 200ms transition, each letter takes
  ~0.6s ≈ 1.7 letters/second — deliberately slower than fluent
  fingerspelling (~4-6 letters/s), since this is a readability aid for an
  avatar; one named constant to tune later.
- **Motion letters (J, Z)**: the build keeps the **motion span** — first to
  last frame whose hand displacement exceeds `max(0.008, 25% of the clip's
  peak)`, padded by 2 frames each side — so the still lead-in/lead-out isn't
  played. Interior tracking drops inside the span stay honestly recorded
  (see below), exactly like WLASL words.
- **One signing hand**: fingerspelling is one-handed, but a resting second
  hand can still be detected in frame. If kept, it would pull the
  content-fit camera out to frame both hands (the same effect seen on the
  two-handed "angry" clip in `PROJECT_STATUS.md` Section 12). The build picks
  the hand detected in the most frames of the segment as the signing hand
  and zero-fills the other with the existing `(0,0,0)` "not detected"
  sentinel. That marks the hand as absent rather than making up new data,
  and the renderer already handles it.

### Manifest: WLASL's shape where the fields apply, WLASL-only fields dropped
`pose_library/fingerspelling/manifest.json`, keyed by lowercase letter:

```json
"a": {
  "filename": "a.json",
  "letter": "a",
  "kind": "static",
  "signing_hand": "right",
  "source": "fingerspelling:self-recorded",
  "license": "Self-recorded by the project author; project-owned, MIT (see LICENSE)",
  "recording": {"filename": "raw/a.mp4", "total_frames_decoded": 61,
                "frames_with_hand": 59, "segment_start_frame": 23},
  "total_frames_decoded": 12,
  "frames_kept": 12,
  "dropped_frame_indices": [],
  "low_confidence": false,
  "quality_notes": null
}
```

- **Kept, same meaning as WLASL**: `filename`, `license`,
  `total_frames_decoded`, `frames_kept`, `dropped_frame_indices`,
  `low_confidence`, `quality_notes`. Note that WLASL's own
  `total_frames_decoded` already describes the **trimmed clip** — the
  segment that was stored, not the full source video — so describing the
  stored letter segment the same way is the *same* semantics, not a
  reinterpretation. That's what lets `reconstructTimeline()` consume letter
  entries unchanged. For static letters `dropped_frame_indices` is always
  `[]` (the window is chosen gap-free by construction); it's kept anyway
  rather than special-cased out, because J/Z legitimately need it and one
  shape is simpler for every consumer than two.
- **Dropped (WLASL-only)**: `wlasl_source`, `wlasl_video_id`,
  `wlasl_instance_id` — there is no WLASL instance.
- **Added**: `letter`, `kind` (`static`/`motion`), `signing_hand`, and a
  `recording` block holding the raw-take provenance (whole-recording frame
  counts, where the segment starts), so the raw `.mp4` → stored segment
  relationship stays auditable.
- **Quality flags are letter-specific, not `compute_quality_flags()`**: that
  function's thresholds were tuned on WLASL word clips (e.g. `<25` kept
  frames = low confidence) and would flag every 12-frame static letter.
  Letters instead flag: hand detected in `<80%` of the recording; a static
  letter with no gap-free run as long as the hold window; a motion letter
  with no detectable motion.

### Storage and git
Per the explicit decision for this feature and `pose_library/PLAN.md`
Section 3's original recommendation: raw recordings
(`pose_library/fingerspelling/raw/*.mp4`) and extracted poses
(`pose_library/fingerspelling/poses/*.json` + `manifest.json`) are
**committed**, not gitignored — self-recorded, project-owned content, with
none of WLASL's licensing questions. `.gitignore`'s global `*.mp4` rule gets
one explicit negation for `pose_library/fingerspelling/raw/*.mp4`.
Expected size: roughly 0.3–1MB per clip. Now that 24 letters come from the
dataset (Section 2b), only the recorded letters (J, Z) have raw clips, so
~1–2MB, not the original 10–25MB estimate.

---

## 2b. Data source: the asl-now-fingerspelling dataset (2026-09-26)

**24 of the 26 letters now come from a real, MIT-licensed third-party
dataset, not from self-recording and not from synthetic placeholders**:
[`sid220/asl-now-fingerspelling`](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
on Hugging Face ("ASLNow!", by Sidney Trzepacz), pinned to revision
`9b3c96ae0adb7744a2c9fc72692842e6b3e25e33`. License record, with both
primary sources quoted verbatim:
[`fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`](fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md).
Converter: `python -m pose_library.convert_hf_fingerspelling`. **J and Z are
not available from this source** (see below), so they were self-recorded
instead (Section 2c).

### What the data actually is — inspected, not assumed
All 2,122 sample files at that revision were downloaded and read:
- **26 folders, one per letter**, 53–155 samples each (A 65, B 69, C 53,
  D 72, E 57, F 61, G 95, H 69, I 68, J 93, K 64, L 84, M 88, N 84, O 98,
  P 72, Q 97, R 75, S 83, T 68, U 106, V 83, W 75, X 106, Y 82, Z 155).
- **Every file is exactly one frame**: a JSON list of 21 `{x, y, z}` points,
  matching the dataset card. No file in any letter has a second frame, a
  timestamp, or any other field.
- **J and Z have no motion data.** They have 93 and 155 files, but each is
  the same single still frame as every other letter, a snapshot of a letter
  that is really a motion (PLAN.md Section 5). Holding one still frame, or
  interpolating between two unrelated ones, would show a motion that wasn't
  captured and misrepresent the sign. So **J and Z are not converted from
  the dataset**. They were self-recorded instead (Section 2c). Before that,
  words containing them were skipped with `NO FINGERSPELLING FOR "J"` via the
  existing missing-letter handling (Section 3), which still applies to any
  letter a checkout is missing.
- **No handedness label**, and mixed hands: a 2D palm-orientation test splits
  every letter roughly 60/40 (B: 52/17), and a 3D chirality measure
  (the palm-plane normal against the side the fingertips curl toward) gives
  per-letter splits that vary from 14/51 (A) to 50/47 (Q). Real
  left-handed participants would give a roughly constant ratio, so this
  can't be trusted as a label. Handedness is recorded as `"unlabeled"`
  rather than guessed.
- **41 of the 1,874 static-letter samples have a landmark outside the
  image** (hand partly out of frame, so MediaPipe extrapolated). These are
  excluded. Every letter keeps at least 48 usable samples.

### Compatibility with this project's pipeline — checked, not assumed
The dataset came from MediaPipe's **Web Hand Landmarker**, not the Python
**Holistic Landmarker** `extract.py` uses. Both emit the same 21-point hand
topology in the same index order (`extract.HAND_LANDMARK_NAMES`: wrist, thumb
CMC→tip, then index/middle/ring/pinky MCP→tip), with x/y normalized to the
image and z relative to the wrist. So the values are placed into the
48-landmark frame **unchanged**: in the right-hand slot (a convention only,
since the renderer draws both slots the same way), with the other hand and
the 6 pose-subset points as the `(0,0,0)` "not detected" sentinel, exactly
like a one-handed recorded letter. Verified on every converted letter: the
stored landmarks equal the source file's values exactly.

### Choosing one sample per letter: the medoid
Each letter uses its **medoid**, the real sample with the lowest *median*
shape distance to that letter's other usable samples. Shape distance is the
mean per-landmark 2D distance after anchoring the wrist at the origin and
scaling by wrist→middle-MCP length. Orientation is **kept**, because it's
part of some letters (G/H point sideways, P/Q point down), and z is dropped
(too loosely scaled to weigh equally). Why not the suggested centroid or
median pose: with mixed hands, a per-landmark average blends mirror images
into a hand nobody signed. The medoid is always one whole, real sample,
from the densest group, so an outlier can't win. The median rather than the sum
keeps a few extreme samples from skewing it. Deterministic, and independent
of file order (tested).

### Format of a converted letter
Identical to a recorded static letter (Section 2): `PoseSequence` JSON, 30
fps, the one chosen frame **held for 12 frames (0.4s, `LETTER_HOLD_SECONDS`)**.
A hold of a real pose, not a synthesized motion. Unlike a recorded hold it
has no natural micro-motion, since there is only one frame. `source` in the
pose JSON is `hf:sid220/asl-now-fingerspelling:<Letter>/<uuid>.json`, shown
in the app's source line. The manifest entry (`build_dataset_letter_entry`,
sharing `_letter_entry` with recorded letters) carries `source`,
`source_url`, `license`, and a `dataset_sample` block (dataset, revision,
exact file, selection method, median distance, sample counts). There is no
`recording` block, because nothing was recorded.

### Storage and precedence
- The raw download is a re-fetchable cache in the **gitignored**
  `pose_library/data/asl_now_fingerspelling/`. Only the 24 converted letters
  (~600KB) are committed. 11 real sample files are committed as test fixtures
  (`tests/fixtures/asl_now_sample/`, same MIT notice).
- **A recording always wins**: `build_fingerspelling` replaces a dataset
  letter when a recording of it exists (an explicit choice), and the converter
  never overwrites a self-recorded letter. Each tool only *removes* entries it
  owns, so a failed or absent recording never deletes a dataset letter.

---

## 2c. J and Z: self-recorded (2026-09-26)

J and Z were recorded with `python -m pose_library.record_fingerspelling
--letters jz` and built with `python -m pose_library.build_fingerspelling`,
exactly as Sections 1–2 describe. The 24 dataset letters were untouched
(verified: the manifest diff only adds `j` and `z`). Both are `motion`
letters with `source: "fingerspelling:self-recorded"` and a `recording`
block, and `raw/j.mp4`/`raw/z.mp4` are committed.

| Letter | Raw take | Hand tracked | Stored motion span | Flags |
|---|---|---|---|---|
| J | 76 frames, ~29.8fps | 76/76 | 30 frames (~1.0s), no gaps | none |
| Z | 76 frames, ~29.6fps | 76/76 | 52 frames (~1.8s), no gaps | none |

A sanity check against the letters' actual shapes, not just the numbers:
**J** is traced with the pinky, and its pinky tip has the longest path of
any fingertip (1.61 normalized units vs 0.83 for the index). **Z** is traced
with the index finger, and its index tip has the longest path (1.36 vs 1.03
for the pinky). A dataset static letter (A) moves 0 by comparison. That's
consistent with correct recordings. It doesn't replace a review by someone
who knows ASL fingerspelling (Section 7).

---

## 3. Integration point

### Serving
`vite.config.js`'s dev-server middleware (the existing `/poses/*` stopgap)
is generalized into one factory mounting two directories:
`/poses/*` → `pose_library/data/poses/` (unchanged) and
`/fingerspelling/*` → `pose_library/fingerspelling/poses/`. The letter
directory can be overridden with a `FINGERSPELLING_POSES_DIR` environment
variable — used only to point the dev server at the synthetic fixtures for
a manual browser check before real recordings exist. Same filename
allow-list and traversal guard as today.

### Lookup: the existing hook, twice — no new hook
`usePoseSequences` gains one optional argument, `basePath` (default
`"/poses"`), and nothing else changes. `App.jsx` calls it twice:
1. `usePoseSequences(submittedWords)` — unchanged word lookup.
2. `usePoseSequences(neededLetters, { basePath: "/fingerspelling" })` —
   where `neededLetters` is the **deduplicated** set of letters across every
   `not_found` word (spelling "banana" fetches `b`, `a`, `n` once each, not six
   times). Computed from the first batch's results, so it's naturally empty
   (and the hook idle) until the word batch resolves.

A new hook isn't needed: the only difference between a word lookup and a
letter lookup is the URL prefix.

### Expansion: a pure module, `frontend/src/lib/fingerspelling.js`
- `spellWord(word)` → `{ letters }` or `{ letters: null, reason }`. Letters
  `a–z` are spelled; apostrophes and hyphens are dropped (punctuation is out
  of scope, and they aren't fingerspelled in this v1); any other character
  (digits, symbols) makes the word unspellable (`reason: "unsupported"`);
  more than `MAX_FINGERSPELL_LETTERS = 20` letters is refused
  (`reason: "too_long"`). The 20-letter cap applies `CLAUDE.md`'s "length-limit
  all user input" rule per word, since a pasted 200-character token would
  otherwise queue ~2 minutes of letters.
- `planPlayback(wordResults, letterBatch)` → the ordered list of playable
  **units** handed to `stitchTimelines` (a library word is one unit; a
  fingerspelled word is one unit per letter, each shaped exactly like a word
  result), plus per-unit `{wordIndex, letterIndex}` bookkeeping for the
  caption, and the per-word display state.

### Failure handling — deliberate choices
- **A letter missing from the alphabet** (e.g. only 20 of 26 recorded): the
  **whole word is skipped**, with the missing letters named (`SKIPPED: "QUIZ"
  (NO FINGERSPELLING FOR "Q", "Z")`). Spelling it with the gap would show a
  *different* word ("UI" for "QUIZ"), which is worse than skipping.
- **The alphabet isn't available at all** (letter manifest 404, e.g. a
  checkout or config without the letter library): not an error that aborts
  the sentence. `not_found` words fall back to plain skipping, with the copy
  `FINGERSPELLING ALPHABET NOT AVAILABLE`. Library words still play.
- **A letter fetch fails with a real error** (HTTP 5xx, network): same rule
  as a word fetch error, per `MULTIWORD_PLAN.md` Section 4 — aborts the
  sequence, because a technical failure shouldn't be passed off as a
  vocabulary gap.
- **A `low_confidence` letter** plays, flagged in the same low-confidence
  banner as words (label `LETTER "Q"`).

---

## 4. Why `resolve_gloss_word()` is redesigned, not built as sketched

`pose_library/PLAN.md` Section 5 sketched a Python
`resolve_gloss_word(word) -> PoseSequence | list[PoseSequence]`, written
before any renderer existed. Since then, the actual consumer turned out to be
the frontend, which fetches static JSON and already has a sequence-stitching
pipeline that treats "a list of units" uniformly. Resolution therefore lives
where the stitching lives (`fingerspelling.js`), producing units in the shape
`stitchTimelines` already takes. Building the Python version now would be
unused code. If build-order step 4's live pipeline ends up resolving
server-side, a Python counterpart can mirror `spellWord`/`planPlayback`
then, with this doc's rules as the spec. `PLAN.md` Section 5 gets a pointer
to this document rather than being silently left stale.

---

## 5. UI treatment

Conventional gloss notation writes a fingerspelled word as hyphenated
letters (`J-O-H-N`), and that is the treatment here, using existing tokens
only (`CaptionBand.module.css`):

- **Chip (not playing)**: `B-A-N-A-N-A` in the mono label style, in
  `--color-accent-amber` with a dotted underline, so it is visibly "spelled,"
  not a library sign. Color is never the only signal: the hyphens and the
  underline carry the meaning too.
- **Active (playing)**: the display-size heading shows the letters, with
  the letter currently being signed at full `--color-text-primary` and the
  rest `--color-text-muted`. Viewers can follow it letter by letter, like
  the word-level highlight already does for sentences.
- **Accessibility**: the hyphenated letters are `aria-hidden`, and a
  visually-hidden `FINGERSPELLED: BANANA` text span carries the accessible
  name, so screen readers don't read "B dash A dash N…". (Implemented as
  hidden text rather than an `aria-label`, which isn't reliably announced
  on a plain `<span>`.)
- **Source line** reads the playing letter's provenance, in the same
  attribution slot WLASL words use:
  `SOURCE: hf:sid220/asl-now-fingerspelling:<Letter>/<uuid>.json` for a
  dataset letter, `SOURCE: fingerspelling:self-recorded` for a recorded one.
- **Status strip**: fingerspelled words count as playable, not skipped:
  `READY — 3/3 WORDS (1 FINGERSPELLED)`.

---

## 6. Explicitly out of scope for v1

- **Numbers/digits** — a word containing one is skipped
  (`reason: "unsupported"`), not spelled.
- **Punctuation** — apostrophes/hyphens are dropped, nothing else is signed.
- **Letter-to-letter transition tuning** — letters get exactly the same
  generic 200ms `stitchTimelines` transition as words; no letter-specific
  timing, no handshape-aware coarticulation.
- **Ambiguous/homonym handling** — no disambiguation; a missing word is
  spelled exactly as typed.
- **Double-letter conventions** (the small bounce/slide for "LL", "OO") —
  consecutive identical letters simply play twice.
- **Gloss prefixes** — `DESC-`/`X-` markers from `gloss_model` output are
  not stripped before spelling (a typed `desc-nice` would spell
  `D-E-S-C-N-I-C-E`); prefix handling belongs to build-order step 4's
  gloss→pose wiring, not this fallback.
- **Signer diversity / multiple takes per letter** — one take per letter.
- **A Python `resolve_gloss_word()`** — see Section 4.

---

## 7. Test bar and sign-off

### Automated (all required, all built in this pass)
- `tests/test_fingerspelling.py`: letter validation/normalization, path
  naming, static hold-window selection (picks the stillest gap-free run,
  never straddles a dropped frame, falls back and flags when no full-length
  run exists), motion-span selection (trims still lead-in/out, falls back
  when there's no motion), signing-hand selection and zero-fill.
- `tests/test_record_fingerspelling.py`: the camera-free logic — measured
  fps, atomic video writing (a real synthetic `.mp4` round-trips through
  OpenCV), letter-argument parsing, resume-at-first-missing-letter, per-letter
  durations.
- `tests/test_build_fingerspelling.py`: the build orchestration with
  MediaPipe mocked (the `test_pose_extract.py` pattern) — writes the pose
  JSON and manifest entry, records failures without raising, rebuilds the
  manifest from whatever letters exist.
- `tests/test_pose_manifest.py` extended: letter-entry shape and
  letter-specific quality flags.
- `frontend/src/lib/fingerspelling.test.js`: `spellWord` rules and
  `planPlayback` expansion (dedup, missing-letter skip, alphabet-unavailable
  fallback, low-confidence letters, unit bookkeeping).
- `frontend/src/hooks/usePoseSequences.test.js` extended: `basePath`.
- `frontend/src/App.test.jsx` extended: a fingerspelled word plays end to
  end on **synthetic** letter data (skeleton renders, spelled caption
  appears, status counts it as playable), a mixed library + spelled
  sentence, the alphabet-unavailable fallback, and a missing-letter skip.
- Full Python and frontend suites, `ruff`, `black`, `eslint`, `prettier`
  clean.

### Manual (real browser, per `CLAUDE.md`'s happy-dom lesson)
Point the dev server at the synthetic fixtures
(`FINGERSPELLING_POSES_DIR=../tests/fixtures/fingerspelling_synthetic`), spell
a word, and confirm by eye: letters play in order, the camera frames the one
synthetic hand, the active-letter highlight advances, no overflow at desktop
and phone sizes.

### Sign-off criteria
- **Infrastructure sign-off**: met (everything above passing).
- **Data sign-off (all 26 letters)**: met 2026-09-26. The 24 dataset
  letters are converted, sanity-checked (landmarks equal the source,
  non-zero, 12-frame holds), and viewed in a real browser (B, L, V, Y
  clearly legible; A and O correct but harder to read, as curled-finger
  letters are in a 2D skeleton). J and Z are recorded, built with full hand
  tracking and no flags, and play in the app ("jazz" spells end to end;
  Section 2c).
- **Fluent-signer review**: not done. No one who knows ASL fingerspelling
  has checked the letters yet; all 26 were checked by eye and by the
  measurements above only. This is the one remaining check for the feature.

### What remains
No data is missing. The remaining step is a review by someone who knows ASL
fingerspelling: spell several real words (including ones with J and Z) and
confirm they read correctly. Any letter that doesn't can be re-recorded with
`python -m pose_library.record_fingerspelling --letters <x>` followed by
`python -m pose_library.build_fingerspelling`. A recording replaces the
dataset version of that letter.

---

## Known gotchas / open questions (carried into implementation)

- **Webcam fps claims are unreliable** — hence measured-fps writing (Section 1).
- **`*.mp4` is globally gitignored** — the negation for
  `pose_library/fingerspelling/raw/*.mp4` must stay below the global rule in
  `.gitignore`, or it silently stops applying.
- **MediaPipe's left/right hand labels** come from the (unmirrored) image;
  `signing_hand` records whatever MediaPipe reported, which isn't
  necessarily the signer's anatomical dominant hand. Only used to decide
  which hand to keep, so the naming doesn't matter for playback.
- **The phantom-hand rendering characteristic** (`PROJECT_STATUS.md`
  Section 12) applies to letters too: the zero-filled non-signing hand is
  the same `(0,0,0)` sentinel one-handed WLASL words already have.
- **Hold duration is a readability guess, not user-tested** — 0.4s/letter
  (`LETTER_HOLD_SECONDS`) is the first thing to tune once real letters exist.
