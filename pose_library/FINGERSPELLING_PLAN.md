# Fingerspelling Alphabet Plan (v1): Self-Recorded A–Z Fallback

Scope: when a gloss word isn't in the 118-word WLASL pose library, spell it out
letter by letter with a self-recorded 26-letter ASL fingerspelling alphabet,
instead of skipping it with "NOT FOUND". This supersedes `pose_library/PLAN.md`
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

**Data status up front, stated plainly**: the 26 letters are **not recorded
yet**. Recording them requires a person signing on camera, which cannot be
automated. Everything else — recording tool, extraction, manifest build,
frontend integration, UI, tests — is built and verified against **synthetic
placeholder data** (clearly labeled, living under `tests/fixtures/`, never
under `pose_library/fingerspelling/`), so the pipeline is ready the moment
real recordings exist. See Section 7 for the exact manual steps.

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
Expected size: roughly 0.3–1MB per clip (26 clips, a few seconds each), on
the order of 10–25MB total — acceptable for a one-time asset; revisit
(e.g. Git LFS) only if re-recordings accumulate history bloat.

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
- **The alphabet isn't built at all** (letter manifest 404 — the actual
  state until recordings exist): not an error that aborts the sentence.
  `not_found` words fall back to exactly today's skip behavior, with honest
  copy: `FINGERSPELLING ALPHABET NOT RECORDED YET`. Library words still play.
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
- **Accessibility**: the chip/heading carries an `aria-label` of
  `"FINGERSPELLED: BANANA"` (screen readers shouldn't read "B dash A dash
  N…"); the per-letter spans are `aria-hidden`.
- **Source line** reads `SOURCE: fingerspelling:self-recorded` while a
  letter plays, the same attribution slot WLASL words use.
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
- **Infrastructure sign-off (this pass)**: everything above passing.
- **Feature sign-off (after recording)**: all 26 letters recorded and built,
  `manifest.json` has 26 entries, nothing unexpected flagged
  `low_confidence`, and a manual check of several real spelled words (at least
  one containing J or Z) reads correctly **to someone who knows ASL
  fingerspelling**. The synthetic data only proves the plumbing, not the
  handshapes.

### The manual step that remains
1. `python -m pose_library.record_fingerspelling` — sign all 26 letters.
2. `python -m pose_library.build_fingerspelling` — extract every letter and
   write `poses/*.json` + `manifest.json`.
3. Restart `npm run dev`, then spell a few words and review.
4. Commit `pose_library/fingerspelling/`.

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
