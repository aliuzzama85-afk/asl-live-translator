# Multi-Word Playback Plan (v1): Cross-Sign Interpolation

Scope per `CLAUDE.md`'s build order, step 3: **"Interpolation/smoothing between
signs."** This is a pure playback/animation-stitching concern — it takes a
short, already-glossed sequence of words and plays them back as one
continuous skeleton animation instead of one word at a time. It is
**explicitly not**: ASR/VAD (step 4, still after this), gloss translation
(Stage 3, already done, a separate upstream concern), or ASL grammar/gloss
reordering (a real NLP problem, not a rendering one — see Section 6). This
document is a plan only, matching `frontend/PLAN.md`'s level of detail — no
components are built or modified yet.

Baseline checked directly before writing this (not assumed): `CLAUDE.md`,
`PROJECT_STATUS.md` Section 11, `frontend/PLAN.md` in full,
`frontend/src/lib/reconstructTimeline.js`,
`frontend/src/hooks/usePoseSequence.js`, `frontend/src/App.jsx`,
`frontend/src/components/SkeletonCanvas.jsx`, and
`frontend/src/components/CaptionBand.jsx` — plus two direct data checks
against the real 118-word library on disk (`pose_library/data/poses/`),
detailed in Section 3, since PLAN.md's own convention is to check real data
rather than assume before designing around it.

---

## 1. Input handling

**Decision: a single text input, multiple words separated by spaces —
reusing `CaptionBand`'s existing search input, not a new textarea or a
one-word-at-a-time queue UI.**

Concretely: `CaptionBand.handleSubmit` already does
`inputValue.trim().toLowerCase()` and hands one normalized string to
`onSearch`. The only change at the input layer is splitting that normalized
string on whitespace (`/\s+/`) into an array of words before it leaves
`CaptionBand`, dropping empty segments from repeated spaces. Everything else
about the input (the existing `<input>`/`<label>`/focus-ring/ARIA setup) is
untouched.

**Why this over the alternatives**:
- **A textarea** implies multi-line sentences or paragraph input, which
  overstates what this stage does — it's still "fed gloss manually" per
  `CLAUDE.md`'s build-order convention, just multiple gloss tokens instead of
  one. A single-line input matches that scope honestly.
- **A queue UI** (add one word, see it appended to a list, submit when done)
  is a heavier interaction pattern (needs its own remove/reorder/clear
  controls, its own accessibility pass) for a v1 whose actual job is proving
  cross-word interpolation works, not building a sequence editor. Nothing
  here precludes adding one later if typing space-separated words in one box
  turns out to be too fiddly in practice.
- Space-separated matches the shape gloss output already has: `gloss_model`
  produces space-separated uppercase tokens (e.g. `WEATHER BE DESC-NICE
  TODAY`). Treating the search box as "paste/type a short gloss sequence" is
  the same mental model the eventual live-pipeline wiring (build-order step
  4) will actually feed it with — no separate parsing convention to
  reconcile later.

**Concrete parsing rules**:
- Trim, then split on `/\s+/`; drop empty segments.
- Case-insensitive (existing lowercase-normalize behavior, unchanged).
- Duplicate consecutive words are valid input, not deduped — ASL repetition
  (e.g. signing a word twice for emphasis) is a real, legitimate case; this
  stage has no business silently dropping a word the user typed twice.
- **Length cap: 20 words.** Per `CLAUDE.md`'s security rule ("sanitize and
  length-limit all user text/audio input"), applied at this layer the same
  way `gloss_model/inference.py`'s `MAX_INPUT_CHARS=500` already applies it
  upstream — this isn't a live/network-facing endpoint yet, but the
  discipline should already be in place before it becomes one. 20 is
  generous for a "short sentence or word list" (CLAUDE.md's own framing of
  this feature) while bounding worst-case fetch count and total stitched
  duration. Words beyond the cap are dropped with a visible message (see
  Section 5), not silently truncated.

---

## 2. Data flow: hook design

### Current contract (unchanged, not modified by this plan)

`usePoseSequence(word)` takes one string, fetches `/poses/manifest.json` and
`/poses/<word>.json` in parallel, and returns one
`{status, word, sequence, manifestEntry, fetchMs}`-shaped discriminated
union (`idle`/`loading`/`ok`/`low_confidence`/`not_found`/`error`). This stays
exactly as it is — no changes to its signature, behavior, or tests.

### New hook: `usePoseSequences(words: string[])`

A **new** hook, not a variadic mutation of `usePoseSequence` — React's rules
of hooks forbid calling `usePoseSequence` in a loop over a dynamic-length
array, so this has to be its own implementation. To avoid duplicating the
fetch logic (per `CLAUDE.md`'s "extend/reuse over duplicating... flag
near-duplicate logic" rule), the actual per-word fetch body in
`usePoseSequence.js` should be factored out into a shared, exported helper —
e.g. `fetchWordPoseResult(word, manifest)` — that both hooks call. This is a
small, mechanical extraction of existing logic, not new logic, and it means
the not-found/error-handling rules only exist in one place.

**Concrete efficiency improvement over calling the single-word logic N
times**: fetch `/poses/manifest.json` **once** per submitted sequence, not
once per word. Today, one `usePoseSequence` call fetches the manifest and one
word's JSON together; for a 5-word sequence, calling that pattern 5 times
independently would re-fetch the same manifest 5 times. `usePoseSequences`
fetches the manifest once, then fetches each word's own `<word>.json` in
parallel (`Promise.all`) and resolves each into the same
`fetchWordPoseResult`-shaped per-word result using that one shared manifest.

**Return shape**:
```
{
  status: "idle" | "loading" | "ready" | "error",
  results: PoseSequenceResult[],   // one per input word, same union usePoseSequence already returns
  fetchMs: number | null,
}
```
`status: "error"` here means the *manifest itself* failed to fetch (a
sequence-level failure, distinct from one word's own `not_found`/`error`,
which lives inside that word's entry in `results`) — see Section 4 for why
that distinction matters.

### What happens to `App.jsx` and `usePoseSequence.js`

**`App.jsx` generalizes to always treat its input as a sequence** — a single
typed word becomes a sequence of length 1, not a separately maintained code
path. This means `App.jsx` switches from calling `usePoseSequence` to calling
`usePoseSequences`, and the single-word-shaped props it threads into
`SkeletonCanvas`/`CaptionBand` today get replaced by the stitched-timeline
shape from Section 3.

**`usePoseSequence.js` itself is not deleted or modified in this pass.** It
stays in place, still independently usable and tested, sharing its fetch
body with the new hook via the extracted helper above. Retiring it entirely
(no remaining callers) is a reasonable follow-up cleanup once multi-word
mode is proven out, but doing that in the same pass as introducing
multi-word playback is more surgery on signed-off Stage 5 code than this
design calls for. **Flagging this explicitly now** (per `CLAUDE.md`'s
"don't just leave it" instruction) rather than silently letting it become
unreferenced dead code without a note.

---

## 3. Transition / interpolation approach

### What "end pose of word A" / "start pose of word B" actually means — checked against real data, not assumed

Read real pose JSON and manifest entries directly (10-word sample:
`about`, `bathroom`, `phone`, `doctor`, `car`, `buy`, `come`, `cost`,
`chair`, `angry`) rather than assuming signs converge on a shared neutral
position. Two concrete findings:

**Finding 1 — signs do not reliably start or end at a common rest
position, and are frequently one-handed.** Sample of first/last-frame wrist
positions:

| Word | First frame active wrist | Last frame active wrist |
|---|---|---|
| `about` | left `(0.616, 0.884)` | left `(0.618, 0.885)` |
| `bathroom` | right `(0.536, 0.829)` | right `(0.538, 0.813)` |
| `phone` | right `(0.282, 0.583)` | right `(0.335, 0.435)` — moves **up** ~15% of frame height |
| `doctor` | right `(0.414, 0.687)` | right `(0.384, 0.876)` — moves **down** ~19% |
| `come` | **left** active, right `(0,0,0)` | left `(0,0,0)`, **right** active |

`phone` and `doctor` alone rule out any assumption of a shared rest pose —
one sign ends near the signer's head, another ends near the waist. `come` is
more extreme still: which *hand* is even tracked changes between the start
and end of the same single sign (plausibly a two-handed sign where one hand
is briefly occluded at one end). Most sampled words are one-handed for their
entire duration, with the unused hand's 21 landmarks reading as the exact
`(0, 0, 0)` sentinel for every frame — not a per-frame gap like
`reconstructTimeline.js`'s within-word dropped frames, but a whole-hand,
whole-sequence absence.

**Decision this drives**: v1 does **not** synthesize a neutral/rest pose to
transition through. There is no such pose already latent in the data to
reuse, and inventing one (e.g. "both hands at chest height, relaxed") would
be new motion-design work — plausible, but a bigger scope than "stitch
existing sequences together" calls for, and explicitly out of scope for v1
(Section 6). Instead: **transitions are a direct interpolation from word A's
last real frame straight to word B's first real frame**, accepting that some
transitions will sweep across a large part of the frame (e.g. `phone` into a
sign that starts low) rather than looking like a relaxed pause. This is a
known, stated v1 limitation (Section "Known gotchas"), not a hidden one.

**Finding 2 — real per-word `fps` varies meaningfully, it is not just
float-precision jitter around 30.** Read directly off 10 real pose JSON
files:

| Word | `fps` |
|---|---|
| `about` | 30.0026 |
| `angry` | 30.0029 |
| `bathroom` | 29.9700 |
| `doctor` | 29.97 |
| `car` | 29.9700 |
| `buy` | 29.9700 |
| `come` | 29.9700 |
| `phone` | **25** |
| `chair` | **25** |
| `cost` | **24** |

Three distinct real frame rates in a 10-word sample (24/25/~29.97/~30.003),
not a single shared value. This matters because of a fact confirmed by
reading `SkeletonCanvas.jsx`'s playback loop directly: it advances through
`timeline.frames` using **one single `timeline.frameDurationMs`
(`1000/fps`) for the entire `Timeline` object** — there is no per-frame
timing field it consults (`TimelineFrame.time` exists in the data shape but
the rAF loop never reads it; it steps by a fixed accumulator threshold
instead). A merged multi-word `Timeline` therefore **cannot** simply
concatenate each word's own native-fps frame list — `SkeletonCanvas` has no
mechanism to change its step rate mid-playback.

**Decision this drives**: before stitching, resample every word's
already-gap-filled per-word timeline (the existing
`reconstructTimeline()` output) onto **one fixed target render rate — 30fps**
— chosen because it's close to most sampled words' native rate (minimizing
resampling distortion for the common case) and is a normal video/animation
rate. Resampling reuses the exact same lerp mechanism `reconstructTimeline.js`
already has (see below), just walking output time `k/30` instead of walking
original dropped-frame indices.

### Reusing, not reinventing, the existing lerp/gap-smoothing logic

`reconstructTimeline.js`'s `lerpPose(poseA, poseB, t)` already does the one
thing this needs: per-landmark linear interpolation that leaves the
`(0,0,0)` "not detected" sentinel untouched rather than lerping through the
canvas origin corner (see its own docstring — this exact behavior is why
`come`'s hand-switch above won't produce a fake sliding motion through
`(0,0,0)` at a word boundary, the same guard that already protects
within-word gaps). **This must be reused verbatim, not reimplemented** — it
is currently a private, unexported function in `reconstructTimeline.js`; the
one small, mechanical change needed there is exporting it (alongside the
already-exported `reconstructTimeline` and
`GAP_INTERPOLATION_THRESHOLD_FRAMES`) so a new sibling module can import it.

**New module: `frontend/src/lib/stitchTimelines.js`**, exporting a function
along these lines:

```
stitchTimelines(
  perWordResults: Array<{ word, sequence, manifestEntry }>,  // only ok/low_confidence entries, see Section 4
  { targetFps = 30, transitionMs = 200 } = {}
): { timeline: Timeline, wordBoundaries: WordBoundary[] }
```

Steps, per word, in order:
1. Run the word's `(sequence, manifestEntry)` through the existing
   `reconstructTimeline()` unchanged — this still does its own job (within-
   word gap reconstruction) exactly as it does today for single-word
   playback.
2. **Resample** that word's dense-but-native-fps frame list onto the fixed
   `targetFps` timeline, using `lerpPose` between the two surrounding
   original-rate frames at each new output timestamp (the same interpolation
   primitive, applied against a fixed output grid instead of the original
   video's own frame positions).
3. If this isn't the first word, **insert `transitionMs` worth of
   `lerpPose`-interpolated frames** (at `targetFps`) between the previous
   word's last resampled frame and this word's first resampled frame. This
   *adds* playback time at each boundary — it does not overlap or trim
   existing frames, keeping the mechanism identical in spirit to
   `reconstructTimeline.js`'s own gap-fill (new frames inserted between two
   known real poses), just applied across a word boundary instead of within
   one word's own dropped-frame gaps.
4. Record a `wordBoundaries` entry — `{ word, wordIndex, startFrameIndex,
   endFrameIndex }` — for this word's span in the final merged frame list,
   used by the UI (Section 5) to know which word is "currently playing"
   without `SkeletonCanvas` itself needing any concept of "words."

**`transitionMs = 200`**: not an arbitrary new number — chosen to match the
same order of magnitude as `reconstructTimeline.js`'s own
`GAP_INTERPOLATION_THRESHOLD_FRAMES` (~200ms at a 30fps source), for
consistency between "how long a smoothed gap looks" within a word and
between words, rather than introducing a second, differently-tuned notion of
"how long a smooth transition should take."

**Output shape**: the merged `timeline` is the **exact same `Timeline`
shape** `reconstructTimeline()` already produces
(`{fps, frameDurationMs, durationSeconds, frames}`, with `frames` being
plain `TimelineFrame` objects). This is the load-bearing design choice of
this whole plan: **`SkeletonCanvas` needs zero changes to its core playback
loop** — it just plays a longer `Timeline` than it used to, with some frames
tagged `state: "interpolated"` at word boundaries the same way within-word
gap-fill frames already are tagged today.

One necessary, small addition to `SkeletonCanvas.jsx`: an optional
`onFrameChange(frameIndex)` callback, fired once per tick alongside the
existing `onEnded` callback, reporting the currently-playing frame index.
This is the **only** change to `SkeletonCanvas.jsx` this plan requires — it
keeps the component itself word-agnostic (it reports a frame index, not a
word), and lets `App.jsx` do the frameIndex → word lookup via
`wordBoundaries` to drive the "now playing" UI in Section 5.

---

## 4. Handling `low_confidence` / `not_found` / `error` mid-sequence

**Decision: `not_found` words are skipped (contribute zero frames, listed
visibly); `low_confidence` words still play (existing single-word
precedent); a manifest-level `error` aborts the whole sequence.** Not one
uniform policy — the three cases are different in kind and justify different
handling:

- **`ok`**: plays normally, included in `stitchTimelines`'s input.
- **`low_confidence`**: **plays normally**, same as single-word mode today
  (the data is real, possibly still recognizable — recall `PLAN.md`
  Section 2's existing "the banner is the signal; playback is not gated on
  it" precedent for single-word mode). This carries over unchanged: a
  degraded word is still worth showing, just flagged.
- **`not_found`**: **skipped, not aborted.** In single-word mode, a
  not-found word takes over the *entire* Stage with a message because
  there's nothing else to show. In a multi-word sequence, that's no longer
  true — there are other real words that *can* play, and per `CLAUDE.md`'s
  own framing ("the avatar always stays a phrase or two behind... like a
  human interpreter"), a human interpreter who hits an unknown word doesn't
  freeze the whole sentence — they skip it (or fingerspell, not available
  yet — see Section 6) and continue. A skipped word contributes **zero
  frames** to the stitched timeline (no held/dimmed placeholder eating real
  playback time for a word with no data at all) but is **listed visibly**
  in a persistent caption-band note (e.g. `SKIPPED: "XYZ" (NOT FOUND)`),
  never silently dropped — consistent with `CLAUDE.md`'s "never a silent
  frozen avatar" rule applied at the sequence level, not just the frame
  level. Because `stitchTimelines` (Section 3) only ever receives the
  filtered `ok`/`low_confidence` entries — `not_found` words are removed
  from the sequence *before* stitching, not passed through and special-cased
  inside it — a `not_found` word never needs its own transition handling at
  all: the real words on either side of it simply become adjacent to each
  other in the filtered list and receive one ordinary `transitionMs` blend
  between them, indistinguishable from any other adjacent pair.
- **`error`** (a fetch/network failure on one word's own JSON, distinct from
  a manifest-level fetch failure): **aborts the whole sequence**, reusing the
  existing single-word error message pattern. This is deliberately different
  from `not_found`: a fetch error is a transient/technical failure, not a
  vocabulary gap, and silently skipping it the same way as `not_found` would
  misrepresent a real bug or network problem as "this word just isn't in the
  library" — the user needs to know something actually broke, not that one
  word was OOV.
- **Manifest-level `error`** (from `usePoseSequences`'s own `status:
  "error"`, Section 2): also aborts, same reasoning, one level up — if the
  manifest itself can't be fetched, no word in the sequence can be resolved
  at all, so there is nothing partial to play.

This asymmetry (skip `not_found`, abort on `error`) is a deliberate choice,
not an oversight — flagged explicitly here so it isn't "fixed" into
uniform handling later without re-reading this reasoning.

**If every word in the sequence is skipped or not found**: falls back to
the existing single-word "no sign found" full-stage message pattern (there's
nothing to play, so there's nothing multi-word-specific left to show).

---

## 5. UI implications

- **`StatusStrip`**: needs a new composite state beyond the existing
  `idle`/`loading`/`ready`/`low_confidence`/`not_found`/`error` set — e.g.
  `ready` gains a count, `● READY — 4/5 WORDS (1 SKIPPED)`, reusing the
  existing LED-dot-plus-label visual pattern (Section 0 design tokens) rather
  than inventing new status chrome. No new color tokens needed — skipped
  words use the same `--color-signal-error` amber/red-family semantics the
  single-word `not_found` state already uses.
- **`CaptionBand`**: today shows one `glossWord`. This becomes **the full
  typed sequence, rendered as a row of words, with the currently-playing
  word visually distinguished** (bold/full-brightness in
  `--color-text-primary`, others dimmed toward `--color-text-muted`) —
  driven by the `wordBoundaries` lookup from Section 3's `onFrameChange`
  callback. This is a natural fit for `CLAUDE.md`'s "live captioning
  overlay... like Otter.ai's real-time transcript view" direction: real
  closed captions commonly highlight the currently-spoken word within a
  visible line, and this is the same pattern applied to gloss words instead
  of speech transcript.
- **Word-search input**: accepts space-separated words per Section 1;
  existing input/label/focus-ring/ARIA markup is otherwise unchanged.
- **Low-confidence / not-found messaging**: the existing single-word
  `not_found` banner **replaces the entire Stage** with a message — that
  pattern is wrong for multi-word mode (Section 4) and must change to a
  smaller, persistent banner (caption band or status strip, `role="status"
  aria-live="polite"`, matching the existing accessibility pattern) that
  lists skipped words **without** hiding the skeleton stage, since the
  skeleton is still actively playing the words that *did* resolve.
  Low-confidence banners work the same way, just potentially listing more
  than one degraded word (`LOW-CONFIDENCE: "PHONE", "DOCTOR"`) instead of
  always exactly one.
- **Accessibility**: every rule from `PLAN.md`'s existing accessibility pass
  (contrast, 16px/13px type-size floor, color-independent status, keyboard
  navigation, `aria-live` banners, `prefers-reduced-motion`) carries over
  unchanged and applies to the new elements too — this is not a new
  accessibility pass from scratch, just the existing one extended to cover
  the word-progress row and the skipped/low-confidence list.

---

## 6. Explicitly out of scope for v1

- **ASL grammar / gloss reordering / topic-comment structuring.** This is a
  real, substantial NLP problem (word order, non-manual grammar, classifier
  constructions) — completely separate from this stage's job, which is
  purely "stitch whatever words are given, in the order given, into one
  animation." Whatever order the user types (or, eventually, whatever order
  `gloss_model` produces) is played verbatim.
- **Live ASR/VAD input** — build-order step 4, still after this step.
- **Fingerspelling fallback for skipped words** — `pose_library/PLAN.md`
  Section 5's fingerspelling design still isn't built (per
  `PROJECT_STATUS.md` Section 5 item 7); a skipped word stays skipped, not
  fingerspelled, until that lands separately.
- **Variable, linguistically-informed transition duration** (e.g. a shorter
  blend between closely related signs, a longer pause at a sentence
  boundary). v1 uses one fixed `transitionMs` for every boundary,
  unconditionally.
- **Non-manual markers / facial or body coarticulation across signs** — the
  pose library only has 42 hand landmarks + 6 pose-subset points; there is
  no facial data to blend even if this were in scope.
- **Scrubbing/seeking within a multi-word sequence** — no scrub bar existed
  for single-word playback either (`PLAN.md` Section 3); this doesn't
  introduce one.
- **Editing a sequence mid-playback** (inserting/removing a word without a
  full resubmit) — each submit is a fresh, complete lookup+stitch, same
  single-shot submit model the app already has today.
- **Caching/prefetching pose data across separate sequence submissions** —
  every submit re-fetches, same as today's single-word behavior; no new
  caching layer introduced.

---

## 7. Test bar — what "done" looks like

Per `CLAUDE.md`'s "don't move to the next pipeline stage until the current
stage's tests pass" rule, this feature is signed off when:

- **`stitchTimelines.test.js`** (new): given known, hand-constructed
  multi-word input (2-3 short synthetic sequences, not fetched from the real
  library), asserts: correct total frame count/duration for a known
  `targetFps`/`transitionMs`; the output `fps` matches `targetFps` exactly
  (single shared rate, not the inputs' native rates); a transition span
  exists at each word boundary with the expected frame count; the
  `(0,0,0)` sentinel is preserved (not lerped through) across a boundary
  where one word's sequence has a fully-absent hand and the adjacent word's
  sequence has that hand present; `wordBoundaries` entries have correct,
  non-overlapping `startFrameIndex`/`endFrameIndex` spans covering the whole
  merged timeline.
- **`usePoseSequences.test.js`** (new): asserts the manifest is fetched
  exactly once regardless of sequence length (mocked `fetch`, assert call
  count); returns one discriminated-union entry per input word; a
  `not_found` word's entry doesn't block the other words' entries from
  resolving; a manifest-fetch failure produces `status: "error"` at the
  sequence level, not per-word.
- **`App.test.jsx` extended** (per the existing real-manifest-word pattern
  from `a024993`, not synthetic fixtures): a real 3-4 word sequence queried
  against the actual 118-word manifest, including at least one real
  `low_confidence` word and one guaranteed-not-in-manifest word placed in
  the *middle* of the sequence (not just at an edge) — asserts the skipped
  word is visibly listed, the sequence still plays, and the other real words
  render correctly in the caption row.
- **Manual check**: play a real multi-word sequence in the running app and
  confirm by eye there's no hard jump-cut at any word boundary (even an
  "ugly," large-sweep transition like `phone` → a low-starting word counts
  as passing — smoothness, not gracefulness, is the bar per Section 3's
  documented limitation), and that a skipped mid-sequence word doesn't
  freeze or blank the stage.
- `ruff`/`eslint`+`prettier` clean, per `CLAUDE.md`'s existing convention,
  on every new/changed file.

---

## Known gotchas / open questions (carried into implementation)

- **Real per-word `fps` varies meaningfully (24/25/~29.97/~30.003 confirmed
  across a 10-word sample) — resampling to one fixed target rate before
  stitching is not optional**, because `SkeletonCanvas.jsx`'s playback loop
  (read directly) steps the entire `Timeline` at one shared
  `frameDurationMs` and has no per-frame timing mechanism.
- **Signs are frequently one-handed, with the unused hand reading as an
  exact `(0,0,0)` sentinel for every frame of that word** — confirmed
  directly against real pose JSON, not assumed. The existing
  zero-point-aware `lerpPose` already handles this correctly and must be
  reused (exported, not reimplemented) for cross-word transitions, or a
  word boundary between a one-handed and two-handed sign would produce a
  fake sliding motion through the canvas origin corner.
- **What a zero-sentinel hand actually looks like across a transition,
  confirmed by reading `lerpPose` and `SkeletonCanvas.jsx`'s draw loop
  directly, not assumed: it is not invisible.** `lerpPose` checks each
  landmark independently and forces the interpolated point to `(0,0,0)`
  whenever *either* endpoint is `(0,0,0)` — so if a hand is zero for all of
  word A and real/tracked in word B, every transition frame's `t` value
  still yields exactly `(0,0,0)` for that hand (the guard applies across
  the *whole* span, not just at `t=0`). But `SkeletonCanvas.jsx`'s draw loop
  only skip-draws a zero point for the 6 pose-subset landmarks
  (`bone.isPoseSubset`/`isPoseSubsetByIndex[i]`, per `skeletonBones.js`) —
  it has no equivalent check for hand landmarks. So that hand's joints and
  bones **do render**, as a static, motionless cluster sitting at wherever
  normalized source-coordinate `(0,0)` projects to under the current camera
  transform, for the entire `transitionMs` span. The instant playback
  crosses into word B's own resampled frames, that hand's real coordinates
  appear — a hard, un-eased pop-in jump from the fixed phantom position to
  word B's actual starting hand position, not a fade-in. This is not a new
  bug introduced by stitching: the identical phantom-point rendering
  already happens throughout a one-handed word's own single-word playback
  today (Stage 5, already shipped) — stitching just means it's also present,
  unchanged, during the inserted transition frames. Known, understood v1
  visual characteristic, same category as the "large mechanical sweep"
  limitation above, not something this plan fixes.
- **No shared neutral/rest pose exists in the data** (confirmed via
  first/last-frame wrist positions varying by up to ~20% of frame height
  across sampled words) — v1 deliberately does not synthesize one; some
  transitions will look like large, mechanical sweeps rather than a relaxed
  pause. Documented limitation, not a bug to silently "fix" by inventing
  motion data that doesn't exist.
- **`lerpPose` is currently private/unexported in `reconstructTimeline.js`**
  — needs exporting (a small, mechanical change) so `stitchTimelines.js` can
  reuse it rather than reimplementing the same zero-point-aware lerp logic a
  second time.
- **`SkeletonCanvas.jsx` needs exactly one new optional prop**
  (`onFrameChange`) and nothing else — it stays word-agnostic, reporting
  frame indices only; all "which word is this" logic lives in `App.jsx` via
  `wordBoundaries`.
- **`usePoseSequence.js` is not retired in this pass** — it stays in place,
  still tested and independently usable, sharing its fetch body with the new
  `usePoseSequences` hook via an extracted helper. Full retirement (once
  nothing calls it) is a flagged follow-up, not silently left unaddressed.
- **Manifest is still fetched fresh on every new sequence submission** — no
  caching layer existed before this feature and none is introduced by it
  (Section 6); each submit is a clean, independent lookup.
