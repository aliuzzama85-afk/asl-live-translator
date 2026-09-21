# Stage 5 Plan (v1): 2D Skeleton Renderer & App Shell

Scope per `CLAUDE.md`'s build order, step 2: **"Pose library + static skeleton
renderer, fed gloss manually."** No live ASR/VAD wiring (that's step 4), no
sentence/multi-word stitching, no 3D avatar (an explicit v2 stretch goal, only
after this works). This document is a plan only — no components are built yet.

Current `frontend/` state checked before writing this: a bare Vite + React 18
scaffold (`package.json` has `react`, `react-dom`, `three` — `three` is
pre-installed for the *future* v2 stretch goal, not used in v1 — plus
`eslint`/`prettier`, no `src/` yet, no CSS framework, no CSS-in-JS library).
Everything below is planned against that real, empty-of-app-code baseline.

---

## 0. Design direction — committed, not a menu

### The aesthetic: **Broadcast Caption Console**

One specific, real reference point: the visual language of **EIA-608/708
broadcast closed captioning** (the black-box, high-contrast caption style used
on live TV) crossed with a **technical monitoring console / waveform-monitor
HUD** (the kind of interface a broadcast engineer or teleprompter operator
looks at, not a general audience). Concretely:

- A near-black stage that the caption text and skeleton sit on top of, exactly
  like a captioning box — not a "dark mode" reskin of a light dashboard.
- The current gloss word rendered **uppercase**, large, in a caption-style
  band pinned to the bottom of the viewport — this isn't just an aesthetic
  choice, it's a fortunate alignment with reality: ASL gloss is already
  conventionally written uppercase (see `gloss_model` output), so "look like a
  caption" and "show the actual data format" are the same thing.
- A thin top status strip styled like console telemetry (small-caps
  monospace labels, a single square LED-style status dot), not a hamburger
  menu or nav bar — there is no navigation in this app, so there should be no
  chrome pretending there is.

**Why this and not generic AI-SaaS default**: it has no centered hero, no
card grid, no rounded-corner-everything, no indigo/violet gradient, and no
default UI-framework typeface. It comes from a real, nameable interface
category (broadcast captioning hardware/software, e.g. CEA-608 caption
encoders and NLE caption-preview panels) rather than a mood board adjective.
It is also literally what this product is: a captioning tool. Leaning into
that is more honest than styling it like a generic AI dashboard.

**Why it survives the accessibility bar**: broadcast captioning standards
(FCC/CVAA guidance on closed captions) independently converged on the same
thing accessibility guidelines want — flat, very high-contrast color blocks,
no text-over-photo, no low-contrast decorative overlays, large sans-serif
text. Adopting that aesthetic isn't in tension with accessibility, it's
*drawing from a design tradition whose entire job is accessibility under
real-time constraints.* Every color/type choice below is checked against
WCAG contrast math, not just picked to look moody.

### Typography

Two real, specific, named, open-license typefaces (both SIL OFL, both
self-hostable — no Google Fonts CDN `<link>`, see Section 7 rationale):

- **Display / caption text — [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk), weight 700.**
  A geometric grotesk with distinctive quirks (the squared-off, slightly
  mechanical curves) — reads as considered and specific, not a default UI
  font, at very large sizes where a plain grotesk would look generic. Used
  *only* for the one thing that most needs to be readable at speed: the
  current gloss word.
- **UI / telemetry text — [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono), weights 500/600.**
  A monospace family literally designed by IBM for technical/product
  interfaces — reinforces the "console," not "dashboard," feel for labels,
  status text, timestamps, and the latency indicator. Monospace also means
  numeric latency values don't jitter in width as digits change, which
  matters for something updating in real time.
- Explicitly **not** Inter, Roboto, system-ui, or any variable-font default —
  none of those are used anywhere in the token set below.

Type scale (all sizes in `rem`, 1rem = 16px base, so nothing renders below a
comfortable reading size):

| Token | Font | Weight | Size | Line-height | Letter-spacing | Use |
|---|---|---|---|---|---|---|
| `--font-display` | Space Grotesk | 700 | `clamp(2rem, 5vw, 4rem)` | 1.1 | -0.01em | current gloss word |
| `--font-heading` | Space Grotesk | 700 | 1.25rem (20px) | 1.3 | 0 | section labels (e.g. "PLAYBACK") |
| `--font-body` | IBM Plex Mono | 500 | 1rem (16px) | 1.5 | 0 | help text, word-search input |
| `--font-label` | IBM Plex Mono | 600 | 0.8125rem (13px) | 1.4 | 0.06em, uppercase | control labels, status strip |
| `--font-mono-data` | IBM Plex Mono | 500 | 0.875rem (14px) | 1.4 | 0 | latency ms, frame counters |

13px is the smallest text anywhere in the UI, used only for short, uppercase,
high-contrast labels next to their controls, never for prose the user must
read continuously — kept intentionally above the 12px floor some interfaces
use for "fine print," per the hard readability requirement.

### Color palette (real hex values, contrast-checked)

Computed against WCAG 2.1 relative luminance (not eyeballed):

| Token | Hex | Role | Contrast on `--color-bg` | Contrast on `--color-panel` |
|---|---|---|---|---|
| `--color-bg` | `#0A0A0B` | app/stage background | — | — |
| `--color-panel` | `#141417` | caption band / status strip background | — | — |
| `--color-hairline` | `#2A2B2F` | 1px dividers, canvas bezel border | decorative only, not text | decorative only |
| `--color-text-primary` | `#F5F5F0` | gloss word, primary UI text | **18.1:1** (AAA) | **16.8:1** (AAA) |
| `--color-text-muted` | `#9AA0A6` | secondary/de-emphasized labels (timestamps, source attribution) | 7.5:1 (AAA, normal text) | 7.0:1 (AA/borderline-AAA) |
| `--color-accent-amber` | `#FFC857` | active/focus state, playhead, "loading/interpolating" status dot | 12.9:1 (AAA) | 12.0:1 (AAA) |
| `--color-signal-ready` | `#35E0C5` | "ready/loaded" status dot, connected state | 11.9:1 (AAA) | 11.1:1 (AAA) |
| `--color-signal-error` | `#FF6B5E` | error / no-data / low-confidence text and icon | 7.1:1 (AAA) | 6.6:1 (AA) |
| `--color-skeleton-bone` | `#F5F5F0` | skeleton bone lines (same as primary text — max contrast against stage) | 18.1:1 | — |
| `--color-skeleton-joint` | `#FFC857` | joint dots (amber, to visually separate "joint" from "bone" at a glance) | 12.9:1 | — |

Rules that follow from this table, stated explicitly so they survive into
implementation: **`--color-text-muted` is never used for status/error text or
anything the user must not miss** — it's AA-only-ish and reserved for genuinely
secondary info (e.g. WLASL source attribution). All *status-bearing* text
(ready/loading/error/low-confidence) uses one of the three semantic colors
(`amber`/`ready`/`error`), all of which clear AAA (7:1) against the stage
background. No color is ever the *only* signal for a state — every status dot
is paired with a text label (see Section 2), since color-only status fails
color-blind users regardless of contrast ratio.

### Spacing / layout grid

8px base unit, a small fixed scale rather than an arbitrary spacing prop:
`--space-1: 4px`, `--space-2: 8px`, `--space-3: 12px`, `--space-4: 16px`,
`--space-6: 24px`, `--space-8: 32px`, `--space-12: 48px`, `--space-16: 64px`.

Layout is three fixed horizontal bands, not a centered card:

1. **Status strip** (top, `--color-panel`, ~40px tall, `--space-3` padding) —
   spans full width, contains the LED-style status dot + label (left) and a
   latency readout in `--font-mono-data` (right).
2. **Stage** (middle, fills remaining space, `--color-bg`) — the skeleton
   canvas, letterboxed inside a `--color-hairline`-bordered frame (a visual
   "monitor bezel," reinforcing the console reference) rather than
   edge-to-edge, so the boundary of "rendered content" vs. "app chrome" is
   always visually obvious.
3. **Caption band** (bottom, `--color-panel`, min-height `--space-16`) — the
   current gloss word in `--font-display`, plus playback controls and the
   word-search input, all in `--font-label`/`--font-body`.

No rounded-corner cards, no drop shadows (flat color blocks throughout, per
the broadcast-caption reference — shadows read as "generic SaaS elevation
system," and add nothing on a single-depth-layer UI like this one).

---

## 1. Rendering approach: 2D Canvas skeleton

### Why Canvas, not SVG or Three.js, for v1

Per `CLAUDE.md`'s build order, 3D/Three.js is out of scope until v2. Between
Canvas and SVG for a 2D skeleton: Canvas is the better fit here because (a)
it's a single `<canvas>` redrawn every animation frame, which matches this
being a real-time-feeling animation loop rather than a DOM tree of ~48
persistent joint elements re-diffed by React on every frame, and (b) it keeps
the render loop as one imperative function outside React's render cycle
(driven by `requestAnimationFrame`, with React only owning the *controls*
around it), which avoids fighting React's reconciliation for a 30–60fps loop.

### Bone connections (explicit landmark-name pairs)

`extract.py`'s real `LANDMARK_NAMES` is 48 entries: 21 `left_hand_*` +
21 `right_hand_*` (MediaPipe's canonical hand topology) + 6 pose-subset
(`left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
`left_wrist`, `right_wrist`). Bones are defined as **name pairs**, resolved to
array indices via a lookup built from the sequence's own `landmark_names` at
render time — **not hardcoded numeric indices** — so this doesn't silently
break if `extract.py` ever adds landmarks (its own docstring flags face
landmarks as a planned future add-on).

Per-hand bone set (applied to both `left_hand_*` and `right_hand_*`, 21 bones
each = 42 total), using MediaPipe's standard hand skeleton:
- Thumb: `wrist→thumb_cmc→thumb_mcp→thumb_ip→thumb_tip`
- Index: `wrist→index_finger_mcp→index_finger_pip→index_finger_dip→index_finger_tip`
- Middle: `wrist→middle_finger_mcp→middle_finger_pip→middle_finger_dip→middle_finger_tip`
- Ring: `wrist→ring_finger_mcp→ring_finger_pip→ring_finger_dip→ring_finger_tip`
- Pinky: `wrist→pinky_mcp→pinky_pip→pinky_dip→pinky_tip`
- Palm base: `index_finger_mcp→middle_finger_mcp→ring_finger_mcp→pinky_mcp`

Arm/shoulder bones (from the 6-point pose subset, 3 bones):
- `left_shoulder→right_shoulder` (shoulder line, gives the figure a torso
  reference even with no torso landmarks)
- `left_shoulder→left_elbow→left_wrist`
- `right_shoulder→right_elbow→right_wrist`

Hand-to-arm bones (2 bones, connecting the two landmark groups so hands don't
float disconnected from the body): `left_wrist(pose)→left_hand_wrist`,
`right_wrist(pose)→right_hand_wrist`. Drawn in a slightly dimmer stroke than
the hand/arm bones themselves (visually secondary — it's a bridge, not a
tracked bone) — see styling below.

**Missing-point handling (a real data quirk, not hypothetical):** when
`result.pose_landmarks` isn't detected, `_pose_subset_points` fills that
frame's 6 pose points with `(0.0, 0.0, 0.0)` rather than omitting them.
`(0, 0, 0)` is a valid-looking coordinate (the canvas origin corner), so the
renderer must explicitly check for "all-zero point" before drawing any bone
touching a pose-subset landmark and skip that bone for that frame — otherwise
a genuinely undetected shoulder/elbow/wrist renders as a bogus line snapping
to the top-left corner, which is worse than not drawing it. Hand landmarks
don't have this issue: a hand is either fully present (21 real points) or the
whole frame was dropped by `extract.py` (never a partial hand), per the
extraction module's own logic.

### Styling (implements Section 0's tokens)

- Bones: `--color-skeleton-bone` (`#F5F5F0`), 3px stroke, rounded line caps.
  Hand-to-arm bridge bones: same color at 40% opacity, 2px stroke.
- Joints: `--color-skeleton-joint` (`#FFC857`) filled circles, 5px radius at
  fingertips/wrist/elbow/shoulder (structurally meaningful points), 3px
  radius at interior finger joints — a subtle size hierarchy so the hand
  doesn't read as 21 identical dots.
- No fill, no gradients, no drop shadow on the skeleton — flat lines/dots
  only, consistent with Section 0's "no elevation system" rule, and it keeps
  the figure legible against the dark stage without visual noise competing
  with the caption band below it.

### Canvas sizing / scaling from normalized 0–1 coordinates

Landmark `x`/`y` are normalized to *each source video's own frame*, per
`extract.py` (MediaPipe's standard output convention) — confirmed no
width/height is ever persisted anywhere in `PoseSequence`, the manifest, or
`download.py`'s output (checked directly; `download.py` reads `width, height`
off decoded frames internally for trimming but never writes them out). Source
WLASL clips are not guaranteed to share one aspect ratio. Given that gap,
**v1 renders into a fixed 1:1 (square) canvas** and maps `x * canvasSize`,
`y * canvasSize` directly — a deliberate, stated simplification: most
interpreter/webcam framing is close-to-square around the signer's upper body,
so the distortion on non-square source clips is expected to be mild, and it's
far simpler than trying to infer aspect ratio after the fact. Flagged in
"Known gotchas" below as a real v1 limitation with a concrete future fix
(have `extract.py` additionally persist source `width`/`height`).

`z` is not used for 2D rendering in v1 (it's present in the data for a
possible v2 3D mapping, per `types.py`'s docstring, but a 2D stick figure has
no use for depth).

Concretely: canvas backing-store size is `cssSize * window.devicePixelRatio`
(sharp on high-DPI screens, a real readability requirement, not a nice-to-
have for a captioning tool), scaled back down via `ctx.scale(dpr, dpr)`, with
`cssSize` itself responsive (fills the available square region of the Stage
band, recalculated on resize/observer, not just on mount).

### Camera fit — implemented, supersedes the direct-mapping description above

The direct `x * canvasSize` / `y * canvasSize` mapping described above was
the initial v1 approach, but real playback showed it left signs illegible:
a single hand's landmarks only cover ~12-17% of the normalized frame (up to
~35-45% across a whole word's hand travel, still small). Implemented instead
(`frontend/src/lib/skeletonBones.js`: `computeContentBounds`,
`computeFitTransform`, `projectPoint`; wired into the draw loop in
`SkeletonCanvas.jsx`): a **soft-follow, per-frame content-fit camera** that
re-fits to each frame's own real (non-`(0,0,0)`-sentinel) landmark bounds,
eased toward via exponential smoothing (`CAMERA_SMOOTHING_ALPHA = 0.18`)
rather than snapped, so the zoom/pan tracks hand motion instead of jittering
frame to frame. The fit excludes the 6-point pose subset
(`filterOutPoseSubset`) — a raised/extended arm (e.g. "phone") can span
60-100% of the frame on its own and would defeat the zoom if included; arm/
shoulder bones still draw through the resulting camera, just possibly
extending past the canvas edge for those signs. `prefers-reduced-motion`
gets an instant snap (`alpha = 1`) instead of eased motion, consistent with
this doc's Accessibility pass (below).

This changes *how* points are projected, not the square-canvas/no-aspect-
correction simplification described above — that limitation, and its
"Known gotchas" entry, still applies unchanged.

---

## 2. Input contract — verified against real code and real data

Read directly (not assumed): `pose_library/types.py`, `pose_library/extract.py`,
`pose_library/manifest.py`, `pose_library/lookup.py`, and a real sample file
(`pose_library/data/poses/about.json`) plus the real generated
`pose_library/data/poses/manifest.json` (118 words, present on disk in this
environment despite `pose_library/data/` being gitignored — won't exist on a
fresh clone until `build_library.py` has been run, see gotchas).

**Confirmed `PoseSequence.to_dict()` / on-disk JSON shape** (verified against
`about.json`):
```json
{
  "gloss": "ABOUT",
  "fps": 30.002586429864643,
  "landmark_names": ["left_hand_wrist", "...", "right_wrist"],
  "frames": [[[0.6157, 0.8835, -0.0000002], ...48 points...], ...68 frames...],
  "source": "wlasl:asldeafined:00416"
}
```
- `gloss`: uppercase string, matches `gloss_model`'s output convention.
- `fps`: a **real, non-round float** (e.g. `30.0025...`, from the source
  video's actual container metadata via `cv2`) — playback timing must use
  this exact value per word, never assume/hardcode 30 or 60.
- `landmark_names`: exactly 48 entries, fixed order (Section 1).
- `frames`: `list[list[[x, y, z]]]` — outer length varies per word (68 for
  "about"; can be as low as 11 for a low-confidence word like "phone", see
  below), inner length always 48, each point always 3 floats.
- `source`: attribution string, `wlasl:<source-site>:<video_id>` — display
  this somewhere in the UI (small, `--color-text-muted`) per the C-UDA
  attribution-retention requirement `pose_library/PLAN.md` establishes; the
  frontend is a consumer of that requirement, not just the backend.

**Confirmed `manifest.json` entry shape** (verified against the real "about"
and "phone" entries):
```json
{
  "filename": "phone.json",
  "wlasl_source": "spreadthesign",
  "wlasl_video_id": "42423",
  "wlasl_instance_id": 6,
  "license": "C-UDA-1.0; WLASL README: academic/computational use only, no commercial use",
  "total_frames_decoded": 68,
  "frames_kept": 11,
  "dropped_frame_indices": [0, 1, ..., 67],
  "low_confidence": true,
  "quality_notes": "low frame count (11 kept, threshold <25); low retention (16.2% of 68 decoded frames kept, threshold <42%); 29 mid-clip tracking gaps, not just leading/trailing (threshold >=10)"
}
```
Of the real 118-word manifest, **14 words are currently `low_confidence: true`**
(`cost`, `discount`, `doctor`, `give`, `long`, `need`, `only`, `or`, `phone`,
`speak`, + 4 more) — roughly 1 in 8, not a rare edge case to deprioritize.

**Decision: the renderer must fetch both files, not just the pose JSON.**
`manifest.json` is the only place `low_confidence`/`quality_notes` and
`dropped_frame_indices` live (they're deliberately kept out of the
`PoseSequence` schema itself, per `manifest.py`'s own docstring — provenance/
quality metadata, not playback data). A lookup that only fetches
`<word>.json` has no way to know a sequence is degraded.

**Low-confidence behavior (hard requirement, not a nice-to-have):** per
`CLAUDE.md`'s "never a silent frozen avatar / clear loading state" rule, a
`low_confidence: true` word must render with a **visible, persistent banner**
in the status strip — e.g. `● LOW-CONFIDENCE SIGN` in `--color-signal-error`,
plus the human-readable `quality_notes` string available on hover/expand —
while still playing the (degraded) skeleton underneath. Not: silently play it
as if fine (misleads the user into thinking that's a clean sign), and not:
refuse to play it at all (the data is real and may still be recognizable,
especially for something like "phone" where the surviving 11 frames are
likely the peak of the sign). The banner is the signal; playback is not
gated on it.

---

## 3. Playback scope (single word only)

- **Word selection**: a single text input in the caption band (styled per
  Section 0 — `--font-body`, high-contrast, a real visible focus ring using
  `--color-accent-amber`, not a default browser outline or none at all).
  Submits a lowercase-normalized lookup against the pose library's manifest —
  no gloss-model or ASR involved at this stage, per the user's explicit
  scoping and `CLAUDE.md`'s "fed gloss manually" build-order step.
- **Controls**: Play/Pause (toggle) and Loop (toggle), both in
  `--font-label`, both real `<button>` elements with visible focus states and
  `aria-pressed` reflecting toggle state (keyboard-operable, not click-only —
  see Section on accessibility below). No scrub bar in v1 — a single word is
  short enough (typically well under 3 seconds) that scrubbing adds control
  surface without much practical benefit yet; revisit once multi-word
  sequences exist.
- **Frame-rate-correct timing**: playback advances using each word's own
  stored `fps` via a `requestAnimationFrame` loop with a delta-time
  accumulator (advance one source frame every `1000 / fps` ms of real
  elapsed time), not a fixed-interval `setInterval` and not "one video frame
  per rAF tick" (which would run playback at the display's refresh rate
  instead of the source's actual fps, distorting apparent sign speed on a
  120Hz+ display).

---

## 4. Interpolation (within one word's sequence, not across words)

**Explicitly in scope for this stage**: smoothing a single word's own
possibly-gappy frame sequence. **Explicitly out of scope**: blending the end
of one word into the start of the next — that requires multi-word sentence
stitching from the playback queue, which per the user's own phasing is a
later stage than this one. Nothing in this stage's data model or component
boundaries should assume cross-word blending is solved yet, but nothing here
blocks adding it later either (see Section 6's boundary statement).

### The real problem: `frames` only contains *kept* frames, not a fixed-timebase array

`extract.py` drops frames outright when neither hand is detected — it does
not zero-fill or interpolate them (module docstring: "Frames where neither
hand is detected are dropped rather than filled with invented/interpolated
values"). That means naively playing `frames[0], frames[1], frames[2], ...`
back-to-back at `1000/fps` ms per step **loses the true elapsed-time gaps**
where frames were dropped — a run of 29 interior dropped frames (real number,
from "phone"'s manifest entry) would otherwise be invisible, and the
remaining 11 frames would play back faster and choppier than the sign
actually looked.

### Recommendation: reconstruct true per-frame timing from the manifest, then linearly interpolate short gaps only

1. **Reconstruct each kept frame's original video-frame index** by walking
   `0..total_frames_decoded` and skipping indices present in
   `dropped_frame_indices` (both from `manifest.json`) — this recovers, for
   each entry in `frames`, how much real time actually elapsed before it,
   using the word's own `fps`.
2. **Leading/trailing drops are trimmed, not played** — this is already true
   of the stored data (the first/last kept frame *is* the first/last frame
   with a detected hand), so playback naturally starts and ends on real
   sign content, not a held T-pose. No extra work needed here beyond not
   re-introducing padding.
3. **Interior gaps below a threshold are linearly interpolated per-landmark**
   (lerp `x`/`y`/`z` independently between the two surrounding kept frames,
   upsampled to a fixed render rate, e.g. 30fps output) — recommended
   threshold: interpolate gaps up to **~6 original frames (~200ms at 30fps
   source)**, which smooths ordinary brief tracking blips without inventing
   motion across a gap large enough to mean the hand plausibly left frame
   entirely.
4. **Interior gaps at/above that threshold are not interpolated as smooth
   motion** — instead, hold the last known pose and pair it with a brief,
   explicit "tracking gap" visual cue (e.g. the skeleton dims to ~50%
   opacity for the held duration) so a large gap reads as "the data has a
   known hole here," not as an invented movement or a silent freeze. This is
   the same "never silently frozen" principle from Section 2, applied at the
   frame level instead of the whole-word level.
5. `total_frames_decoded`/`dropped_frame_indices` are per-manifest-entry, so
   this reconstruction is only possible because Section 2 already established
   the renderer fetches the manifest alongside the pose JSON — one more
   reason that's a hard requirement, not a nice-to-have.

---

## 5. OOV / fingerspelling fallback

Checked `pose_library/lookup.py` directly: **only `get_pose_sequence()`
exists today**, returning `PoseSequence | None`. The `resolve_gloss_word()` /
`get_fingerspelling_sequence()` composing functions that `pose_library/PLAN.md`
Section 5 designs are **not yet implemented** — Section 5 itself states the
actual 26-letter self-recording is explicitly deferred/future work. So there
is no fingerspelling data or interface to consume today.

**What this stage does today for a miss**: a clear, visible message in the
caption band — e.g. `NO SIGN FOUND FOR "XYZ" — FINGERSPELLING NOT YET
AVAILABLE`, in `--color-signal-error`, `--font-heading` — replacing the
skeleton stage with that message rather than leaving the last-played word's
skeleton frozen on screen (the literal "silently frozen avatar" `CLAUDE.md`
prohibits) and rather than silently clearing to a blank canvas with no
explanation.

**What this stage leaves ready for later, without building it now**: the
frontend's word-lookup hook returns a small discriminated result type instead
of a bare `PoseSequence | null` —
`{ status: "ok", sequence, manifestEntry } | { status: "low_confidence", sequence, manifestEntry } | { status: "not_found", word } | { status: "error", message }`
— so that adding a future `{ status: "fingerspelled", letters: PoseSequence[] }`
case (matching `resolve_gloss_word`'s documented `PoseSequence | list[PoseSequence]`
return shape once it exists) is an additive branch on an existing switch, not
a rewrite of the fetch/state layer or the renderer's props contract. The
renderer itself only needs to eventually accept "a sequence, or a sequence of
sequences to play back-to-back" — worth noting as the shape it should not be
designed to preclude, not something to build now.

---

## 6. Realistic first-pass scope — explicit in/out list

**In scope for this stage:**
- Single ASL word → skeleton animation, fed by a manual text input against
  the real pose library (118 words currently on disk).
- Play/pause/loop controls, frame-accurate timing from stored `fps`.
- Within-word interpolation smoothing (Section 4).
- Visible, non-silent handling of: low-confidence words, OOV/no-data words,
  and general loading/latency state (a small "fetching…" status in the top
  strip between text submit and first frame render — never a blank stage
  with no explanation for however many ms the fetch takes).
- Accessibility pass (Section below).

**Explicitly not in scope for this stage** (later stages per `CLAUDE.md`'s
build order, not omissions):
- Live speech/ASR/VAD input (build-order step 4).
- Multi-word sentence playback or cross-word transition blending (comes with
  the playback queue from `pipeline-engineer`, not yet built).
- Fingerspelling rendering (data doesn't exist yet, Section 5).
- 3D rigged avatar / Three.js / Mixamo (explicit v2 stretch goal; `three` in
  `package.json` stays unused by v1 code).
- Any live latency measurement against a real pipeline (there is no live
  pipeline yet to measure) — the "latency" readout in Section 0's status
  strip is scoped down to *this stage's own* fetch-to-first-frame time, not a
  simulated end-to-end pipeline number, since fabricating one would be
  actively misleading.

Per `CLAUDE.md`'s "don't move to the next pipeline stage until the current
stage's tests pass" rule: this stage's own test bar (once built) is smooth,
correctly-timed playback of a representative sample of the 118 real words —
including at least one `low_confidence` word and one guaranteed-OOV word — not
just a single happy-path word.

---

## 7. Design tokens — concrete values and where they live

**Styling approach**: plain CSS with CSS custom properties, not
Tailwind/CSS-in-JS. Checked `frontend/package.json` and `node_modules`
directly — nothing styling-related is installed beyond a transitive
`postcss` (a Vite dependency, not a project choice); the project has made no
styling-library decision yet, so this plan picks the smallest option that
fits a small, latency-sensitive React app: native CSS custom properties plus
per-component CSS Modules, zero new runtime dependencies, no build-time
utility-class generation to reason about.

- `frontend/src/styles/tokens.css` — single source of truth, all
  custom properties from Section 0 defined on `:root` (colors, font stacks,
  type scale, spacing scale), imported once in `main.jsx`.
- `frontend/src/styles/global.css` — CSS reset + `body` background/font
  defaults (`--color-bg`, `--font-body`), imported once alongside tokens.
- Per-component `*.module.css` files colocated with each component (e.g.
  `SkeletonCanvas.module.css`, `CaptionBand.module.css`), consuming the
  root custom properties via `var(--token-name)` — never a hardcoded hex or
  px value inside a component's own stylesheet, so Section 0's palette/scale
  stays the single source of truth and a future contrast fix is a one-line
  change in `tokens.css`.
- **Fonts self-hosted**, not loaded from the Google Fonts CDN: both Space
  Grotesk and IBM Plex Mono are OFL-licensed, downloaded as `.woff2` into
  `frontend/public/fonts/`, declared via `@font-face` in `global.css`. Two
  reasons this is a real decision, not just a preference: (1) no third-party
  network dependency in the critical render path of a tool that's supposed
  to feel instant, and (2) it avoids the well-known FOUT flash of a system
  fallback font swapping to the CDN font a beat later, which would visibly
  undercut "considered, distinctive" on first paint.

### Accessibility pass (required before this stage is considered done)

- **Contrast**: every text/background pairing in Section 0's table checked
  against WCAG 2.1 math (not eyeballed); everything meets at least AA, all
  primary/status text meets AAA (7:1+); `--color-text-muted` explicitly
  restricted to non-critical secondary text only (stated in Section 0).
- **Type size**: 16px minimum for anything read continuously (`--font-body`),
  13px floor only for short uppercase labels directly beside their control,
  never for prose.
- **Color independence**: every status (ready/loading/error/low-confidence)
  is paired text + color, never color alone (Section 0), so it doesn't
  depend on color vision to interpret.
- **Keyboard navigation**: word-search input, Play/Pause, and Loop are all
  native `<input>`/`<button>` elements (real tab stops, native `:focus-visible`
  behavior available for free) with a visible custom focus ring
  (`--color-accent-amber`, 2px offset outline) replacing rather than removing
  the default outline — never `outline: none` with nothing in its place.
  Toggle buttons expose `aria-pressed`; the low-confidence/error/loading
  banners are rendered in a live region (`aria-live="polite"`, `role="status"`)
  so a screen reader user gets the same "don't silently freeze" guarantee a
  sighted user gets from the visible banner.
- **Motion**: skeleton interpolation and the loading indicator both respect
  `prefers-reduced-motion` — reduced-motion users get frame-stepped (no
  intermediate lerp) playback and a static "loading" label instead of an
  animated spinner/pulse, same information either way.

---

## Known gotchas / open questions (carried into implementation)

- **No HTTP layer exists yet to serve `pose_library/data/poses/*.json` to a
  browser.** `pose_library` is a Python module reading/writing local files;
  nothing in the repo currently exposes it over HTTP. Proposed stopgap for
  this stage specifically (a frontend-tooling decision, not new backend
  logic): configure Vite's dev server to serve `pose_library/data/poses/` as
  static assets (e.g. `server.fs.allow` plus a mapped path, or copying/
  symlinking into `frontend/public/poses/` as part of a small local dev
  script), so the frontend can `fetch("/poses/manifest.json")` and
  `fetch("/poses/<word>.json")` directly with no server process to run. This
  is acceptable *only* because this stage is explicitly "fed manually" with
  no live pipeline; revisit once Stage 1/2 live wiring needs a real
  request/response or streaming backend anyway, at which point pose lookup
  can move behind that same server instead of living as a frontend-only
  static-file assumption.
- **`fps` is a real, non-round float** (`30.002586429864643` for "about"),
  sourced from each video's own container metadata — never hardcode `30` or
  `60` anywhere in playback timing.
- **`landmark_names` order (48 entries) is fixed today but not guaranteed
  forever** — `extract.py`'s own docstring calls out face landmarks as a
  planned future add-on. The bone map must resolve landmark pairs by
  building a name→index lookup from each sequence's own `landmark_names`
  array at load time, never by hardcoded numeric index, so a future schema
  addition doesn't silently misalign bones.
- **Pose-subset zero-fill (`(0.0, 0.0, 0.0)`) is a real, valid-looking point,
  not a sentinel `null`.** The renderer must treat an exact `(0,0,0)`
  pose-subset point as "not detected" and skip bones touching it, or a
  missing shoulder/elbow/wrist renders as a bogus line to the canvas corner.
- **No source video width/height is persisted anywhere** in `PoseSequence`,
  `manifest.json`, or `download.py`'s output, despite `download.py` reading
  it internally for trimming. v1 assumes a fixed 1:1 canvas and accepts mild
  aspect-ratio distortion on non-square source clips as a known limitation
  (Section 1); a clean future fix is having `extract.py` persist source
  `width`/`height` alongside each `PoseSequence` so the renderer can letterbox
  exactly instead of assuming square.
- **14 of 118 words in the real manifest are `low_confidence: true` today**
  (~12%) — budget real UI testing time against actual low-confidence words
  (e.g. "phone," 11/68 frames kept) during implementation, not just a
  synthetic test fixture, mirroring the same scope note
  `pose_library/PLAN.md` makes about fingerspelling not being a rare edge
  case either.
- **`pose_library/data/` (including `manifest.json`) is gitignored** — it
  exists in this environment because `build_library.py` has already been run
  locally, but won't exist on a fresh clone. The frontend's data-fetch layer
  should fail into the same visible "no data" state (not a silent blank
  screen or an uncaught fetch error) whether the cause is "word not in
  library" or "library not built yet on this machine."
