/**
 * Stitches multiple words' pose sequences into one continuous playback
 * `Timeline`, with a short interpolated transition inserted at each word
 * boundary instead of a hard jump-cut.
 *
 * See `frontend/MULTIWORD_PLAN.md` Section 3 for the full design rationale,
 * including the two real-data findings that drive it: real per-word `fps`
 * varies meaningfully (confirmed 24/25/~29.97/~30.003 across a sample), and
 * `SkeletonCanvas.jsx`'s playback loop steps an entire `Timeline` at one
 * shared `frameDurationMs` with no per-frame timing mechanism -- so every
 * word must be resampled onto one fixed target fps *before* concatenation,
 * not stitched at each word's own native rate.
 *
 * Reuses `reconstructTimeline` (within-word gap reconstruction, unchanged)
 * and `lerpPose` (the existing zero-point-aware lerp, exported from
 * `reconstructTimeline.js` for exactly this reuse) rather than
 * reimplementing either.
 */

import { reconstructTimeline, lerpPose } from "./reconstructTimeline.js";

/** Fixed output frame rate every word is resampled onto before stitching,
 * chosen because it's close to most real words' own native fps (minimizing
 * resampling distortion for the common case). See PLAN.md Section 3,
 * Finding 2. */
export const DEFAULT_TARGET_FPS = 30;

/** Duration of the inserted transition between two consecutive words, in
 * milliseconds. Matches the same order of magnitude as
 * `reconstructTimeline.js`'s own `GAP_INTERPOLATION_THRESHOLD_FRAMES`
 * (~200ms at a 30fps source), per PLAN.md Section 3, rather than
 * introducing a second, differently-tuned notion of "how long a smooth
 * transition should take." */
export const DEFAULT_TRANSITION_MS = 200;

/**
 * @typedef {Object} StitchedTimelineFrame
 * @property {number} index - Position in the merged timeline (0-based).
 * @property {number} time - Seconds elapsed since the first frame, at the
 *   fixed output fps.
 * @property {Array<[number, number, number]>} pose - 48 [x, y, z] points.
 * @property {"real"|"interpolated"} state - `"real"` when this output frame
 *   coincides exactly with one of the word's own reconstructed frames (no
 *   resampling interpolation needed -- true for every word's very first
 *   frame, since output time 0 always aligns exactly); `"interpolated"`
 *   for every resampled or transition frame otherwise.
 * @property {boolean} dimmed - Carried through from whichever
 *   within-word-gap frame(s) a resampled point falls between (see
 *   `resampleToFixedFps`), so a `reconstructTimeline`-flagged tracking gap
 *   stays visible as a dim cue even after resampling. Transition frames
 *   between words are never dimmed -- a word boundary isn't a tracking gap.
 */

/**
 * @typedef {Object} WordBoundary
 * @property {string} word - The gloss word this span belongs to.
 * @property {number} wordIndex - This word's position in the `perWordResults`
 *   input array (0-based) -- i.e. an index into the *filtered*
 *   ok/low_confidence list passed to `stitchTimelines`, not the original
 *   submitted sequence (which may also contain skipped `not_found` words --
 *   see PLAN.md Section 4).
 * @property {number} startFrameIndex - First merged-timeline frame index
 *   belonging to this word (inclusive). For every word after the first,
 *   this includes the transition frames leading *into* it, so spans are
 *   contiguous and gapless across the whole merged timeline -- no frame
 *   belongs to neither word.
 * @property {number} endFrameIndex - Last merged-timeline frame index
 *   belonging to this word (inclusive): its own last real/resampled frame,
 *   never including the transition *out* of it (that belongs to the next
 *   word's span instead). `endFrameIndex < startFrameIndex` for a
 *   degenerate zero-frame word (no real content to show or transition
 *   into/out of).
 */

/**
 * Resamples a `reconstructTimeline()`-produced dense timeline (at its own
 * native fps) onto a fixed output fps, via `lerpPose` between whichever two
 * original frames bracket each new output timestamp.
 *
 * @param {import("./reconstructTimeline.js").Timeline} denseTimeline
 * @param {number} targetFps
 * @returns {Array<{pose: Array<[number, number, number]>, state: "real"|"interpolated", dimmed: boolean}>}
 */
function resampleToFixedFps(denseTimeline, targetFps) {
  const { frames, durationSeconds } = denseTimeline;
  if (frames.length === 0) {
    return [];
  }
  if (frames.length === 1 || durationSeconds === 0) {
    return [{ pose: frames[0].pose, state: "real", dimmed: frames[0].dimmed }];
  }

  const stepCount = Math.max(1, Math.round(durationSeconds * targetFps));
  const output = [];
  let cursor = 0;

  for (let k = 0; k <= stepCount; k += 1) {
    const tOut = (k / stepCount) * durationSeconds;

    while (cursor < frames.length - 2 && frames[cursor + 1].time < tOut) {
      cursor += 1;
    }

    const a = frames[cursor];
    const b = frames[cursor + 1] ?? frames[cursor];

    if (b.time === a.time) {
      output.push({ pose: a.pose, state: k === 0 ? "real" : "interpolated", dimmed: a.dimmed });
      continue;
    }

    const localT = (tOut - a.time) / (b.time - a.time);
    if (localT <= 0) {
      output.push({ pose: a.pose, state: k === 0 ? "real" : "interpolated", dimmed: a.dimmed });
    } else if (localT >= 1) {
      output.push({ pose: b.pose, state: "interpolated", dimmed: b.dimmed });
    } else {
      output.push({
        pose: lerpPose(a.pose, b.pose, localT),
        state: "interpolated",
        dimmed: a.dimmed || b.dimmed,
      });
    }
  }

  return output;
}

/**
 * Builds the interpolated transition frames inserted between two words,
 * excluding both endpoints (the previous word's last resampled frame and
 * the next word's first resampled frame are already present in the merged
 * timeline on either side) -- the same convention
 * `reconstructTimeline.js`'s own gap-fill uses for its interior interpolated
 * frames.
 *
 * @param {Array<[number, number, number]>} poseA - The previous word's last frame.
 * @param {Array<[number, number, number]>} poseB - The next word's first frame.
 * @param {number} transitionMs
 * @param {number} targetFps
 * @returns {Array<{pose: Array<[number, number, number]>, state: "interpolated", dimmed: false}>}
 */
function buildTransitionFrames(poseA, poseB, transitionMs, targetFps) {
  const frameDurationMs = 1000 / targetFps;
  const count = Math.max(1, Math.round(transitionMs / frameDurationMs));
  const frames = [];
  for (let i = 1; i <= count; i += 1) {
    const t = i / (count + 1);
    frames.push({ pose: lerpPose(poseA, poseB, t), state: "interpolated", dimmed: false });
  }
  return frames;
}

/**
 * Stitches multiple words' `(sequence, manifestEntry)` pairs into one
 * merged `Timeline`, with `transitionMs` of interpolated frames inserted at
 * each word boundary.
 *
 * Only ever receives `ok`/`low_confidence` entries -- `not_found` words are
 * filtered out of the sequence *before* calling this function (per PLAN.md
 * Section 4), so a `not_found` word never needs special transition
 * handling here: the real words on either side of it simply become
 * adjacent in `perWordResults` and get one ordinary transition between
 * them, indistinguishable from any other adjacent pair.
 *
 * @param {Array<{word: string, sequence: object, manifestEntry: object}>} perWordResults
 * @param {{targetFps?: number, transitionMs?: number}} [options]
 * @returns {{timeline: import("./reconstructTimeline.js").Timeline, wordBoundaries: WordBoundary[]}}
 */
export function stitchTimelines(
  perWordResults,
  { targetFps = DEFAULT_TARGET_FPS, transitionMs = DEFAULT_TRANSITION_MS } = {}
) {
  const frameDurationMs = 1000 / targetFps;

  if (perWordResults.length === 0) {
    return {
      timeline: { fps: targetFps, frameDurationMs, durationSeconds: 0, frames: [] },
      wordBoundaries: [],
    };
  }

  const mergedFrames = [];
  const wordBoundaries = [];

  perWordResults.forEach(({ word, sequence, manifestEntry }, wordIndex) => {
    const dense = reconstructTimeline(sequence, manifestEntry);
    const resampled = resampleToFixedFps(dense, targetFps);

    if (resampled.length === 0) {
      // Degenerate zero-frame word: nothing to show and nothing to
      // transition into/out of. Record an empty span rather than crashing
      // or silently dropping the boundary entry.
      wordBoundaries.push({
        word,
        wordIndex,
        startFrameIndex: mergedFrames.length,
        endFrameIndex: mergedFrames.length - 1,
      });
      return;
    }

    // Captured *before* inserting any incoming transition frames below, so
    // a word's recorded span includes the transition leading into it --
    // giving every word (after the first) a span that starts the instant
    // the previous word's own real content ends, with no frame belonging
    // to neither word. This keeps `wordBoundaries` spans contiguous and
    // gapless across the whole merged timeline (see the test bar in
    // `frontend/MULTIWORD_PLAN.md` Section 7).
    const startFrameIndex = mergedFrames.length;

    if (mergedFrames.length > 0) {
      const prevPose = mergedFrames[mergedFrames.length - 1].pose;
      const nextPose = resampled[0].pose;
      for (const tf of buildTransitionFrames(prevPose, nextPose, transitionMs, targetFps)) {
        mergedFrames.push(tf);
      }
    }

    for (const f of resampled) {
      mergedFrames.push(f);
    }
    const endFrameIndex = mergedFrames.length - 1;

    wordBoundaries.push({ word, wordIndex, startFrameIndex, endFrameIndex });
  });

  const frames = mergedFrames.map((f, i) => ({
    index: i,
    time: (i * frameDurationMs) / 1000,
    pose: f.pose,
    state: f.state,
    dimmed: f.dimmed,
  }));

  const durationSeconds = frames.length > 0 ? frames[frames.length - 1].time : 0;

  return {
    timeline: { fps: targetFps, frameDurationMs, durationSeconds, frames },
    wordBoundaries,
  };
}
