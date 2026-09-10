/**
 * Reconstructs a dense, evenly-spaced (1/fps) playback timeline from a
 * `PoseSequence`'s *kept-only* frames plus its manifest entry's drop record.
 *
 * `extract.py` drops frames outright when neither hand is detected -- it
 * does not zero-fill or interpolate them. Naively playing `frames[0],
 * frames[1], ...` back-to-back loses the true elapsed-time gaps where
 * frames were dropped. This module recovers each kept frame's true original
 * video-frame index (by walking 0..total_frames_decoded and skipping
 * `dropped_frame_indices`), then fills the gaps between kept frames:
 *
 * - No gap: the two kept frames are simply adjacent.
 * - Short interior gap (<= GAP_INTERPOLATION_THRESHOLD_FRAMES original
 *   frames): linearly interpolated per-landmark between the two
 *   surrounding kept frames.
 * - Long interior gap (> GAP_INTERPOLATION_THRESHOLD_FRAMES): not
 *   interpolated as smooth motion -- the last known pose is held and
 *   flagged `dimmed: true` for the held span, so a real tracking hole reads
 *   as a visible "gap" cue rather than invented motion or a silent freeze.
 *
 * Leading/trailing dropped frames are never re-introduced: the returned
 * timeline spans only from the first kept frame's original index to the
 * last kept frame's original index, inclusive -- per PLAN.md Section 4
 * point 2, that trimming is already true of the stored data.
 *
 * See `frontend/PLAN.md` Section 4 for the full design rationale.
 */

/** Interior gaps at/below this many *original* dropped frames are
 * linearly interpolated; above it, the last known pose is held and dimmed
 * instead. ~6 frames is ~200ms at a 30fps source -- long enough to smooth
 * ordinary brief tracking blips, short enough not to invent motion across a
 * gap large enough that the hand plausibly left frame entirely. */
export const GAP_INTERPOLATION_THRESHOLD_FRAMES = 6;

/**
 * Returns true if a landmark point is the exact `(0, 0, 0)` sentinel
 * `extract.py`'s `_pose_subset_points` fills in when a pose-subset landmark
 * (shoulder/elbow/wrist) wasn't detected in an otherwise-kept frame. It's a
 * hardcoded literal fill value, not a computed near-zero result, so exact
 * equality is the correct check (unlike real coordinates, which are never
 * exactly 0 -- see `about.json`'s z values like `-2.19e-7`).
 *
 * @param {[number, number, number]} point - A single landmark's [x, y, z].
 * @returns {boolean} Whether this point is the "not detected" sentinel.
 */
function isZeroPoint(point) {
  return point[0] === 0 && point[1] === 0 && point[2] === 0;
}

/**
 * Linearly interpolates a full pose (all landmarks) between two poses.
 *
 * Per-landmark: if either endpoint is the `(0,0,0)` "not detected"
 * sentinel, the interpolated point is also `(0,0,0)` rather than lerped --
 * otherwise an undetected pose-subset landmark would produce a fake motion
 * path sliding into/out of the canvas origin corner, which is worse than no
 * motion at all (mirrors the renderer's own zero-point skip rule from
 * PLAN.md Section 1, applied here so it can't be defeated by interpolation).
 *
 * @param {Array<[number, number, number]>} poseA - Pose at t=0.
 * @param {Array<[number, number, number]>} poseB - Pose at t=1.
 * @param {number} t - Interpolation fraction in [0, 1].
 * @returns {Array<[number, number, number]>} The interpolated pose.
 */
function lerpPose(poseA, poseB, t) {
  return poseA.map((pointA, i) => {
    const pointB = poseB[i];
    if (isZeroPoint(pointA) || isZeroPoint(pointB)) {
      return [0, 0, 0];
    }
    return [
      pointA[0] + (pointB[0] - pointA[0]) * t,
      pointA[1] + (pointB[1] - pointA[1]) * t,
      pointA[2] + (pointB[2] - pointA[2]) * t,
    ];
  });
}

/**
 * @typedef {Object} TimelineFrame
 * @property {number} index - Position in the dense timeline (0-based).
 * @property {number} originalIndex - The reconstructed original decoded
 *   video-frame index this timeline entry corresponds to.
 * @property {number} time - Seconds elapsed since the first kept frame
 *   (i.e. `time` of `frames[0]` is always 0), derived from the sequence's
 *   own real `fps`.
 * @property {Array<[number, number, number]>} pose - 48 [x, y, z] points,
 *   in `landmark_names` order.
 * @property {"real"|"interpolated"|"held"} state - `"real"` for an actual
 *   kept frame, `"interpolated"` for a short-gap-filled synthetic frame,
 *   `"held"` for a long-gap frame repeating the last known real pose.
 * @property {boolean} dimmed - True for `"held"` frames -- the renderer
 *   should visually dim the skeleton for these, per PLAN.md Section 4
 *   point 4's "never silently frozen" rule at the frame level.
 */

/**
 * @typedef {Object} Timeline
 * @property {number} fps - The sequence's own real (non-round) fps.
 * @property {number} frameDurationMs - `1000 / fps`, convenience for
 *   playback loops.
 * @property {number} durationSeconds - Total span of the timeline in
 *   seconds (time of the last frame).
 * @property {TimelineFrame[]} frames - The dense, gap-filled timeline.
 */

/**
 * Builds a dense playback timeline from a pose sequence and its manifest
 * entry.
 *
 * @param {{fps: number, landmark_names: string[], frames: Array<Array<[number, number, number]>>}} sequence -
 *   The parsed `PoseSequence`-shaped object (only `fps`/`frames` are used).
 * @param {{total_frames_decoded: number, dropped_frame_indices: number[]}} manifestEntry -
 *   The word's manifest entry (only these two fields are used).
 * @returns {Timeline} The reconstructed, gap-filled timeline.
 * @throws {Error} If the manifest's drop record is inconsistent with the
 *   sequence's kept-frame count (a data integrity problem upstream, not
 *   something safe to silently paper over here).
 */
export function reconstructTimeline(sequence, manifestEntry) {
  const { fps, frames } = sequence;
  const { total_frames_decoded: totalFramesDecoded, dropped_frame_indices: droppedIndices } =
    manifestEntry;

  if (frames.length === 0) {
    return { fps, frameDurationMs: 1000 / fps, durationSeconds: 0, frames: [] };
  }

  const droppedSet = new Set(droppedIndices);
  const originalIndices = [];
  for (let i = 0; i < totalFramesDecoded; i += 1) {
    if (!droppedSet.has(i)) {
      originalIndices.push(i);
    }
  }

  if (originalIndices.length !== frames.length) {
    throw new Error(
      `reconstructTimeline: manifest/sequence mismatch -- ${originalIndices.length} ` +
        `non-dropped original indices but ${frames.length} kept frames. ` +
        "The manifest entry does not match this sequence's actual frame count."
    );
  }

  const firstOriginalIndex = originalIndices[0];
  const timelineFrames = [];

  for (let k = 0; k < frames.length; k += 1) {
    const originalIndex = originalIndices[k];
    timelineFrames.push({
      index: timelineFrames.length,
      originalIndex,
      time: (originalIndex - firstOriginalIndex) / fps,
      pose: frames[k],
      state: "real",
      dimmed: false,
    });

    if (k === frames.length - 1) {
      break;
    }

    const nextOriginalIndex = originalIndices[k + 1];
    const gapSize = nextOriginalIndex - originalIndex - 1;
    if (gapSize <= 0) {
      continue;
    }

    const poseBefore = frames[k];
    const poseAfter = frames[k + 1];
    const shortGap = gapSize <= GAP_INTERPOLATION_THRESHOLD_FRAMES;

    for (let g = 1; g <= gapSize; g += 1) {
      const gapOriginalIndex = originalIndex + g;
      if (shortGap) {
        const t = g / (gapSize + 1);
        timelineFrames.push({
          index: timelineFrames.length,
          originalIndex: gapOriginalIndex,
          time: (gapOriginalIndex - firstOriginalIndex) / fps,
          pose: lerpPose(poseBefore, poseAfter, t),
          state: "interpolated",
          dimmed: false,
        });
      } else {
        timelineFrames.push({
          index: timelineFrames.length,
          originalIndex: gapOriginalIndex,
          time: (gapOriginalIndex - firstOriginalIndex) / fps,
          pose: poseBefore,
          state: "held",
          dimmed: true,
        });
      }
    }
  }

  return {
    fps,
    frameDurationMs: 1000 / fps,
    durationSeconds: timelineFrames[timelineFrames.length - 1].time,
    frames: timelineFrames,
  };
}
