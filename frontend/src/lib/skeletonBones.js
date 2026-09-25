/**
 * Skeleton bone topology, per `frontend/PLAN.md` Section 1.
 *
 * Bones are defined as landmark **name** pairs, never hardcoded numeric
 * indices -- `extract.py`'s own docstring flags face landmarks as a planned
 * future addition to `LANDMARK_NAMES`, so resolving bones by name against
 * each sequence's own `landmark_names` array (via `buildBoneIndexList`)
 * means an order/length change upstream can't silently misalign bones.
 */

const HAND_BONES = [
  // Thumb
  ["wrist", "thumb_cmc"],
  ["thumb_cmc", "thumb_mcp"],
  ["thumb_mcp", "thumb_ip"],
  ["thumb_ip", "thumb_tip"],
  // Index
  ["wrist", "index_finger_mcp"],
  ["index_finger_mcp", "index_finger_pip"],
  ["index_finger_pip", "index_finger_dip"],
  ["index_finger_dip", "index_finger_tip"],
  // Middle
  ["wrist", "middle_finger_mcp"],
  ["middle_finger_mcp", "middle_finger_pip"],
  ["middle_finger_pip", "middle_finger_dip"],
  ["middle_finger_dip", "middle_finger_tip"],
  // Ring
  ["wrist", "ring_finger_mcp"],
  ["ring_finger_mcp", "ring_finger_pip"],
  ["ring_finger_pip", "ring_finger_dip"],
  ["ring_finger_dip", "ring_finger_tip"],
  // Pinky
  ["wrist", "pinky_mcp"],
  ["pinky_mcp", "pinky_pip"],
  ["pinky_pip", "pinky_dip"],
  ["pinky_dip", "pinky_tip"],
  // Palm base
  ["index_finger_mcp", "middle_finger_mcp"],
  ["middle_finger_mcp", "ring_finger_mcp"],
  ["ring_finger_mcp", "pinky_mcp"],
];

/** Joint radius hierarchy (Section 1 styling): larger dots at structurally
 * meaningful points, smaller at interior finger joints. Keyed by the
 * landmark's *unprefixed* suffix (post `left_hand_`/`right_hand_` strip). */
const LARGE_JOINT_SUFFIXES = new Set([
  "wrist",
  "thumb_tip",
  "index_finger_tip",
  "middle_finger_tip",
  "ring_finger_tip",
  "pinky_tip",
]);

/**
 * @typedef {Object} Bone
 * @property {number} a - Index of the first landmark.
 * @property {number} b - Index of the second landmark.
 * @property {"primary"|"bridge"} kind - `"bridge"` for hand-to-arm bones,
 *   drawn dimmer per Section 1; `"primary"` for everything else.
 * @property {boolean} isPoseSubset - True if either endpoint is one of the
 *   6 pose-subset landmarks (shoulder/elbow/wrist), which can be zero-filled
 *   per-frame and must be skip-checked at render time.
 */

/**
 * Builds the full bone list (indices into a specific sequence's own
 * `landmark_names`) plus a per-index joint-radius lookup, resolving every
 * bone by landmark **name**, not position.
 *
 * @param {string[]} landmarkNames - The sequence's own `landmark_names`.
 * @returns {{bones: Bone[], jointRadiusByIndex: number[], isPoseSubsetByIndex: boolean[]}}
 *   The resolved bone list, a same-length-as-landmarkNames array of joint
 *   radii (px), and a same-length boolean array flagging which indices are
 *   pose-subset landmarks (resolved by name, never a hardcoded index range,
 *   so a future `landmark_names` change can't silently misalign this).
 */
export function buildSkeletonTopology(landmarkNames) {
  const indexByName = new Map(landmarkNames.map((name, i) => [name, i]));

  const bones = [];
  const addBone = (nameA, nameB, kind) => {
    const a = indexByName.get(nameA);
    const b = indexByName.get(nameB);
    if (a === undefined || b === undefined) {
      return; // Landmark set doesn't include this pair -- skip, don't crash.
    }
    const isPoseSubset = POSE_SUBSET_NAMES.has(nameA) || POSE_SUBSET_NAMES.has(nameB);
    bones.push({ a, b, kind, isPoseSubset });
  };

  for (const side of ["left_hand", "right_hand"]) {
    for (const [suffixA, suffixB] of HAND_BONES) {
      addBone(`${side}_${suffixA}`, `${side}_${suffixB}`, "primary");
    }
  }

  addBone("left_shoulder", "right_shoulder", "primary");
  addBone("left_shoulder", "left_elbow", "primary");
  addBone("left_elbow", "left_wrist", "primary");
  addBone("right_shoulder", "right_elbow", "primary");
  addBone("right_elbow", "right_wrist", "primary");

  addBone("left_wrist", "left_hand_wrist", "bridge");
  addBone("right_wrist", "right_hand_wrist", "bridge");

  const jointRadiusByIndex = landmarkNames.map((name) => {
    const suffix = name.replace(/^(left_hand_|right_hand_)/, "");
    if (POSE_SUBSET_NAMES.has(name) || LARGE_JOINT_SUFFIXES.has(suffix)) {
      return 5;
    }
    return 3;
  });

  const isPoseSubsetByIndex = landmarkNames.map((name) => POSE_SUBSET_NAMES.has(name));

  return { bones, jointRadiusByIndex, isPoseSubsetByIndex };
}

const POSE_SUBSET_NAMES = new Set([
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
]);

/**
 * A pose-subset landmark's `(0, 0, 0)` fill value means "not detected in
 * this frame" -- see `_pose_subset_points` in `extract.py` and PLAN.md's
 * "Known gotchas". A real landmark coordinate is never exactly `(0, 0, 0)`.
 *
 * @param {[number, number, number]} point
 * @returns {boolean}
 */
export function isUndetectedPoint(point) {
  return point[0] === 0 && point[1] === 0 && point[2] === 0;
}

/**
 * Drops the pose-subset (shoulder/elbow/wrist) points from a pose, keeping
 * only hand landmarks.
 *
 * Used to fit the camera to hand content specifically: a shoulder-to-wrist
 * chain can span most of the frame on its own for a sign where the hand
 * is raised or extended away from the body (e.g. a real "phone" extraction
 * where that chain alone spans ~60-100% of the frame) -- including those
 * points in a content-fit bounding box defeats the zoom entirely for
 * exactly the signs that most need it. The arm/shoulder bones still get
 * drawn through the resulting (hand-fitted) camera; they simply may extend
 * beyond the canvas for a raised or extended hand, which is an acceptable
 * trade-off for legible fingers.
 *
 * @param {Array<[number, number, number]>} pose - One frame's full point list.
 * @param {boolean[]} isPoseSubsetByIndex - From `buildSkeletonTopology`.
 * @returns {Array<[number, number, number]>} Only the hand-landmark points.
 */
export function filterOutPoseSubset(pose, isPoseSubsetByIndex) {
  return pose.filter((_, i) => !isPoseSubsetByIndex[i]);
}

/** Padding around the fitted content bounds, as a fraction of the larger
 * span dimension on each side. Found empirically: a real word's own hand
 * landmarks only cover roughly 12-17% of a source video's 0-1 coordinate
 * space per hand (~50% combined with arms/shoulders) -- PLAN.md's original
 * "map x*canvasSize directly" approach left the signing content tiny and
 * illegible. Fitting to real content instead needs *some* margin so bones
 * and joints near the edge aren't clipped by the canvas or the bezel
 * border. */
const FIT_PADDING_FRACTION = 0.18;

/**
 * Computes the bounding box of every *real* (non-`(0,0,0)`-sentinel)
 * landmark across an entire timeline's frames.
 *
 * Computed once per whole sign (not per frame) so the camera framing stays
 * stable across playback -- a per-frame fit would make the skeleton appear
 * to jitter/zoom as the bounding box of visible points changes frame to
 * frame, which would be worse than the original static-framing problem.
 *
 * @param {Array<Array<[number, number, number]>>} poses - One pose (array
 *   of landmark points) per timeline frame.
 * @returns {{minX: number, minY: number, maxX: number, maxY: number}} The
 *   real-content bounding box, or the full `[0, 1]` square if no real point
 *   was found at all (a degenerate sequence -- fail safe to the old
 *   full-frame behavior rather than divide by zero).
 */
export function computeContentBounds(poses) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const pose of poses) {
    for (const point of pose) {
      if (isUndetectedPoint(point)) continue;
      if (point[0] < minX) minX = point[0];
      if (point[0] > maxX) maxX = point[0];
      if (point[1] < minY) minY = point[1];
      if (point[1] > maxY) maxY = point[1];
    }
  }

  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  }
  return { minX, minY, maxX, maxY };
}

/**
 * Builds a square, aspect-preserving fit transform from a content bounding
 * box, so `projectPoint` can map normalized landmark coordinates onto the
 * canvas zoomed and centered on the real content instead of the full,
 * mostly-empty source-video frame.
 *
 * @param {{minX: number, minY: number, maxX: number, maxY: number}} bounds -
 *   From `computeContentBounds`.
 * @returns {{centerX: number, centerY: number, span: number}} `span` is the
 *   side length (in source-coordinate units) of the padded square window
 *   centered on the content -- the larger of width/height plus padding, so
 *   neither axis is stretched relative to the other.
 */
export function computeFitTransform(bounds) {
  const width = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const height = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const span = Math.max(width, height) * (1 + FIT_PADDING_FRACTION * 2);
  return { centerX, centerY, span };
}

/**
 * Projects one normalized `[x, y]` landmark coordinate through a fit
 * transform into canvas pixel space.
 *
 * @param {[number, number, number]} point - A landmark's `[x, y, z]`.
 * @param {{centerX: number, centerY: number, span: number}} transform -
 *   From `computeFitTransform`.
 * @param {number} cssSize - The canvas's CSS pixel size (it's square).
 * @returns {[number, number]} `[canvasX, canvasY]`.
 */
export function projectPoint(point, transform, cssSize) {
  const nx = (point[0] - transform.centerX) / transform.span + 0.5;
  const ny = (point[1] - transform.centerY) / transform.span + 0.5;
  return [nx * cssSize, ny * cssSize];
}

/**
 * Advances a soft-follow camera one step toward a frame's own real hand
 * content -- or holds it exactly steady if the frame has *no* trackable
 * hand content at all.
 *
 * Root-caused during multi-word playback (see
 * `frontend/MULTIWORD_PLAN.md`'s "Known gotchas"): a cross-word transition
 * frame where `lerpPose` interpolates between two words that use
 * *different* hands (e.g. one word's last real frame has only its left
 * hand tracked, the next word's first real frame has only its right)
 * forces *both* hands to the `(0,0,0)` sentinel for the entire transition
 * span, per-landmark, per `lerpPose`'s own zero-endpoint guard -- not just
 * one hand staying a static phantom (the already-documented, narrower
 * "phantom static hand" case where the *other* hand still tracks
 * normally). With zero real points anywhere in the frame,
 * `computeContentBounds` falls back to the full `[0,1]` frame -- easing
 * toward *that* every such frame causes a jarring zoom-out/zoom-in
 * oscillation at every hand-switching word boundary, confirmed directly
 * against the real "about"/"bathroom"/"doctor"/"angry" library data (the
 * `about`->`bathroom` transition: `about`'s last frame has 21 real left-
 * hand points and 0 right; `bathroom`'s first frame is the exact mirror).
 * Holding the camera at its last real position instead -- exactly like
 * `reconstructTimeline.js`'s own "hold, don't invent" rule for a missing
 * within-word gap -- keeps it stable through the transition, rather than
 * chasing a bounding box that doesn't represent any real content.
 *
 * @param {{centerX: number, centerY: number, span: number}} camera - The
 *   current camera state, updated in place (matching the caller's existing
 *   mutable-ref usage) and also returned for convenience.
 * @param {Array<[number, number, number]>} pose - The frame's full point
 *   list (pose-subset landmarks included; filtered out internally).
 * @param {boolean[]} isPoseSubsetByIndex - From `buildSkeletonTopology`.
 * @param {number} alpha - Smoothing factor in `(0, 1]`; `1` snaps
 *   instantly (the `prefers-reduced-motion` case).
 * @returns {{centerX: number, centerY: number, span: number}} The same
 *   `camera` object.
 */
export function stepCamera(camera, pose, isPoseSubsetByIndex, alpha) {
  const handPose = filterOutPoseSubset(pose, isPoseSubsetByIndex);
  if (handPose.every(isUndetectedPoint)) {
    return camera;
  }
  const rawFit = computeFitTransform(computeContentBounds([handPose]));
  camera.centerX += (rawFit.centerX - camera.centerX) * alpha;
  camera.centerY += (rawFit.centerY - camera.centerY) * alpha;
  camera.span += (rawFit.span - camera.span) * alpha;
  return camera;
}
