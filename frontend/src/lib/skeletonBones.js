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
