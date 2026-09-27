/**
 * Per-frame 3D geometry for the hand renderer: where every joint, bone, and
 * palm surface goes, how thick it is, and whether it's drawn at all.
 *
 * Pure (no Three.js, no DOM), so it can be unit-tested without a GPU; the
 * WebGL side (`handScene.js`) only turns this into instance matrices. See
 * `frontend/RENDERING_UPGRADE_PLAN.md` Sections 3 and 5.
 */

import { isUndetectedPoint, toWorld } from "./skeletonBones.js";

/** Thickness as a fraction of each hand's own palm length (wrist -> middle
 * knuckle), so a hand filmed small or large looks equally solid. Every joint
 * is at least ~1.2x the radius of the bone ends it joins: with near-equal
 * radii the sphere and cylinder surfaces meet almost tangentially and
 * render as a jagged, z-fighting seam (seen in the browser check). Slimmer
 * than a real finger on purpose, so curled fingers in a closed handshape
 * (A, S, O) stay visually separate. Tuned by eye on real signs
 * (RENDERING_UPGRADE_PLAN.md Section 7). */
export const HAND_RADII = {
  wrist: 0.1,
  knuckle: 0.082, // *_mcp and thumb_cmc
  middleJoint: 0.076, // *_pip and thumb_mcp
  outerJoint: 0.072, // *_dip and thumb_ip
  tip: 0.064,
  fingerBone: 0.06,
  metacarpal: 0.045, // wrist -> knuckle, inside the palm
  palmEdge: 0.035, // knuckle -> knuckle; the palm fill carries this edge
};

/** Arms are context, not the sign: drawn thinner than a finger (a true
 * forearm would be ~3x a finger's width and dominate the frame, since the
 * camera zooms to the hands). */
export const ARM_RADII = { joint: 0.075, bone: 0.06 };

/** Bone tip radius as a fraction of its base radius: bones narrow toward
 * the fingertip (topology bones always run proximal -> distal). The
 * renderer's cylinder shape uses it (`handScene.js`). */
export const BONE_TAPER = 0.8;

/** Used when a hand's palm length can't be measured (its wrist or middle
 * knuckle isn't tracked): the library's median, measured over all 144 served
 * sequences (8,796 tracked hand-frames; median 0.091, p10-p90 0.055-0.141). */
export const FALLBACK_PALM_LENGTH = 0.09;

const SIDES = ["left", "right"];

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function jointRadiusFactor(suffix) {
  if (suffix === "wrist") return HAND_RADII.wrist;
  if (suffix.endsWith("_tip")) return HAND_RADII.tip;
  if (suffix.endsWith("_mcp") && !suffix.startsWith("thumb")) return HAND_RADII.knuckle;
  if (suffix === "thumb_cmc") return HAND_RADII.knuckle;
  if (suffix.endsWith("_pip") || suffix === "thumb_mcp") return HAND_RADII.middleJoint;
  return HAND_RADII.outerJoint; // *_dip, thumb_ip
}

function boneRadiusFactor(suffixA, suffixB) {
  if (suffixA === "wrist") return HAND_RADII.metacarpal;
  if (suffixA.endsWith("_mcp") && suffixB.endsWith("_mcp")) return HAND_RADII.palmEdge;
  return HAND_RADII.fingerBone;
}

/** Fingertips and the wrist get the amber joint color; interior joints
 * are bone-colored, so each finger reads as one continuous capsule ending
 * in a marked tip. With every joint amber (tried first), a fingertip looked
 * the same as a knuckle, and in a closed handshape like A the folded fingers
 * read as fingers pointing up. Same emphasis as the 2D view's larger dots at
 * tips and wrist (`buildSkeletonTopology`'s `LARGE_JOINT_SUFFIXES`). */
function isAccent(suffix) {
  return suffix === "wrist" || suffix.endsWith("_tip");
}

/** Splits `left_hand_index_finger_tip` into `["left", "index_finger_tip"]`;
 * `null` for a pose-subset name. */
function handPart(name) {
  const match = /^(left|right)_hand_(.+)$/.exec(name);
  return match ? [match[1], match[2]] : null;
}

/**
 * Precomputes, once per landmark set, what each landmark and bone is.
 *
 * @param {string[]} landmarkNames
 * @param {{bones: Array<{a: number, b: number, kind: string, isPoseSubset: boolean}>}} topology -
 *   From `buildSkeletonTopology`.
 */
export function describeHand(landmarkNames, topology) {
  const index = new Map(landmarkNames.map((n, i) => [n, i]));
  const joints = landmarkNames.map((name, i) => {
    const part = handPart(name);
    return part
      ? { index: i, side: part[0], suffix: part[1], isArm: false, isAccent: isAccent(part[1]) }
      : { index: i, side: null, suffix: name, isArm: true, isAccent: false };
  });
  const bones = topology.bones.map((bone) => {
    const a = joints[bone.a];
    const b = joints[bone.b];
    const isArm = bone.kind === "bridge" || a.isArm || b.isArm;
    return {
      a: bone.a,
      b: bone.b,
      isArm,
      side: isArm ? null : a.side,
      radiusFactor: isArm ? ARM_RADII.bone : boneRadiusFactor(a.suffix, b.suffix),
    };
  });
  const palms = SIDES.map((side) => {
    const at = (suffix) => index.get(`${side}_hand_${suffix}`);
    const ring = [
      "thumb_cmc",
      "index_finger_mcp",
      "middle_finger_mcp",
      "ring_finger_mcp",
      "pinky_mcp",
    ].map(at);
    return {
      side,
      wrist: at("wrist"),
      middleMcp: at("middle_finger_mcp"),
      // A fan from the wrist through the five knuckle bases.
      triangles: ring.slice(0, -1).map((v, k) => [at("wrist"), v, ring[k + 1]]),
    };
  });
  return { joints, bones, palms };
}

/**
 * Computes one frame's geometry.
 *
 * Anything touching an untracked `(0, 0, 0)` landmark is hidden, so an
 * untracked hand (or arm point) isn't drawn at all. That's the "phantom
 * static hand" fix (RENDERING_UPGRADE_PLAN.md Section 5).
 *
 * @param {Array<[number, number, number]>} pose - One frame, data coordinates.
 * @param {ReturnType<typeof describeHand>} hand
 * @returns {{
 *   joints: Array<{position: [number, number, number], radius: number, visible: boolean, isArm: boolean, isAccent: boolean}>,
 *   bones: Array<{start: [number, number, number], end: [number, number, number], radius: number, visible: boolean, isArm: boolean}>,
 *   palms: Array<{side: string, visible: boolean, triangles: Array<Array<[number, number, number]>>}>,
 * }} World coordinates.
 */
export function computeHandGeometry(pose, hand) {
  const world = pose.map((point, i) => toWorld(point, hand.joints[i].isArm));
  const real = pose.map((point) => !isUndetectedPoint(point));

  const palmLength = {};
  for (const palm of hand.palms) {
    const measurable =
      palm.wrist !== undefined &&
      palm.middleMcp !== undefined &&
      real[palm.wrist] &&
      real[palm.middleMcp];
    const length = measurable ? distance(world[palm.wrist], world[palm.middleMcp]) : 0;
    palmLength[palm.side] = length > 1e-6 ? length : FALLBACK_PALM_LENGTH;
  }
  const trackedSides = hand.palms.filter((p) => p.wrist !== undefined && real[p.wrist]);
  const armScale =
    trackedSides.length > 0
      ? trackedSides.reduce((sum, p) => sum + palmLength[p.side], 0) / trackedSides.length
      : FALLBACK_PALM_LENGTH;

  const joints = hand.joints.map((j) => ({
    position: world[j.index],
    radius: j.isArm ? ARM_RADII.joint * armScale : jointRadiusFactor(j.suffix) * palmLength[j.side],
    visible: real[j.index],
    isArm: j.isArm,
    isAccent: j.isAccent,
  }));

  const bones = hand.bones.map((b) => ({
    start: world[b.a],
    end: world[b.b],
    radius: b.radiusFactor * (b.isArm ? armScale : palmLength[b.side]),
    visible: real[b.a] && real[b.b] && distance(world[b.a], world[b.b]) > 1e-9,
    isArm: b.isArm,
  }));

  const palms = hand.palms.map((palm) => {
    const indices = palm.triangles.flat();
    const complete = indices.every((i) => i !== undefined && real[i]);
    return {
      side: palm.side,
      visible: complete,
      triangles: complete ? palm.triangles.map((tri) => tri.map((i) => world[i])) : [],
    };
  });

  return { joints, bones, palms };
}
