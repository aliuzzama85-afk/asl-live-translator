import { describe, expect, it } from "vitest";

import {
  ARM_RADII,
  BONE_TAPER,
  FALLBACK_PALM_LENGTH,
  HAND_RADII,
  computeHandGeometry,
  describeHand,
} from "./handGeometry.js";
import { buildSkeletonTopology, toWorld } from "./skeletonBones.js";

const HAND_SUFFIXES = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_finger_mcp",
  "index_finger_pip",
  "index_finger_dip",
  "index_finger_tip",
  "middle_finger_mcp",
  "middle_finger_pip",
  "middle_finger_dip",
  "middle_finger_tip",
  "ring_finger_mcp",
  "ring_finger_pip",
  "ring_finger_dip",
  "ring_finger_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
];
const POSE_SUBSET = [
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
];
// The same 48-landmark layout as the real library files.
const LANDMARK_NAMES = [
  ...HAND_SUFFIXES.map((s) => `left_hand_${s}`),
  ...HAND_SUFFIXES.map((s) => `right_hand_${s}`),
  ...POSE_SUBSET,
];
const TOPOLOGY = buildSkeletonTopology(LANDMARK_NAMES);
const HAND = describeHand(LANDMARK_NAMES, TOPOLOGY);
const indexOf = (name) => LANDMARK_NAMES.indexOf(name);

/** An upright synthetic hand: wrist at (cx, cy), each finger a column of
 * joints going up, middle knuckle exactly `palm` above the wrist. */
function handPoints(cx, cy, palm) {
  return HAND_SUFFIXES.map((suffix) => {
    if (suffix === "wrist") return [cx, cy, 0];
    const fingers = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"];
    const finger = fingers.findIndex((f) => suffix.startsWith(f));
    const joint = ["cmc", "mcp", "pip", "dip", "ip", "tip"].findIndex((j) =>
      suffix.endsWith(`_${j}`)
    );
    const x = cx + (finger - 2) * palm * 0.25;
    // Knuckles (mcp, and the thumb's cmc) sit `palm` above the wrist.
    const level = suffix === "thumb_cmc" ? 0.5 : suffix.endsWith("_mcp") ? 1 : 1 + joint * 0.3;
    // Knuckles at z = 0 (so palm length is exactly `palm`); fingers curl forward.
    return [x, cy - palm * level, joint <= 1 ? 0 : -0.01 * joint];
  });
}

function makePose({ left = handPoints(0.3, 0.6, 0.1), right = handPoints(0.7, 0.6, 0.1) } = {}) {
  const arm = [
    [0.3, 0.3, -1.2],
    [0.7, 0.3, -1.1],
    [0.25, 0.8, -0.9],
    [0.75, 0.8, -0.8],
    [0.3, 0.65, -0.5],
    [0.7, 0.65, -0.4],
  ];
  return [...left, ...right, ...arm];
}

const UNTRACKED_HAND = HAND_SUFFIXES.map(() => [0, 0, 0]);

describe("describeHand", () => {
  it("classifies hand landmarks by side and the pose subset as arm", () => {
    const leftTip = HAND.joints[indexOf("left_hand_index_finger_tip")];
    expect(leftTip).toMatchObject({ side: "left", suffix: "index_finger_tip", isArm: false });
    expect(HAND.joints[indexOf("right_elbow")]).toMatchObject({ side: null, isArm: true });
  });

  it("treats shoulder/elbow bones and the hand-to-arm bridges as arm bones", () => {
    const armBones = HAND.bones.filter((b) => b.isArm);
    // 5 shoulder/elbow/wrist bones + 2 bridges into the hand wrists.
    expect(armBones).toHaveLength(7);
    expect(HAND.bones.filter((b) => !b.isArm)).toHaveLength(46);
  });

  it("builds each palm as a 4-triangle fan from the wrist through the knuckle bases", () => {
    const left = HAND.palms.find((p) => p.side === "left");
    expect(left.triangles).toHaveLength(4);
    for (const triangle of left.triangles) {
      expect(triangle[0]).toBe(indexOf("left_hand_wrist"));
    }
    expect(left.triangles[0][1]).toBe(indexOf("left_hand_thumb_cmc"));
    expect(left.triangles[3][2]).toBe(indexOf("left_hand_pinky_mcp"));
  });
});

describe("computeHandGeometry", () => {
  it("places every joint at its world position, arms flattened to the wrist plane", () => {
    const pose = makePose();
    const geometry = computeHandGeometry(pose, HAND);
    const tip = indexOf("right_hand_index_finger_tip");
    expect(geometry.joints[tip].position).toEqual(toWorld(pose[tip]));
    const elbow = indexOf("left_elbow");
    expect(geometry.joints[elbow].position).toEqual([pose[elbow][0], -pose[elbow][1], 0]);
  });

  it("runs each bone from its topology start to end", () => {
    const pose = makePose();
    const geometry = computeHandGeometry(pose, HAND);
    HAND.bones.forEach((bone, i) => {
      expect(geometry.bones[i].start).toBe(geometry.joints[bone.a].position);
      expect(geometry.bones[i].end).toBe(geometry.joints[bone.b].position);
    });
  });

  it("scales hand thickness with that hand's own palm length", () => {
    const small = computeHandGeometry(makePose({ right: handPoints(0.7, 0.6, 0.05) }), HAND);
    const large = computeHandGeometry(makePose({ right: handPoints(0.7, 0.6, 0.2) }), HAND);
    const wrist = indexOf("right_hand_wrist");
    expect(small.joints[wrist].radius).toBeCloseTo(HAND_RADII.wrist * 0.05, 12);
    expect(large.joints[wrist].radius).toBeCloseTo(HAND_RADII.wrist * 0.2, 12);
    // The other hand, unchanged, keeps its own scale.
    const leftWrist = indexOf("left_hand_wrist");
    expect(small.joints[leftWrist].radius).toBeCloseTo(HAND_RADII.wrist * 0.1, 12);
  });

  it("falls back to the library-median palm length when it can't be measured", () => {
    const right = handPoints(0.7, 0.6, 0.1);
    right[HAND_SUFFIXES.indexOf("middle_finger_mcp")] = [0, 0, 0];
    const geometry = computeHandGeometry(makePose({ right }), HAND);
    expect(geometry.joints[indexOf("right_hand_wrist")].radius).toBeCloseTo(
      HAND_RADII.wrist * FALLBACK_PALM_LENGTH,
      12
    );
  });

  it("draws arms thinner than fingers", () => {
    const geometry = computeHandGeometry(makePose(), HAND);
    const armBone = geometry.bones.find((b) => b.isArm);
    expect(armBone.radius).toBeCloseTo(ARM_RADII.bone * 0.1, 12);
    expect(ARM_RADII.bone).toBeLessThanOrEqual(HAND_RADII.fingerBone);
  });

  it("keeps every joint at least 1.2x the radius of the bone ends it joins", () => {
    // Near-equal radii made the sphere/cylinder seam z-fight in the browser.
    const geometry = computeHandGeometry(makePose(), HAND);
    HAND.bones.forEach((bone, i) => {
      const { radius } = geometry.bones[i];
      expect(geometry.joints[bone.a].radius).toBeGreaterThanOrEqual(1.2 * radius - 1e-12);
      expect(geometry.joints[bone.b].radius).toBeGreaterThanOrEqual(
        1.2 * radius * BONE_TAPER - 1e-12
      );
    });
  });

  it("hides an untracked hand entirely: no phantom joints, bones, or palm", () => {
    const geometry = computeHandGeometry(makePose({ left: UNTRACKED_HAND }), HAND);
    const leftJoints = geometry.joints.filter((_, i) => HAND.joints[i].side === "left");
    expect(leftJoints.every((j) => !j.visible)).toBe(true);
    const leftBones = geometry.bones.filter((_, i) => HAND.bones[i].side === "left");
    expect(leftBones.every((b) => !b.visible)).toBe(true);
    expect(geometry.palms.find((p) => p.side === "left").visible).toBe(false);
    // The bridge from the (tracked) arm wrist into the missing hand goes too.
    const bridge = HAND.bones.findIndex((b) => b.b === indexOf("left_hand_wrist") && b.isArm);
    expect(geometry.bones[bridge].visible).toBe(false);
    // The tracked hand is untouched.
    const rightJoints = geometry.joints.filter((_, i) => HAND.joints[i].side === "right");
    expect(rightJoints.every((j) => j.visible)).toBe(true);
    expect(geometry.palms.find((p) => p.side === "right").visible).toBe(true);
  });

  it("hides just the pieces touching one untracked landmark", () => {
    const right = handPoints(0.7, 0.6, 0.1);
    right[HAND_SUFFIXES.indexOf("pinky_mcp")] = [0, 0, 0];
    const geometry = computeHandGeometry(makePose({ right }), HAND);
    const pinkyMcp = indexOf("right_hand_pinky_mcp");
    expect(geometry.joints[pinkyMcp].visible).toBe(false);
    HAND.bones.forEach((bone, i) => {
      const touches = bone.a === pinkyMcp || bone.b === pinkyMcp;
      expect(geometry.bones[i].visible).toBe(!touches);
    });
    // The palm fan needs every knuckle base, so it's hidden rather than torn.
    expect(geometry.palms.find((p) => p.side === "right").visible).toBe(false);
  });

  it("hides an untracked arm point and its bones", () => {
    const pose = makePose();
    pose[indexOf("right_elbow")] = [0, 0, 0];
    const geometry = computeHandGeometry(pose, HAND);
    const elbow = indexOf("right_elbow");
    expect(geometry.joints[elbow].visible).toBe(false);
    const elbowBones = HAND.bones
      .map((bone, i) => ({ bone, i }))
      .filter(({ bone }) => bone.a === elbow || bone.b === elbow);
    expect(elbowBones).toHaveLength(2);
    for (const { i } of elbowBones) expect(geometry.bones[i].visible).toBe(false);
  });

  it("hides a zero-length bone (two landmarks at the same point)", () => {
    const right = handPoints(0.7, 0.6, 0.1);
    right[HAND_SUFFIXES.indexOf("index_finger_tip")] = [
      ...right[HAND_SUFFIXES.indexOf("index_finger_dip")],
    ];
    const geometry = computeHandGeometry(makePose({ right }), HAND);
    const dip = indexOf("right_hand_index_finger_dip");
    const tip = indexOf("right_hand_index_finger_tip");
    const i = HAND.bones.findIndex((b) => b.a === dip && b.b === tip);
    expect(geometry.bones[i].visible).toBe(false);
  });

  it("gives the palm world-space triangles when the hand is fully tracked", () => {
    const pose = makePose();
    const palm = computeHandGeometry(pose, HAND).palms.find((p) => p.side === "right");
    expect(palm.triangles).toHaveLength(4);
    expect(palm.triangles[0][0]).toEqual(toWorld(pose[indexOf("right_hand_wrist")]));
  });

  it("marks only fingertips and wrists as accent joints", () => {
    const geometry = computeHandGeometry(makePose(), HAND);
    const accents = LANDMARK_NAMES.filter((_, i) => geometry.joints[i].isAccent);
    expect(accents).toHaveLength(12); // 5 tips + 1 wrist, per hand
    expect(accents.every((n) => n.endsWith("_tip") || n.endsWith("_hand_wrist"))).toBe(true);
  });
});
