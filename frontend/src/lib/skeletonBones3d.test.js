// Tests for skeletonBones.js's 3D camera-fit extension
// (frontend/RENDERING_UPGRADE_PLAN.md Section 4). Kept separate from
// skeletonBones.test.js so the existing 2D tests stay exactly as they were.
import { describe, expect, it } from "vitest";

import {
  DEPTH_SCALE,
  VIEW_3D,
  cameraPlacement3D,
  computeContentBounds,
  computeContentBounds3D,
  computeFitTransform,
  computeFitTransform3D,
  stepCamera3D,
  toWorld,
} from "./skeletonBones.js";

describe("computeContentBounds3D", () => {
  it("tracks depth alongside x and y, ignoring sentinel points", () => {
    const poses = [
      [
        [0.4, 0.5, -0.05],
        [0, 0, 0],
      ],
      [[0.6, 0.7, 0.02]],
    ];
    expect(computeContentBounds3D(poses)).toEqual({
      minX: 0.4,
      minY: 0.5,
      minZ: -0.05,
      maxX: 0.6,
      maxY: 0.7,
      maxZ: 0.02,
    });
  });

  it("falls back to the full frame at zero depth when nothing is real", () => {
    expect(computeContentBounds3D([[[0, 0, 0]]])).toEqual({
      minX: 0,
      minY: 0,
      minZ: 0,
      maxX: 1,
      maxY: 1,
      maxZ: 0,
    });
  });

  it("agrees with the 2D bounds on x and y", () => {
    const poses = [
      [
        [0.2, 0.3, -0.1],
        [0.5, 0.9, 0.1],
      ],
    ];
    const { minX, minY, maxX, maxY } = computeContentBounds3D(poses);
    expect(computeContentBounds(poses)).toEqual({ minX, minY, maxX, maxY });
  });
});

describe("computeFitTransform3D", () => {
  it("matches the 2D fit's center and span when depth is the smallest extent", () => {
    const bounds = { minX: 0.4, minY: 0.5, minZ: -0.01, maxX: 0.6, maxY: 0.6, maxZ: 0.01 };
    const fit2D = computeFitTransform(bounds);
    const fit3D = computeFitTransform3D(bounds);
    expect(fit3D.centerX).toBeCloseTo(fit2D.centerX);
    expect(fit3D.centerY).toBeCloseTo(fit2D.centerY);
    expect(fit3D.span).toBeCloseTo(fit2D.span);
    expect(fit3D.centerZ).toBeCloseTo(0);
  });

  it("grows the span to cover depth when depth is the largest extent", () => {
    const flat = computeFitTransform3D({
      minX: 0.5,
      minY: 0.5,
      minZ: 0,
      maxX: 0.55,
      maxY: 0.55,
      maxZ: 0,
    });
    const deep = computeFitTransform3D({
      minX: 0.5,
      minY: 0.5,
      minZ: -0.2,
      maxX: 0.55,
      maxY: 0.55,
      maxZ: 0,
    });
    expect(deep.span / flat.span).toBeCloseTo(0.2 / 0.05);
    expect(deep.centerZ).toBeCloseTo(-0.1);
  });
});

describe("stepCamera3D", () => {
  const isPoseSubset = [false, false, true];
  const pose = [
    [0.4, 0.4, -0.02],
    [0.6, 0.6, 0.02],
    [0.9, 0.1, -1.5], // pose-subset point: ignored by the fit
  ];
  const target = computeFitTransform3D(computeContentBounds3D([pose.slice(0, 2)]));

  it("eases every field, depth included, by alpha", () => {
    const camera = { centerX: 0, centerY: 0, centerZ: 1, span: 1 };
    stepCamera3D(camera, pose, isPoseSubset, 0.25);
    expect(camera.centerX).toBeCloseTo(target.centerX * 0.25);
    expect(camera.centerY).toBeCloseTo(target.centerY * 0.25);
    expect(camera.centerZ).toBeCloseTo(1 + (target.centerZ - 1) * 0.25);
    expect(camera.span).toBeCloseTo(1 + (target.span - 1) * 0.25);
  });

  it("snaps straight to the fit with alpha = 1 (reduced motion)", () => {
    const camera = { centerX: 0, centerY: 0, centerZ: 0, span: 1 };
    stepCamera3D(camera, pose, isPoseSubset, 1);
    expect(camera.centerX).toBeCloseTo(target.centerX);
    expect(camera.centerY).toBeCloseTo(target.centerY);
    expect(camera.centerZ).toBeCloseTo(target.centerZ);
    expect(camera.span).toBeCloseTo(target.span);
  });

  it("holds perfectly still on a frame with no real hand point", () => {
    const camera = { centerX: 0.3, centerY: 0.4, centerZ: -0.01, span: 0.2 };
    const before = { ...camera };
    const noHands = [
      [0, 0, 0],
      [0, 0, 0],
      [0.9, 0.1, -1.5], // a real arm point doesn't count
    ];
    const returned = stepCamera3D(camera, noHands, isPoseSubset, 0.5);
    expect(returned).toBe(camera);
    expect(camera).toEqual(before);
  });
});

describe("toWorld", () => {
  it("flips y up and z toward the viewer", () => {
    expect(toWorld([0.2, 0.3, -0.05])).toEqual([0.2, -0.3, 0.05 * DEPTH_SCALE]);
  });

  it("puts pose-subset points in the wrist plane (z = 0)", () => {
    expect(toWorld([0.2, 0.3, -1.8], true)).toEqual([0.2, -0.3, 0]);
  });
});

describe("cameraPlacement3D", () => {
  const fit = { centerX: 0.5, centerY: 0.4, centerZ: -0.02, span: 0.3 };

  it("aims at the fit's center in world space", () => {
    expect(cameraPlacement3D(fit).target).toEqual(toWorld([0.5, 0.4, -0.02]));
  });

  it("stands back exactly far enough for the span to fill the field of view", () => {
    const { target, position, distance } = cameraPlacement3D(fit);
    const fov = (VIEW_3D.fovDeg * Math.PI) / 180;
    expect(distance).toBeCloseTo(0.15 / Math.tan(fov / 2));
    const offset = position.map((p, i) => p - target[i]);
    expect(Math.hypot(...offset)).toBeCloseTo(distance);
  });

  it("views from the front, off to the right and above, per VIEW_3D", () => {
    const { target, position } = cameraPlacement3D(fit);
    const [dx, dy, dz] = position.map((p, i) => p - target[i]);
    expect(dz).toBeGreaterThan(0); // in front: toward the viewer
    expect(dx).toBeGreaterThan(0);
    expect(dy).toBeGreaterThan(0);
    expect((Math.atan2(dx, dz) * 180) / Math.PI).toBeCloseTo(VIEW_3D.yawDeg);
    expect((Math.asin(dy / Math.hypot(dx, dy, dz)) * 180) / Math.PI).toBeCloseTo(VIEW_3D.pitchDeg);
  });

  it("looks dead-on along +Z with zero yaw and pitch", () => {
    const { target, position } = cameraPlacement3D(fit, { fovDeg: 30, yawDeg: 0, pitchDeg: 0 });
    expect(position[0]).toBeCloseTo(target[0]);
    expect(position[1]).toBeCloseTo(target[1]);
    expect(position[2]).toBeGreaterThan(target[2]);
  });
});
