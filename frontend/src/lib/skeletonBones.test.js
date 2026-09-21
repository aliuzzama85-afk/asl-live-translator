import { describe, expect, it } from "vitest";

import {
  computeContentBounds,
  computeFitTransform,
  filterOutPoseSubset,
  isUndetectedPoint,
  projectPoint,
} from "./skeletonBones.js";

describe("computeContentBounds", () => {
  it("computes the bounding box across all real points in all poses", () => {
    const poses = [
      [
        [0.4, 0.5, 0],
        [0.6, 0.7, 0],
      ],
      [
        [0.3, 0.55, 0],
        [0.65, 0.6, 0],
      ],
    ];
    expect(computeContentBounds(poses)).toEqual({
      minX: 0.3,
      minY: 0.5,
      maxX: 0.65,
      maxY: 0.7,
    });
  });

  it("ignores (0,0,0) undetected-sentinel points", () => {
    const poses = [
      [
        [0, 0, 0],
        [0.5, 0.5, 0],
      ],
      [
        [0, 0, 0],
        [0.55, 0.52, 0],
      ],
    ];
    const bounds = computeContentBounds(poses);
    expect(bounds).toEqual({ minX: 0.5, minY: 0.5, maxX: 0.55, maxY: 0.52 });
  });

  it("falls back to the full [0,1] square when every point is undetected", () => {
    const poses = [[[0, 0, 0]], [[0, 0, 0]]];
    expect(computeContentBounds(poses)).toEqual({ minX: 0, minY: 0, maxX: 1, maxY: 1 });
  });

  it("falls back to the full [0,1] square for an empty pose list", () => {
    expect(computeContentBounds([])).toEqual({ minX: 0, minY: 0, maxX: 1, maxY: 1 });
  });
});

describe("computeFitTransform", () => {
  it("centers on the bounding box midpoint", () => {
    const transform = computeFitTransform({ minX: 0.2, minY: 0.4, maxX: 0.4, maxY: 0.5 });
    expect(transform.centerX).toBeCloseTo(0.3, 10);
    expect(transform.centerY).toBeCloseTo(0.45, 10);
  });

  it("uses the larger of width/height (plus padding) so aspect isn't distorted", () => {
    // width=0.2, height=0.1 -- span must be driven by the larger (width).
    const transform = computeFitTransform({ minX: 0, minY: 0, maxX: 0.2, maxY: 0.1 });
    expect(transform.span).toBeGreaterThan(0.2);
    expect(transform.span).toBeCloseTo(0.2 * 1.36, 5); // 1 + 0.18*2 padding
  });

  it("never produces a zero span for a degenerate (single-point) bounding box", () => {
    const transform = computeFitTransform({ minX: 0.5, minY: 0.5, maxX: 0.5, maxY: 0.5 });
    expect(transform.span).toBeGreaterThan(0);
    expect(Number.isFinite(transform.span)).toBe(true);
  });
});

describe("projectPoint", () => {
  it("maps the transform's center to the middle of the canvas", () => {
    const transform = { centerX: 0.5, centerY: 0.5, span: 0.5 };
    expect(projectPoint([0.5, 0.5, 0], transform, 300)).toEqual([150, 150]);
  });

  it("maps a point at the edge of the fitted span to the canvas edge", () => {
    const transform = { centerX: 0.5, centerY: 0.5, span: 0.4 };
    // 0.5 + 0.2 is exactly the right/bottom edge of the fitted window.
    const [x, y] = projectPoint([0.7, 0.7, 0], transform, 300);
    expect(x).toBeCloseTo(300, 5);
    expect(y).toBeCloseTo(300, 5);
  });

  it("zooms in: a small real bounding box maps to most of the canvas, not a small corner", () => {
    // A hand spanning only 15% of the source frame (the real-world case
    // this fix targets, per PLAN.md's legibility note).
    const bounds = computeContentBounds([
      [
        [0.4, 0.4, 0],
        [0.55, 0.55, 0],
      ],
    ]);
    const transform = computeFitTransform(bounds);
    const [x1, y1] = projectPoint([0.4, 0.4, 0], transform, 300);
    const [x2, y2] = projectPoint([0.55, 0.55, 0], transform, 300);
    const onScreenSpan = Math.hypot(x2 - x1, y2 - y1);
    // Un-fitted, this 0.15-unit diagonal would render as ~0.15*300*sqrt(2)
    // =~ 64px on a 300px canvas. Fitted, it should fill most of the frame.
    expect(onScreenSpan).toBeGreaterThan(150);
  });
});

describe("isUndetectedPoint", () => {
  it("is true only for the exact (0,0,0) sentinel", () => {
    expect(isUndetectedPoint([0, 0, 0])).toBe(true);
    expect(isUndetectedPoint([0, 0, 0.0001])).toBe(false);
    expect(isUndetectedPoint([0.0, 0.0, -0.0])).toBe(true);
  });
});

describe("filterOutPoseSubset", () => {
  it("drops pose-subset points, keeping hand points in order", () => {
    const pose = [
      [0.1, 0.1, 0], // hand
      [0.2, 0.2, 0], // pose subset
      [0.3, 0.3, 0], // hand
    ];
    const isPoseSubsetByIndex = [false, true, false];
    expect(filterOutPoseSubset(pose, isPoseSubsetByIndex)).toEqual([
      [0.1, 0.1, 0],
      [0.3, 0.3, 0],
    ]);
  });

  it("prevents a raised/extended hand's arm chain from defeating the fit", () => {
    // A real shape this fix targets: hand landmarks clustered near the top
    // of the frame, but the shoulder/elbow/wrist chain stretches across
    // nearly the whole frame (e.g. a hand raised to the head, as in the
    // real "phone" extraction).
    const pose = [
      [0.45, 0.1, 0], // hand landmark
      [0.46, 0.11, 0], // hand landmark
      [0.5, 0.95, 0], // pose subset: shoulder, far away
      [0.9, 0.9, 0], // pose subset: elbow, far away
    ];
    const isPoseSubsetByIndex = [false, false, true, true];

    const fullBounds = computeContentBounds([pose]);
    const handOnlyBounds = computeContentBounds([filterOutPoseSubset(pose, isPoseSubsetByIndex)]);

    const fullSpan = Math.max(fullBounds.maxX - fullBounds.minX, fullBounds.maxY - fullBounds.minY);
    const handSpan = Math.max(
      handOnlyBounds.maxX - handOnlyBounds.minX,
      handOnlyBounds.maxY - handOnlyBounds.minY
    );
    expect(handSpan).toBeLessThan(fullSpan / 10);
  });
});
