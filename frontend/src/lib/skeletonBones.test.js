import { describe, expect, it } from "vitest";

import {
  computeContentBounds,
  computeFitTransform,
  filterOutPoseSubset,
  isUndetectedPoint,
  projectPoint,
  stepCamera,
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

describe("stepCamera", () => {
  const isPoseSubsetByIndex = [false, false, true, true]; // 2 hand points, 2 pose-subset

  it("eases toward a normal frame's real content, same as the original inline math", () => {
    const camera = { centerX: 0, centerY: 0, span: 1 };
    const pose = [
      [0.4, 0.4, 0],
      [0.6, 0.6, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const before = { ...camera };
    const result = stepCamera(camera, pose, isPoseSubsetByIndex, 0.18);

    const rawFit = computeFitTransform(
      computeContentBounds([filterOutPoseSubset(pose, isPoseSubsetByIndex)])
    );
    expect(result.centerX).toBeCloseTo(
      before.centerX + (rawFit.centerX - before.centerX) * 0.18,
      10
    );
    expect(result.centerY).toBeCloseTo(
      before.centerY + (rawFit.centerY - before.centerY) * 0.18,
      10
    );
    expect(result.span).toBeCloseTo(before.span + (rawFit.span - before.span) * 0.18, 10);
  });

  it("mutates and returns the same camera object", () => {
    const camera = { centerX: 0, centerY: 0, span: 1 };
    const pose = [
      [0.4, 0.4, 0],
      [0.6, 0.6, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const result = stepCamera(camera, pose, isPoseSubsetByIndex, 0.18);
    expect(result).toBe(camera);
  });

  it("holds the camera exactly steady on a frame with zero real hand points, instead of chasing the full-frame fallback", () => {
    // Both hand landmarks are the (0,0,0) sentinel -- the real, confirmed
    // shape of a cross-word transition frame between two differently-
    // handed signs (e.g. "about"'s last frame is left-hand-only,
    // "bathroom"'s first frame is right-hand-only, so `lerpPose` forces
    // *both* hands to zero for the whole transition -- see
    // `frontend/MULTIWORD_PLAN.md`'s "Known gotchas" and `stepCamera`'s
    // own docstring). Without this fix, `computeContentBounds` would fall
    // back to the full [0,1] frame here and the camera would snap toward
    // a huge span, then snap back -- a jarring zoom oscillation at every
    // such boundary.
    const camera = { centerX: 0.3, centerY: 0.7, span: 0.2 };
    const pose = [
      [0, 0, 0],
      [0, 0, 0],
      [0.5, 0.9, 0], // real pose-subset point -- irrelevant, must still be ignored
      [0, 0, 0],
    ];
    const result = stepCamera(camera, pose, isPoseSubsetByIndex, 0.18);
    expect(result).toEqual({ centerX: 0.3, centerY: 0.7, span: 0.2 });
  });

  it("resumes easing normally on the very next frame once real hand content returns", () => {
    const camera = { centerX: 0.3, centerY: 0.7, span: 0.2 };
    const degenerate = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    stepCamera(camera, degenerate, isPoseSubsetByIndex, 0.18);
    expect(camera).toEqual({ centerX: 0.3, centerY: 0.7, span: 0.2 }); // held

    const real = [
      [0.1, 0.1, 0],
      [0.2, 0.2, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    stepCamera(camera, real, isPoseSubsetByIndex, 0.18);
    // Eased away from the held position toward the new real content --
    // not still frozen, not snapped instantly either.
    expect(camera.centerX).not.toBe(0.3);
    expect(camera.centerX).toBeLessThan(0.3);
    expect(camera.centerX).toBeGreaterThan(0.15);
  });

  it("an alpha of 1 (reduced motion) snaps to real content, but still holds on a degenerate frame", () => {
    const camera = { centerX: 0.3, centerY: 0.7, span: 0.2 };
    const real = [
      [0.1, 0.1, 0],
      [0.2, 0.2, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    stepCamera(camera, real, isPoseSubsetByIndex, 1);
    const rawFit = computeFitTransform(
      computeContentBounds([
        [
          [0.1, 0.1, 0],
          [0.2, 0.2, 0],
        ],
      ])
    );
    expect(camera.centerX).toBeCloseTo(rawFit.centerX, 10);

    const degenerate = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const before = { ...camera };
    stepCamera(camera, degenerate, isPoseSubsetByIndex, 1);
    expect(camera).toEqual(before); // still held, even at alpha=1
  });
});
