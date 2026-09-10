import { describe, expect, it } from "vitest";

import { GAP_INTERPOLATION_THRESHOLD_FRAMES, reconstructTimeline } from "./reconstructTimeline.js";

/** Builds a minimal single-landmark sequence, for arithmetic-friendly
 * assertions. Point at kept-frame k is [k, k, k] unless overridden. */
function makeSequence({ fps = 30, points } = {}) {
  return {
    fps,
    landmark_names: ["only_point"],
    frames: points.map((p) => [p]),
    source: "test:fixture",
  };
}

function manifest({ totalFramesDecoded, droppedIndices }) {
  return {
    total_frames_decoded: totalFramesDecoded,
    dropped_frame_indices: droppedIndices,
  };
}

describe("reconstructTimeline", () => {
  it("passes through frames unchanged when there are no dropped frames", () => {
    const sequence = makeSequence({
      fps: 30,
      points: [
        [0, 0, 0.1],
        [1, 1, 1.1],
        [2, 2, 2.1],
      ],
    });
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 3, droppedIndices: [] })
    );

    expect(timeline.frames).toHaveLength(3);
    expect(timeline.frames.every((f) => f.state === "real")).toBe(true);
    expect(timeline.frames.every((f) => f.dimmed === false)).toBe(true);
    expect(timeline.frames.map((f) => f.time)).toEqual([0, 1 / 30, 2 / 30]);
    expect(timeline.frames.map((f) => f.pose)).toEqual([
      [[0, 0, 0.1]],
      [[1, 1, 1.1]],
      [[2, 2, 2.1]],
    ]);
    expect(timeline.durationSeconds).toBeCloseTo(2 / 30);
  });

  it("linearly interpolates a gap at exactly the threshold size", () => {
    // Two kept frames (original index 0 and 7), with a gap of exactly
    // GAP_INTERPOLATION_THRESHOLD_FRAMES (6) frames between them: indices
    // 1..6 dropped. Kept frame 0 = point 0, kept frame 1 (orig idx 7) = point 7.
    expect(GAP_INTERPOLATION_THRESHOLD_FRAMES).toBe(6);
    // Points start at 1, not 0 -- a real (0,0,0) landmark point is treated
    // as the "not detected" sentinel (see the dedicated test below), so a
    // fixture testing ordinary numeric lerp math must avoid it.
    const sequence = makeSequence({
      fps: 10,
      points: [
        [1, 1, 1],
        [8, 8, 8],
      ],
    });
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 8, droppedIndices: [1, 2, 3, 4, 5, 6] })
    );

    expect(timeline.frames).toHaveLength(8);
    expect(timeline.frames[0].state).toBe("real");
    expect(timeline.frames[7].state).toBe("real");
    for (let i = 1; i <= 6; i += 1) {
      expect(timeline.frames[i].state).toBe("interpolated");
      expect(timeline.frames[i].dimmed).toBe(false);
      // Linear: point value should equal 1 + original-index-offset (1..8 lerp).
      expect(timeline.frames[i].pose[0][0]).toBeCloseTo(1 + i);
      expect(timeline.frames[i].time).toBeCloseTo(i / 10);
    }
  });

  it("holds and dims a gap one frame above the threshold", () => {
    // Gap of 7 dropped frames (indices 1..7) between kept frame 0 and kept
    // frame at original index 8.
    const sequence = makeSequence({
      fps: 10,
      points: [
        [0, 0, 0],
        [100, 100, 100],
      ],
    });
    const timeline = reconstructTimeline(
      sequence,
      manifest({
        totalFramesDecoded: 9,
        droppedIndices: [1, 2, 3, 4, 5, 6, 7],
      })
    );

    expect(timeline.frames).toHaveLength(9);
    expect(timeline.frames[0].state).toBe("real");
    expect(timeline.frames[8].state).toBe("real");
    for (let i = 1; i <= 7; i += 1) {
      expect(timeline.frames[i].state).toBe("held");
      expect(timeline.frames[i].dimmed).toBe(true);
      // Held pose is exactly the last known real pose, not invented motion.
      expect(timeline.frames[i].pose).toEqual([[0, 0, 0]]);
    }
  });

  it("handles multiple consecutive gaps of different sizes independently", () => {
    // Kept original indices: 0, 3 (gap=2, short), 12 (gap=8, long), 14 (gap=1, short).
    const sequence = makeSequence({
      fps: 20,
      points: [
        [1, 1, 1],
        [3, 3, 3],
        [12, 12, 12],
        [14, 14, 14],
      ],
    });
    const dropped = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13];
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 15, droppedIndices: dropped })
    );

    expect(timeline.frames).toHaveLength(15);
    const states = timeline.frames.map((f) => f.state);
    expect(states[0]).toBe("real");
    expect(states[1]).toBe("interpolated");
    expect(states[2]).toBe("interpolated");
    expect(states[3]).toBe("real");
    for (let i = 4; i <= 11; i += 1) {
      expect(states[i]).toBe("held");
    }
    expect(states[12]).toBe("real");
    expect(states[13]).toBe("interpolated");
    expect(states[14]).toBe("real");
  });

  it("never interpolates/holds at index 0 or the last index (leading/trailing already trimmed)", () => {
    const sequence = makeSequence({
      fps: 15,
      points: [
        [5, 5, 5],
        [9, 9, 9],
      ],
    });
    // A short gap right after the first kept frame, and right before the
    // last -- exercises the "gap at the very start/end after trimming"
    // edge case: the trimming has already happened upstream (frames[0] is
    // the first *kept* frame), so index 0 of the timeline must always be
    // "real", never itself part of a gap.
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 5, droppedIndices: [1, 2, 3] })
    );
    expect(timeline.frames[0].state).toBe("real");
    expect(timeline.frames[timeline.frames.length - 1].state).toBe("real");
  });

  it("treats a (0,0,0) endpoint as 'not detected' rather than lerping into/out of the origin", () => {
    const sequence = {
      fps: 10,
      landmark_names: ["shoulder"],
      frames: [
        [[0.5, 0.5, 0.5]],
        [[0, 0, 0]], // undetected pose-subset landmark sentinel
      ],
      source: "test:fixture",
    };
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 3, droppedIndices: [1] })
    );
    // The interpolated frame between a real point and the (0,0,0) sentinel
    // must itself be (0,0,0), not a lerped path toward the origin corner.
    expect(timeline.frames[1].pose).toEqual([[0, 0, 0]]);
  });

  it("throws when the manifest's drop record doesn't match the sequence's kept-frame count", () => {
    const sequence = makeSequence({
      fps: 30,
      points: [
        [0, 0, 0],
        [1, 1, 1],
      ],
    });
    expect(() =>
      reconstructTimeline(
        sequence,
        manifest({ totalFramesDecoded: 5, droppedIndices: [] }) // implies 5 kept, but only 2 frames
      )
    ).toThrow(/mismatch/);
  });

  it("returns an empty timeline for a zero-frame sequence", () => {
    const sequence = makeSequence({ fps: 30, points: [] });
    const timeline = reconstructTimeline(
      sequence,
      manifest({ totalFramesDecoded: 0, droppedIndices: [] })
    );
    expect(timeline.frames).toEqual([]);
    expect(timeline.durationSeconds).toBe(0);
  });

  describe("real fixture: 'phone' (11/68 frames kept, the real low-confidence word)", () => {
    // Real dropped_frame_indices, total_frames_decoded, and fps read
    // directly from pose_library/data/poses/manifest.json and phone.json.
    // Frame *content* is synthetic (kept-frame k has point value k) purely
    // so interpolation math is easy to assert, but the gap structure below
    // -- the leading run, the single big 29-frame interior gap, and the
    // trailing run -- is phone's real, on-disk shape.
    const PHONE_FPS = 25.0;
    const PHONE_TOTAL_FRAMES_DECODED = 68;
    const PHONE_DROPPED_FRAME_INDICES = [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59,
      60, 61, 62, 63, 64, 65, 66, 67,
    ];

    it("reconstructs the real gap structure: 8 kept, 1 long held gap, 3 kept", () => {
      const points = Array.from({ length: 11 }, (_, k) => [k, k, k]);
      const sequence = makeSequence({ fps: PHONE_FPS, points });
      const timeline = reconstructTimeline(
        sequence,
        manifest({
          totalFramesDecoded: PHONE_TOTAL_FRAMES_DECODED,
          droppedIndices: PHONE_DROPPED_FRAME_INDICES,
        })
      );

      // Leading drops (0-13) and trailing drops (54-67) are trimmed away:
      // the timeline spans original index 14..53 inclusive = 40 frames.
      expect(timeline.frames).toHaveLength(40);

      const realCount = timeline.frames.filter((f) => f.state === "real").length;
      const heldCount = timeline.frames.filter((f) => f.state === "held").length;
      const interpolatedCount = timeline.frames.filter((f) => f.state === "interpolated").length;
      expect(realCount).toBe(11);
      expect(heldCount).toBe(29);
      expect(interpolatedCount).toBe(0);

      // First 8 timeline entries (orig idx 14..21) are the real leading run.
      for (let i = 0; i < 8; i += 1) {
        expect(timeline.frames[i].state).toBe("real");
      }
      // Then a single 29-frame held/dimmed gap (orig idx 22..50).
      for (let i = 8; i < 37; i += 1) {
        expect(timeline.frames[i].state).toBe("held");
        expect(timeline.frames[i].dimmed).toBe(true);
        // Held at the last real pose before the gap (kept frame index 7 -> point [7,7,7]).
        expect(timeline.frames[i].pose).toEqual([[7, 7, 7]]);
      }
      // Then the real trailing run of 3 (orig idx 51..53).
      for (let i = 37; i < 40; i += 1) {
        expect(timeline.frames[i].state).toBe("real");
      }

      expect(timeline.fps).toBe(PHONE_FPS);
      expect(timeline.durationSeconds).toBeCloseTo((53 - 14) / PHONE_FPS);
    });
  });
});
