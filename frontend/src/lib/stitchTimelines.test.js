import { describe, expect, it } from "vitest";

import { DEFAULT_TARGET_FPS, stitchTimelines } from "./stitchTimelines.js";

/** A minimal single-landmark sequence with no dropped frames, so
 * `reconstructTimeline`'s dense output equals the raw frames unchanged --
 * keeps resampling math easy to predict (same convention
 * `reconstructTimeline.test.js` uses). */
function makeWord(word, { fps = 10, points }) {
  return {
    word,
    sequence: {
      gloss: word.toUpperCase(),
      fps,
      landmark_names: ["only_point"],
      frames: points.map((p) => [p]),
      source: `test:${word}`,
    },
    manifestEntry: {
      total_frames_decoded: points.length,
      dropped_frame_indices: [],
    },
  };
}

describe("stitchTimelines", () => {
  it("stitches two words at the default target fps, with a transition span at the boundary", () => {
    // Word "a": both frames exactly (0,0,0) -- the "unused hand" sentinel.
    // Word "b": real, non-zero points.
    const a = makeWord("a", {
      points: [
        [0, 0, 0],
        [0, 0, 0],
      ],
    });
    const b = makeWord("b", {
      points: [
        [1, 1, 1],
        [2, 2, 2],
      ],
    });

    const { timeline, wordBoundaries } = stitchTimelines([a, b], {
      targetFps: 10,
      transitionMs: 200,
    });

    // Output fps is always the single shared target rate, never either
    // word's own native fps (both happen to already be 10 here, but the
    // *mechanism* -- not coincidence -- is what's under test elsewhere).
    expect(timeline.fps).toBe(10);
    expect(timeline.frameDurationMs).toBe(100);

    // a: 2 resampled frames. transition: 200ms / 100ms-per-frame = 2
    // frames. b: 2 resampled frames. Total = 6.
    expect(timeline.frames).toHaveLength(6);
    expect(timeline.durationSeconds).toBeCloseTo(0.5); // (6-1) * 100ms

    // Word "a"'s own resampled content occupies frames 0-1, matching its
    // 2 real frames essentially unchanged (native fps == target fps).
    expect(wordBoundaries[0]).toEqual({
      word: "a",
      wordIndex: 0,
      startFrameIndex: 0,
      endFrameIndex: 1,
    });
    expect(timeline.frames[0].pose).toEqual([[0, 0, 0]]);
    expect(timeline.frames[1].pose).toEqual([[0, 0, 0]]);

    // Word "b"'s span includes the incoming transition (frames 2-5),
    // per stitchTimelines' documented "transition belongs to the word it
    // leads into" convention -- spans are contiguous, no gap frame belongs
    // to neither word.
    expect(wordBoundaries[1]).toEqual({
      word: "b",
      wordIndex: 1,
      startFrameIndex: 2,
      endFrameIndex: 5,
    });

    // The two transition frames (indices 2-3): lerpPose's zero-sentinel
    // guard means *any* interpolation against word a's exact (0,0,0)
    // endpoint stays (0,0,0) for the whole span, not a lerped path toward
    // word b's real values -- confirmed here, not assumed (this is the
    // same behavior documented in MULTIWORD_PLAN.md's "Known gotchas").
    expect(timeline.frames[2].pose).toEqual([[0, 0, 0]]);
    expect(timeline.frames[2].state).toBe("interpolated");
    expect(timeline.frames[2].dimmed).toBe(false);
    expect(timeline.frames[3].pose).toEqual([[0, 0, 0]]);

    // Word "b"'s own resampled content (frames 4-5) is real, unblended.
    expect(timeline.frames[4].pose).toEqual([[1, 1, 1]]);
    expect(timeline.frames[5].pose).toEqual([[2, 2, 2]]);

    // wordBoundaries spans are non-overlapping and cover the whole
    // timeline: a ends where b's (transition-inclusive) span begins.
    expect(wordBoundaries[0].endFrameIndex + 1).toBe(wordBoundaries[1].startFrameIndex);
    expect(wordBoundaries[1].endFrameIndex).toBe(timeline.frames.length - 1);
  });

  it("produces a real, non-zero lerp between two ordinary (non-sentinel) endpoints", () => {
    const a = makeWord("a", { points: [[0, 0, 0.5]] }); // single-frame word
    const b = makeWord("b", { points: [[10, 10, 10.5]] });

    const { timeline } = stitchTimelines([a, b], { targetFps: 10, transitionMs: 100 });

    // 1 frame (a) + 1 transition frame (100ms / 100ms-per-frame, at least
    // 1) + 1 frame (b) = 3.
    expect(timeline.frames).toHaveLength(3);
    expect(timeline.frames[0].pose).toEqual([[0, 0, 0.5]]);
    expect(timeline.frames[2].pose).toEqual([[10, 10, 10.5]]);
    // The single transition frame is the midpoint (t = 1/(1+1) = 0.5).
    expect(timeline.frames[1].pose[0][0]).toBeCloseTo(5);
    expect(timeline.frames[1].pose[0][2]).toBeCloseTo(5.5);
    expect(timeline.frames[1].state).toBe("interpolated");
  });

  it("stitches three words with correct, non-overlapping, gapless wordBoundaries spans", () => {
    const a = makeWord("a", { points: [[0, 0, 0]] });
    const b = makeWord("b", { points: [[1, 1, 1]] });
    const c = makeWord("c", { points: [[2, 2, 2]] });

    const { timeline, wordBoundaries } = stitchTimelines([a, b, c], {
      targetFps: 10,
      transitionMs: 100,
    });

    expect(wordBoundaries).toHaveLength(3);
    expect(wordBoundaries[0].wordIndex).toBe(0);
    expect(wordBoundaries[1].wordIndex).toBe(1);
    expect(wordBoundaries[2].wordIndex).toBe(2);

    // Spans are contiguous (each word's start is exactly the previous
    // word's end + 1) and together cover the entire merged timeline.
    expect(wordBoundaries[0].startFrameIndex).toBe(0);
    expect(wordBoundaries[1].startFrameIndex).toBe(wordBoundaries[0].endFrameIndex + 1);
    expect(wordBoundaries[2].startFrameIndex).toBe(wordBoundaries[1].endFrameIndex + 1);
    expect(wordBoundaries[2].endFrameIndex).toBe(timeline.frames.length - 1);

    // No overlap: every span's start is <= its own end, and each word's
    // end is strictly before the next word's start minus one already
    // covers non-overlap by construction (contiguous spans can't overlap).
    for (const b2 of wordBoundaries) {
      expect(b2.startFrameIndex).toBeLessThanOrEqual(b2.endFrameIndex);
    }
  });

  it("uses DEFAULT_TARGET_FPS and DEFAULT_TRANSITION_MS when no options are given", () => {
    const a = makeWord("a", { fps: 24, points: [[0, 0, 0]] });
    const b = makeWord("b", { fps: 25, points: [[1, 1, 1]] });

    const { timeline } = stitchTimelines([a, b]);

    expect(timeline.fps).toBe(DEFAULT_TARGET_FPS);
    expect(timeline.frameDurationMs).toBeCloseTo(1000 / DEFAULT_TARGET_FPS);
  });

  it("resamples differing native fps values onto one shared output fps", () => {
    // Word "a" at 24fps spans 1 second (24 real frames -> resampled to the
    // 30fps target should yield 31 frames, k=0..30).
    const aPoints = Array.from({ length: 24 }, (_, i) => [i, i, i]);
    const a = makeWord("a", { fps: 24, points: aPoints });
    const b = makeWord("b", { fps: 25, points: [[100, 100, 100]] });

    const { timeline, wordBoundaries } = stitchTimelines([a, b], {
      targetFps: 30,
      transitionMs: 0,
    });

    expect(timeline.fps).toBe(30);
    // Word a's own span at 30fps: round(duration * 30) + 1 frames, where
    // duration = 23/24s (23 gaps between 24 frames at 24fps).
    const expectedAFrames = Math.round((23 / 24) * 30) + 1;
    expect(wordBoundaries[0].endFrameIndex - wordBoundaries[0].startFrameIndex + 1).toBe(
      expectedAFrames
    );
    // First and last frame of word a's resampled span still land exactly
    // on its real endpoints (native fps and target fps both start at t=0).
    expect(timeline.frames[0].pose).toEqual([[0, 0, 0]]);
    expect(timeline.frames[wordBoundaries[0].endFrameIndex].pose).toEqual([[23, 23, 23]]);
  });

  it("returns an empty timeline and no boundaries for an empty input", () => {
    const { timeline, wordBoundaries } = stitchTimelines([], { targetFps: 30 });
    expect(timeline.frames).toEqual([]);
    expect(timeline.durationSeconds).toBe(0);
    expect(timeline.fps).toBe(30);
    expect(wordBoundaries).toEqual([]);
  });

  it("handles a single word with no transition frames at all", () => {
    const a = makeWord("a", {
      points: [
        [0, 0, 0],
        [1, 1, 1],
      ],
    });
    const { timeline, wordBoundaries } = stitchTimelines([a], { targetFps: 10 });

    expect(timeline.frames).toHaveLength(2);
    expect(wordBoundaries).toEqual([
      { word: "a", wordIndex: 0, startFrameIndex: 0, endFrameIndex: 1 },
    ]);
  });
});
