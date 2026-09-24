import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { App } from "./App.jsx";

/**
 * Integration test for the App shell, per `frontend/PLAN.md` Section 6's
 * explicit test bar: "smooth, correctly-timed playback of a representative
 * sample of the 118 real words -- including at least one `low_confidence`
 * word and one guaranteed-OOV word -- not just a single happy-path word."
 *
 * The two manifest entries below are copied **verbatim** from the real
 * `pose_library/data/poses/manifest.json` on disk (queried directly, not
 * assumed) -- every field, including the real `dropped_frame_indices`
 * arrays, is the actual data:
 * - "about": low_confidence: false, 68/116 frames kept -- the clean case.
 * - "phone": low_confidence: true, 11/68 frames kept, a real 29-frame
 *   mid-clip tracking gap -- matches the gap analysis's cited low-confidence
 *   example, reverified against the manifest before use.
 *
 * `xyzzynotasign` was confirmed absent from all 118 real manifest keys
 * before use -- a guaranteed miss, not a plausible-but-untested word.
 *
 * Only the pose *frame coordinates* are synthetic (small, arithmetic-
 * friendly points, matching the convention `reconstructTimeline.test.js`
 * already uses) -- generated at exactly the real `frames_kept` count so
 * `reconstructTimeline` reconstructs the real gap structure (including
 * phone's real 29-frame held/dimmed gap) against genuine manifest data.
 *
 * `usePoseSequence`'s `fetch("/poses/...")` calls are mocked directly
 * (rather than exercising the real Vite dev-server middleware, which only
 * runs under `vite dev`/`vite preview`, not under Vitest), so this test
 * exercises the real App -> usePoseSequence -> reconstructTimeline ->
 * SkeletonCanvas pipeline end-to-end against that real manifest data.
 */

// Verbatim from pose_library/data/poses/manifest.json.
const REAL_ABOUT_MANIFEST_ENTRY = {
  filename: "about.json",
  wlasl_source: "asldeafined",
  wlasl_video_id: "00416",
  wlasl_instance_id: 2,
  license: "C-UDA-1.0; WLASL README: academic/computational use only, no commercial use",
  total_frames_decoded: 116,
  frames_kept: 68,
  dropped_frame_indices: [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 91, 92, 93,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115,
  ],
  low_confidence: false,
  quality_notes: null,
};

// Verbatim from pose_library/data/poses/manifest.json.
const REAL_PHONE_MANIFEST_ENTRY = {
  filename: "phone.json",
  wlasl_source: "spreadthesign",
  wlasl_video_id: "42423",
  wlasl_instance_id: 6,
  license: "C-UDA-1.0; WLASL README: academic/computational use only, no commercial use",
  total_frames_decoded: 68,
  frames_kept: 11,
  dropped_frame_indices: [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67,
  ],
  low_confidence: true,
  quality_notes:
    "low frame count (11 kept, threshold <25); low retention (16.2% of 68 decoded frames kept, " +
    "threshold <42%); 29 mid-clip tracking gaps, not just leading/trailing (threshold >=10)",
};

const LANDMARK_NAMES = ["left_hand_wrist", "right_hand_wrist", "left_shoulder", "right_shoulder"];

/** Small, arithmetic-friendly synthetic points (same convention as
 * reconstructTimeline.test.js) -- real pose coordinates aren't needed since
 * nothing here asserts on rendered pixel positions. */
function makeFrame(k) {
  return LANDMARK_NAMES.map((_, i) => [0.4 + i * 0.05 + k * 0.001, 0.5, 0]);
}

function makeSequence(gloss, source, framesKept) {
  return {
    gloss,
    fps: 30,
    landmark_names: LANDMARK_NAMES,
    frames: Array.from({ length: framesKept }, (_, k) => makeFrame(k)),
    source,
  };
}

const MANIFEST = {
  about: REAL_ABOUT_MANIFEST_ENTRY,
  phone: REAL_PHONE_MANIFEST_ENTRY,
};

const SEQUENCES = {
  about: makeSequence("ABOUT", "wlasl:asldeafined:00416", REAL_ABOUT_MANIFEST_ENTRY.frames_kept),
  phone: makeSequence("PHONE", "wlasl:spreadthesign:42423", REAL_PHONE_MANIFEST_ENTRY.frames_kept),
};

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url) => {
      const path = String(url);
      if (path.endsWith("/poses/manifest.json")) {
        return jsonResponse(MANIFEST);
      }
      const match = path.match(/\/poses\/([^/]+)\.json$/);
      const word = match?.[1];
      if (word && SEQUENCES[word]) {
        return jsonResponse(SEQUENCES[word]);
      }
      return jsonResponse({ error: "not_found" }, 404);
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Submits one or more space-separated words through the real CaptionBand
 * search form, the same path a user takes -- not a direct state injection.
 * Per `frontend/MULTIWORD_PLAN.md` Section 1, multiple words are typed
 * space-separated into the same single input the original single-word app
 * used. */
async function searchFor(text) {
  const input = screen.getByLabelText("WORD");
  fireEvent.change(input, { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "LOOKUP" }));
}

describe("App integration: real pose-library words", () => {
  it('renders a clean "ok" word ("about"): skeleton, gloss caption, and READY status all appear', async () => {
    render(<App />);

    await searchFor("about");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument();
    });

    // The skeleton canvas is present and labeled, per PLAN.md Section 1/7's
    // "role=img" accessibility treatment -- not asserting on pixels.
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();

    // Status strip reflects the ready state, not loading/error/low-confidence.
    expect(screen.getByText("READY")).toBeInTheDocument();

    // Source attribution is shown (C-UDA attribution requirement, PLAN.md
    // Section 2).
    expect(screen.getByText("SOURCE: wlasl:asldeafined:00416")).toBeInTheDocument();

    // No low-confidence or OOV banner for a clean word.
    expect(screen.queryByText(/LOW-CONFIDENCE SIGN/)).not.toBeInTheDocument();
    expect(screen.queryByText(/NO SIGN FOUND/)).not.toBeInTheDocument();
  });

  it('renders a real low_confidence word ("phone"): degraded-skeleton banner and real quality_notes appear, skeleton still plays', async () => {
    render(<App />);

    await searchFor("phone");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "PHONE" })).toBeInTheDocument();
    });

    // The skeleton still renders underneath -- a low-confidence word is not
    // refused, per PLAN.md Section 2's "banner is the signal; playback is
    // not gated on it."
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();

    // Status strip's low-confidence label.
    expect(screen.getByText("LOW-CONFIDENCE SIGN")).toBeInTheDocument();

    // CaptionBand's banner surfaces the real quality_notes text from the
    // manifest, not a generic message.
    expect(screen.getByText(/low frame count \(11 kept, threshold <25\)/)).toBeInTheDocument();
    expect(screen.getByText(/29 mid-clip tracking gaps/)).toBeInTheDocument();
  });

  it('renders "NO SIGN FOUND" for a guaranteed-OOV word, per App.jsx\'s actual StageMessage copy', async () => {
    render(<App />);

    await searchFor("xyzzynotasign");

    // The message appears twice by design (Stage replaces the skeleton with
    // it, per PLAN.md Section 5; CaptionBand additionally banners it) --
    // assert both, rather than picking one arbitrarily with getByText.
    await waitFor(() => {
      expect(screen.getAllByText(/NO SIGN FOUND FOR "XYZZYNOTASIGN"/)).toHaveLength(2);
    });

    expect(screen.getAllByText(/FINGERSPELLING NOT YET AVAILABLE/)).toHaveLength(2);

    // No skeleton canvas for a miss -- the Stage message replaces it
    // entirely (PLAN.md Section 5: "replacing the skeleton stage with that
    // message" rather than a frozen last-played skeleton).
    expect(
      screen.queryByRole("img", { name: "ASL sign skeleton animation" })
    ).not.toBeInTheDocument();

    // Status strip reflects the miss.
    expect(screen.getByText("NO SIGN FOUND")).toBeInTheDocument();
  });

  it("plays a real multi-word sequence with a not_found word placed mid-sequence: skipped visibly, the rest still plays", async () => {
    render(<App />);

    // "xyzzynotasign" (confirmed absent from all 118 real manifest keys,
    // see the OOV test above) sits *between* the two real words, not at
    // an edge -- per PLAN.md Section 7's explicit test-bar requirement.
    await searchFor("about xyzzynotasign phone");

    // The sequence still plays: the skeleton canvas is present, not
    // replaced by a full-stage message, per PLAN.md Section 4's "skip, not
    // abort" rule for a partial not_found.
    await waitFor(() => {
      expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    });

    // Both real words appear in the caption row (one as the active
    // heading, the other as a chip) -- neither is silently dropped.
    expect(screen.getByText("ABOUT")).toBeInTheDocument();
    expect(screen.getByText("PHONE")).toBeInTheDocument();

    // The skipped word is listed visibly, not silently dropped, without
    // taking over the whole Stage (PLAN.md Section 4/5). The status strip
    // *also* says "SKIPPED" (as part of its own composite count label,
    // asserted separately below), so this targets CaptionBand's own
    // dedicated banner specifically, not just any "SKIPPED" text anywhere.
    expect(screen.getByText(/SKIPPED: "XYZZYNOTASIGN"/)).toBeInTheDocument();
    expect(screen.queryByText(/NO SIGN FOUND FOR "XYZZYNOTASIGN"/)).not.toBeInTheDocument();

    // The real low-confidence word ("phone") still surfaces its real
    // quality_notes, same as the single-word low-confidence test above.
    expect(screen.getByText(/29 mid-clip tracking gaps/)).toBeInTheDocument();

    // Status strip reflects the composite ready state: 2 of 3 words
    // playable, 1 skipped -- not a bare "READY" (that's reserved for a
    // single clean word, unchanged from before this feature) and not
    // "NO SIGN FOUND" (that's the all-missing fallback, not this case).
    expect(screen.getByText(/READY.*2\/3 WORDS.*1 SKIPPED/)).toBeInTheDocument();
  });

  it("falls back to the full 'no sign found' message when every word in the sequence is missing", async () => {
    render(<App />);

    await searchFor("xyzzynotasign alsomissing");

    await waitFor(() => {
      expect(screen.getAllByText(/NO SIGN FOUND FOR/)).toHaveLength(2);
    });
    expect(screen.getAllByText(/"XYZZYNOTASIGN", "ALSOMISSING"/)).toHaveLength(2);
    expect(
      screen.queryByRole("img", { name: "ASL sign skeleton animation" })
    ).not.toBeInTheDocument();
    expect(screen.getByText("NO SIGN FOUND")).toBeInTheDocument();
  });
});
