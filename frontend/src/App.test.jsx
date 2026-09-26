import fs from "node:fs";
import path from "node:path";

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";

import { App } from "./App.jsx";
import captionStyles from "./components/CaptionBand.module.css";
import { createFakeAsr } from "./test/fakeRecognizer.js";

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

    // This suite's fetch mock serves no /fingerspelling/* (as on a checkout
    // without the letter library), so the miss says why it couldn't be
    // spelled either, rather than aborting.
    expect(screen.getAllByText(/FINGERSPELLING ALPHABET NOT AVAILABLE/)).toHaveLength(2);

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

/**
 * Fingerspelling, end to end -- on SYNTHETIC placeholder letters.
 *
 * Pure plumbing tests, independent of the real letter data (24 real letters
 * from the asl-now dataset are tested separately, further down). These
 * letters are hand-constructed
 * stand-ins: manifest entries shaped exactly like
 * `pose_library.manifest.build_letter_entry` output (a 12-frame gap-free
 * static hold), with the same small arithmetic-friendly frames the word
 * fixtures above use. They prove the plumbing -- not_found -> letters ->
 * the same usePoseSequences -> stitchTimelines -> SkeletonCanvas path --
 * not any real handshape.
 */
const SYNTHETIC_HOLD_FRAMES = 12;

function syntheticLetterEntry(letter, { lowConfidence = false } = {}) {
  return {
    filename: `${letter}.json`,
    letter,
    kind: "static",
    signing_hand: "right",
    source: "fingerspelling:self-recorded",
    license: "Self-recorded by the project author; project-owned, MIT (see LICENSE)",
    recording: {
      filename: `raw/${letter}.mp4`,
      total_frames_decoded: 60,
      frames_with_hand: 60,
      segment_start_frame: 20,
    },
    total_frames_decoded: SYNTHETIC_HOLD_FRAMES,
    frames_kept: SYNTHETIC_HOLD_FRAMES,
    dropped_frame_indices: [],
    low_confidence: lowConfidence,
    quality_notes: lowConfidence ? "synthetic low-confidence note" : null,
  };
}

/** Replaces the suite's fetch mock with one that also serves a synthetic
 * letter library containing exactly `letters`. */
function stubFetchWithLetters(letters, { lowConfidence = [] } = {}) {
  const letterManifest = Object.fromEntries(
    letters.map((l) => [l, syntheticLetterEntry(l, { lowConfidence: lowConfidence.includes(l) })])
  );
  const fetchMock = vi.fn(async (url) => {
    const path = String(url);
    if (path === "/poses/manifest.json") return jsonResponse(MANIFEST);
    if (path === "/fingerspelling/manifest.json") return jsonResponse(letterManifest);
    const match = path.match(/^\/(poses|fingerspelling)\/([^/]+)\.json$/);
    if (match?.[1] === "poses" && SEQUENCES[match[2]]) {
      return jsonResponse(SEQUENCES[match[2]]);
    }
    if (match?.[1] === "fingerspelling" && letterManifest[match[2]]) {
      const letter = match[2];
      return jsonResponse(
        makeSequence(letter.toUpperCase(), "fingerspelling:self-recorded", SYNTHETIC_HOLD_FRAMES)
      );
    }
    return jsonResponse({ error: "not_found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The letter currently highlighted inside the playing (heading) word. */
function activeLetter() {
  const heading = screen.queryByRole("heading");
  return heading?.querySelector(`.${captionStyles.letterActive}`)?.textContent ?? null;
}

describe("App integration: fingerspelling (SYNTHETIC letter data)", () => {
  it("spells an out-of-library word end to end, advancing letter by letter", async () => {
    const fetchMock = stubFetchWithLetters(["a", "b", "c"]);
    render(<App />);

    await searchFor("cab");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    expect(screen.getByText("READY — 1/1 WORDS (1 FINGERSPELLED)")).toBeInTheDocument();
    expect(screen.getByText("SOURCE: fingerspelling:self-recorded")).toBeInTheDocument();
    expect(screen.queryByText(/SKIPPED:/)).not.toBeInTheDocument();

    // Real playback through stitchTimelines + SkeletonCanvas's rAF loop: the
    // highlighted letter advances C -> A -> B in spelling order.
    expect(activeLetter()).toBe("C");
    await waitFor(() => expect(activeLetter()).toBe("A"), { timeout: 4000 });
    await waitFor(() => expect(activeLetter()).toBe("B"), { timeout: 4000 });

    // Each letter library file is fetched once, alongside the one word miss.
    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls.filter((u) => u.startsWith("/fingerspelling/")).sort()).toEqual([
      "/fingerspelling/a.json",
      "/fingerspelling/b.json",
      "/fingerspelling/c.json",
      "/fingerspelling/manifest.json",
    ]);
  });

  it("plays a sentence mixing library signs and a spelled word, in order", async () => {
    stubFetchWithLetters(["a", "b", "c"]);
    render(<App />);

    await searchFor("about cab phone");

    await waitFor(() => {
      expect(screen.getByText(/READY — 3\/3 WORDS \(1 FINGERSPELLED\)/)).toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    // "about" plays first; "cab" is a spelled chip, not a skipped one.
    expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument();
    expect(screen.getByText("FINGERSPELLED: CAB")).toBeInTheDocument();
    expect(screen.queryByText(/SKIPPED:/)).not.toBeInTheDocument();
    // The real low-confidence word still surfaces its real notes.
    expect(screen.getByText(/29 mid-clip tracking gaps/)).toBeInTheDocument();
  });

  it("skips a word whose letters aren't all recorded, naming the missing ones", async () => {
    stubFetchWithLetters(["a", "c"]);
    render(<App />);

    await searchFor("about cab");

    await waitFor(() => {
      expect(
        screen.getByText('SKIPPED: "CAB" (NOT FOUND — NO FINGERSPELLING FOR "B")')
      ).toBeInTheDocument();
    });
    // The rest of the sentence still plays.
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    expect(screen.getByText(/READY — 1\/2 WORDS \(1 SKIPPED\)/)).toBeInTheDocument();
  });

  it("flags a low-confidence letter in the same banner as words", async () => {
    stubFetchWithLetters(["a", "b", "c"], { lowConfidence: ["b"] });
    render(<App />);

    await searchFor("cab");

    await waitFor(() => {
      expect(screen.getByText(/LOW-CONFIDENCE: LETTER "B"/)).toBeInTheDocument();
    });
    expect(screen.getByText(/synthetic low-confidence note/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
  });

  it("explains an unspellable miss without fetching any letters", async () => {
    const fetchMock = stubFetchWithLetters(["a", "b", "c"]);
    render(<App />);

    await searchFor("abc1");

    await waitFor(() => {
      expect(screen.getAllByText(/ONLY A–Z CAN BE FINGERSPELLED/)).toHaveLength(2);
    });
    expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith("/fingerspelling/"))).toBe(
      false
    );
  });
});

/**
 * Fingerspelling with the REAL letters committed in
 * pose_library/fingerspelling/poses/ -- not synthetic: 24 converted from the
 * MIT-licensed sid220/asl-now-fingerspelling dataset, plus J and Z
 * self-recorded (the dataset has no motion data for them). Pins that the
 * committed data plays through the real pipeline and that each letter
 * carries its own provenance into the UI. (Missing-letter handling is
 * covered by the synthetic tests above, independent of the real data.)
 */
const REAL_LETTER_MANIFEST = JSON.parse(
  fs.readFileSync(
    path.resolve(__dirname, "../../pose_library/fingerspelling/poses/manifest.json"),
    "utf-8"
  )
);

function realLetterPose(letter) {
  return JSON.parse(
    fs.readFileSync(
      path.resolve(__dirname, `../../pose_library/fingerspelling/poses/${letter}.json`),
      "utf-8"
    )
  );
}

function stubFetchWithRealLetters() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url) => {
      const p = String(url);
      if (p === "/poses/manifest.json") return jsonResponse({});
      if (p === "/fingerspelling/manifest.json") return jsonResponse(REAL_LETTER_MANIFEST);
      const match = p.match(/^\/fingerspelling\/([a-z])\.json$/);
      if (match && REAL_LETTER_MANIFEST[match[1]]) return jsonResponse(realLetterPose(match[1]));
      return jsonResponse({ error: "not_found" }, 404);
    })
  );
}

describe("App integration: fingerspelling with the real converted letters", () => {
  it("spells a word end to end from the committed dataset letters, with their provenance", async () => {
    stubFetchWithRealLetters();
    render(<App />);

    await searchFor("cab");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    const source = REAL_LETTER_MANIFEST.c && realLetterPose("c").source;
    expect(source).toMatch(/^hf:sid220\/asl-now-fingerspelling:C\//);
    expect(screen.getByText(`SOURCE: ${source}`)).toBeInTheDocument();
    await waitFor(() => expect(activeLetter()).toBe("A"), { timeout: 4000 });
  });

  it("spells a word with the self-recorded J and Z, each showing its own source", async () => {
    stubFetchWithRealLetters();
    render(<App />);

    await searchFor("jazz");

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "FINGERSPELLED: JAZZ" })).toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    expect(screen.queryByText(/NO FINGERSPELLING FOR/)).not.toBeInTheDocument();
    // J plays first, and it's the self-recorded letter, not a dataset one.
    expect(activeLetter()).toBe("J");
    expect(realLetterPose("j").source).toBe("fingerspelling:self-recorded");
    expect(screen.getByText("SOURCE: fingerspelling:self-recorded")).toBeInTheDocument();
    // Then A, from the dataset.
    await waitFor(() => expect(activeLetter()).toBe("A"), { timeout: 4000 });
    expect(screen.getByText(/^SOURCE: hf:sid220\/asl-now-fingerspelling:A\//)).toBeInTheDocument();
  });
});

/**
 * Live speech, end to end -- with a FAKE recognizer (it implements the ASR
 * adapter interface; see src/test/fakeRecognizer.js) and a fake gloss client
 * (it stands in for the local gloss server). Everything after the gloss
 * client -- the queue, `setSubmittedWords`, the real usePoseSequences,
 * fingerspelling, stitchTimelines, SkeletonCanvas -- is the real code. Pose
 * data is the same fixtures the tests above use (real manifest entries,
 * synthetic frames; synthetic letters).
 */
const GLOSSES = {
  about: ["about"],
  "about phone": ["about", "phone"],
  "a cab": ["cab"],
  "just pronouns": [],
};

function fakeGlossClient({ health = "ready" } = {}) {
  return {
    waitForGlossService: vi.fn(async () => ({ state: health })),
    glossPhrase: vi.fn(async (text, phraseId) => ({
      ok: true,
      result: {
        phrase_id: phraseId,
        text,
        gloss: (GLOSSES[text] ?? []).join(" ").toUpperCase(),
        words: GLOSSES[text] ?? [],
        dropped: [],
        inference_ms: 5,
      },
    })),
  };
}

async function turnMicOn() {
  fireEvent.click(screen.getByRole("button", { name: "MIC" }));
  // First use: the privacy notice must be acknowledged before listening.
  fireEvent.click(await screen.findByRole("button", { name: "OK" }));
}

describe("App integration: live speech (fake recognizer)", () => {
  beforeEach(() => {
    localStorage.clear();
    stubFetchWithLetters(["a", "b", "c"]);
  });

  it("speech -> transcript -> translating -> signing, through the real playback path", async () => {
    const fake = createFakeAsr();
    const gloss = fakeGlossClient();
    render(<App asr={fake.asr} glossClient={gloss} />);

    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => fake.emit({ type: "listening" }));
    expect(screen.getByText("LISTENING")).toBeInTheDocument();

    act(() => fake.emit({ type: "partial", text: "about ph" }));
    expect(screen.getByText(/about ph/)).toBeInTheDocument();

    act(() => fake.emit({ type: "final", text: "about phone", at: performance.now() }));
    expect(gloss.glossPhrase).toHaveBeenCalledWith("about phone", 1);

    await waitFor(() => {
      expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    });
    expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument();
    expect(screen.getByText("PHONE")).toBeInTheDocument();
    // The status strip says SIGNING, and the transcript line tags the phrase.
    expect(screen.getByText("SIGNING")).toBeInTheDocument();
    expect(screen.getByText("about phone")).toBeInTheDocument();
    expect(screen.getByText(/· SIGNING/)).toBeInTheDocument();
    expect(screen.getByText("SPEECH→SIGN")).toBeInTheDocument();
    expect(screen.getByText(/\d+ms/)).toBeInTheDocument();
  });

  it("a spoken out-of-library word is fingerspelled", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => fake.emit({ type: "final", text: "a cab", at: performance.now() }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument();
    });
  });

  it("queues a second phrase until the first finishes signing", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => {
      fake.emit({ type: "final", text: "about", at: performance.now() });
      fake.emit({ type: "final", text: "a cab", at: performance.now() });
    });

    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    expect(screen.getByText(/SIGNING — 1 PHRASE QUEUED/)).toBeInTheDocument();
    // Only after ABOUT has finished playing does the queued phrase start.
    await waitFor(
      () => expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument(),
      { timeout: 8000 }
    );
  }, 15000);

  it("a phrase with nothing to sign says so and doesn't stall the queue", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => {
      fake.emit({ type: "final", text: "just pronouns", at: performance.now() });
      fake.emit({ type: "final", text: "about", at: performance.now() });
    });
    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
  });

  it("LOOP is visibly disabled, with its reason, while the mic is on", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();

    const loopButton = screen.getByRole("button", { name: "LOOP" });
    expect(loopButton).toBeDisabled();
    expect(screen.getByText("LOOP IS OFF DURING LIVE SPEECH")).toBeInTheDocument();
    expect(loopButton).toHaveAttribute("aria-describedby", "loop-disabled-reason");
    // Typing is disabled too, with the reason in the placeholder.
    expect(screen.getByLabelText("WORD")).toBeDisabled();
  });

  it("a denied microphone turns the mic off with a clear message", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => fake.emit({ type: "error", code: "not-allowed", message: "" }));

    expect(screen.getByText(/MICROPHONE BLOCKED/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "MIC" })).toHaveAttribute("aria-pressed", "false");
  });

  it("an unreachable gloss service is reported, and listening never starts", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient({ health: "unreachable" })} />);
    await turnMicOn();

    await waitFor(() => {
      expect(screen.getByText(/TRANSLATION SERVICE NOT RUNNING/)).toBeInTheDocument();
    });
    expect(fake.instances).toHaveLength(0);
    expect(screen.getByRole("button", { name: "MIC" })).toHaveAttribute("aria-pressed", "false");
  });

  it("the privacy notice is shown once, then remembered", async () => {
    const fake = createFakeAsr();
    const { unmount } = render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    await turnMicOn();
    unmount();

    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    fireEvent.click(screen.getByRole("button", { name: "MIC" }));
    expect(screen.queryByText(/SPEECH IS PROCESSED BY YOUR BROWSER/)).not.toBeInTheDocument();
  });

  it("without the Web Speech API, MIC is disabled and says why", () => {
    render(<App asr={{ supported: false, kind: "none", createRecognizer: null }} />);
    const mic = screen.getByRole("button", { name: "MIC" });
    expect(mic).toBeDisabled();
    expect(screen.getByText("LIVE SPEECH NEEDS CHROME, EDGE, OR SAFARI")).toBeInTheDocument();
    expect(mic).toHaveAttribute("aria-describedby", "mic-disabled-reason");
  });
});

describe("App integration: live-mode prompts never claim to be listening early", () => {
  beforeEach(() => {
    localStorage.clear();
    stubFetchWithLetters(["a"]);
  });

  it("while the privacy notice waits for OK, nothing says LISTENING", async () => {
    const fake = createFakeAsr();
    render(<App asr={fake.asr} glossClient={fakeGlossClient()} />);
    fireEvent.click(screen.getByRole("button", { name: "MIC" }));

    expect(await screen.findByText("CONFIRM THE NOTICE TO START")).toBeInTheDocument();
    expect(
      screen.getAllByText("CONFIRM THE NOTICE BELOW TO START LISTENING").length
    ).toBeGreaterThan(0);
    expect(screen.queryByText(/LISTENING — START SPEAKING/)).not.toBeInTheDocument();
    expect(fake.instances).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "OK" }));
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => fake.emit({ type: "listening" }));
    expect(screen.getAllByText("LISTENING — START SPEAKING").length).toBeGreaterThan(0);
  });
});
