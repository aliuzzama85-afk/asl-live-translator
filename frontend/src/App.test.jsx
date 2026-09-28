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
 * search form in EXACT WORDS mode (each word looked up literally, no gloss
 * model), the same path a user takes -- not a direct state injection. The
 * app defaults to TRANSLATE, so the mode is selected explicitly first.
 * Per `frontend/MULTIWORD_PLAN.md` Section 1, multiple words are typed
 * space-separated into the same single input the original single-word app
 * used. */
async function searchExactWords(text) {
  fireEvent.click(screen.getByRole("button", { name: "EXACT WORDS" }));
  const input = screen.getByLabelText("SIGN WORDS");
  fireEvent.change(input, { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "LOOKUP" }));
}

/** Submits English through the search form in TRANSLATE mode (the default),
 * which sends it to the gloss client like a spoken phrase. */
function typeEnglish(text) {
  fireEvent.change(screen.getByLabelText("ENGLISH"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "SIGN" }));
}

describe("App integration: real pose-library words", () => {
  it('renders a clean "ok" word ("about"): skeleton, gloss caption, and READY status all appear', async () => {
    render(<App />);

    await searchExactWords("about");

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

    await searchExactWords("phone");

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

    await searchExactWords("xyzzynotasign");

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
    await searchExactWords("about xyzzynotasign phone");

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

    await searchExactWords("xyzzynotasign alsomissing");

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

    await searchExactWords("cab");

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

    await searchExactWords("about cab phone");

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

    await searchExactWords("about cab");

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

    await searchExactWords("cab");

    await waitFor(() => {
      expect(screen.getByText(/LOW-CONFIDENCE: LETTER "B"/)).toBeInTheDocument();
    });
    expect(screen.getByText(/synthetic low-confidence note/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
  });

  it("explains an unspellable miss without fetching any letters", async () => {
    const fetchMock = stubFetchWithLetters(["a", "b", "c"]);
    render(<App />);

    await searchExactWords("abc1");

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

    await searchExactWords("cab");

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

    await searchExactWords("jazz");

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
  // Shaped like real server output: the raw gloss keeps pronoun markers and
  // stop-list words, which `dropped` lists and `words` leaves out.
  "tell me about the cab": {
    gloss: "X-I ABOUT BE CAB",
    words: ["about", "cab"],
    dropped: ["X-I", "BE"],
  },
};

function fakeGlossClient({ health = "ready" } = {}) {
  return {
    waitForGlossService: vi.fn(async () => ({ state: health })),
    glossPhrase: vi.fn(async (text, phraseId) => {
      const entry = GLOSSES[text] ?? [];
      const { gloss, words, dropped } = Array.isArray(entry)
        ? { gloss: entry.join(" ").toUpperCase(), words: entry, dropped: [] }
        : entry;
      return {
        ok: true,
        result: { phrase_id: phraseId, text, gloss, words, dropped, inference_ms: 5 },
      };
    }),
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
    expect(screen.getByLabelText("ENGLISH")).toBeDisabled();
    expect(screen.getByLabelText("ENGLISH")).toHaveAttribute(
      "placeholder",
      "Mic is on — turn it off to type"
    );
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

/**
 * Typed input, TRANSLATE mode (the default) vs EXACT WORDS. TRANSLATE sends
 * typed English through the same gloss client and phrase queue as speech;
 * everything after the (fake) gloss client is the real playback path.
 */
describe("App integration: typed input modes", () => {
  beforeEach(() => {
    localStorage.clear();
    stubFetchWithLetters(["a", "b", "c"]);
  });

  function poseRequests() {
    return fetch.mock.calls.map(([url]) => String(url)).filter((u) => u.startsWith("/poses/"));
  }

  it("TRANSLATE is the default and plays the model's words, not the typed ones", async () => {
    const gloss = fakeGlossClient();
    render(<App asr={createFakeAsr().asr} glossClient={gloss} />);

    expect(screen.getByRole("button", { name: "TRANSLATE" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    typeEnglish("tell me about the cab");

    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    expect(gloss.waitForGlossService).toHaveBeenCalled();
    expect(gloss.glossPhrase).toHaveBeenCalledWith("tell me about the cab", 1);
    // Played through the real usePoseSequences -> fingerspelling -> stitch path:
    // ABOUT signed, CAB (not in the library) fingerspelled.
    expect(screen.getByRole("img", { name: "ASL sign skeleton animation" })).toBeInTheDocument();
    expect(screen.getByText("FINGERSPELLED: CAB")).toBeInTheDocument();
    expect(screen.getByText("SIGNING")).toBeInTheDocument();
    // The typed words the model dropped were never looked up or spelled.
    expect(poseRequests().sort()).toEqual([
      "/poses/about.json",
      "/poses/cab.json",
      "/poses/manifest.json",
    ]);
    // The gloss line shows exactly what the model produced, and what isn't signed.
    expect(screen.getByText("X-I ABOUT BE CAB")).toBeInTheDocument();
    expect(screen.getByText(/NOT SIGNED: X-I BE/)).toBeInTheDocument();
    expect(screen.getByText("tell me about the cab")).toBeInTheDocument();
    expect(screen.getByText("TEXT→SIGN")).toBeInTheDocument();
  });

  it("EXACT WORDS never calls the gloss client", async () => {
    const gloss = fakeGlossClient();
    render(<App asr={createFakeAsr().asr} glossClient={gloss} />);

    await searchExactWords("about cab");

    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    expect(screen.getByText("FINGERSPELLED: CAB")).toBeInTheDocument();
    expect(gloss.waitForGlossService).not.toHaveBeenCalled();
    expect(gloss.glossPhrase).not.toHaveBeenCalled();
    expect(screen.queryByText(/GLOSS:/)).not.toBeInTheDocument();
    // LOOP works as before in EXACT WORDS mode.
    const loopButton = screen.getByRole("button", { name: "LOOP" });
    expect(loopButton).toBeEnabled();
    fireEvent.click(loopButton);
    expect(loopButton).toHaveAttribute("aria-pressed", "true");
  });

  it("an unreachable gloss service shows the banner, and nothing is signed literally", async () => {
    const gloss = fakeGlossClient({ health: "unreachable" });
    render(<App asr={createFakeAsr().asr} glossClient={gloss} />);

    typeEnglish("about cab");

    await waitFor(() => {
      expect(screen.getByText(/TRANSLATION SERVICE NOT RUNNING/)).toBeInTheDocument();
    });
    expect(screen.getByText(/EXACT WORDS MODE WORKS WITHOUT IT/)).toBeInTheDocument();
    expect(gloss.glossPhrase).not.toHaveBeenCalled();
    expect(poseRequests()).toEqual([]);
    expect(
      screen.queryByRole("img", { name: "ASL sign skeleton animation" })
    ).not.toBeInTheDocument();
    expect(screen.getByText("ERROR")).toBeInTheDocument();
  });

  it("a server that was ready and then stopped is reported without the startup grace", async () => {
    const gloss = fakeGlossClient();
    gloss.waitForGlossService
      .mockResolvedValueOnce({ state: "ready" })
      .mockResolvedValueOnce({ state: "unreachable" });
    render(<App asr={createFakeAsr().asr} glossClient={gloss} />);

    typeEnglish("about");
    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    // First check: the server may still be starting, so the normal grace applies.
    expect(gloss.waitForGlossService.mock.calls[0][0].graceMs).toBeUndefined();

    typeEnglish("a cab");
    await waitFor(() => {
      expect(screen.getByText(/TRANSLATION SERVICE NOT RUNNING/)).toBeInTheDocument();
    });
    // Seen ready before, so "no answer" now means stopped: reported at once.
    expect(gloss.waitForGlossService.mock.calls[1][0].graceMs).toBe(0);
    expect(gloss.glossPhrase).toHaveBeenCalledTimes(1);
  });

  it("LOOP is visibly disabled in TRANSLATE mode, with its reason", async () => {
    render(<App asr={createFakeAsr().asr} glossClient={fakeGlossClient()} />);
    typeEnglish("about");
    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());

    const loopButton = screen.getByRole("button", { name: "LOOP" });
    expect(loopButton).toBeDisabled();
    expect(
      screen.getByText("LOOP IS OFF IN TRANSLATE MODE — USE EXACT WORDS TO LOOP")
    ).toBeInTheDocument();
    expect(loopButton).toHaveAttribute("aria-describedby", "loop-disabled-reason");
  });

  it("EXACT WORDS is locked while a translated phrase plays, then drops its stale gloss line", async () => {
    render(<App asr={createFakeAsr().asr} glossClient={fakeGlossClient()} />);
    typeEnglish("about");
    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    expect(screen.getByText("GLOSS:")).toBeInTheDocument();

    // A literal lookup would replace the phrase being signed, so it waits.
    fireEvent.click(screen.getByRole("button", { name: "EXACT WORDS" }));
    expect(screen.getByLabelText("SIGN WORDS")).toBeDisabled();
    expect(screen.getByLabelText("SIGN WORDS")).toHaveAttribute(
      "placeholder",
      "Translated phrases still playing…"
    );
    expect(screen.getByText("LOOP IS OFF UNTIL QUEUED PHRASES FINISH")).toBeInTheDocument();

    await waitFor(() => expect(screen.getByLabelText("SIGN WORDS")).toBeEnabled(), {
      timeout: 8000,
    });
    await searchExactWords("cab");
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument();
    });
    // The old phrase's transcript and gloss don't sit above the unrelated sequence.
    expect(screen.queryByText("GLOSS:")).not.toBeInTheDocument();
    expect(screen.queryByText("about")).not.toBeInTheDocument();
  }, 15000);

  it("the mode toggle is two real, focusable buttons with aria-pressed, and relabels the input", () => {
    render(<App asr={createFakeAsr().asr} glossClient={fakeGlossClient()} />);

    const group = screen.getByRole("group", { name: "Typed input mode" });
    const translate = screen.getByRole("button", { name: "TRANSLATE" });
    const exact = screen.getByRole("button", { name: "EXACT WORDS" });
    expect(group).toContainElement(translate);
    expect(group).toContainElement(exact);
    // Native <button type="button">: keyboard focus plus Enter/Space activation
    // come from the browser, as for PLAY/LOOP/MIC.
    for (const button of [translate, exact]) {
      expect(button.tagName).toBe("BUTTON");
      expect(button).toHaveAttribute("type", "button");
      button.focus();
      expect(button).toHaveFocus();
    }
    expect(translate).toHaveAttribute("aria-pressed", "true");
    expect(exact).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByLabelText("ENGLISH")).toHaveAttribute(
      "placeholder",
      "Type English, e.g. where is the bathroom"
    );

    fireEvent.click(exact);
    expect(translate).toHaveAttribute("aria-pressed", "false");
    expect(exact).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("SIGN WORDS")).toHaveAttribute(
      "placeholder",
      "Type sign words, e.g. about angry"
    );
    expect(screen.getByRole("button", { name: "LOOKUP" })).toBeInTheDocument();

    fireEvent.click(translate);
    expect(translate).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "SIGN" })).toBeInTheDocument();
  });

  it("typed and spoken phrases share one queue, in order", async () => {
    const fake = createFakeAsr();
    const gloss = fakeGlossClient();
    render(<App asr={fake.asr} glossClient={gloss} />);

    // Speak a phrase, turn the mic off while it's still signing, then type one.
    await turnMicOn();
    await waitFor(() => expect(fake.instances).toHaveLength(1));
    act(() => fake.emit({ type: "final", text: "about", at: performance.now() }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "MIC ON" }));

    // Typing in TRANSLATE mode is allowed while that phrase plays; it queues.
    expect(screen.getByLabelText("ENGLISH")).toBeEnabled();
    typeEnglish("a cab");
    await waitFor(() => expect(screen.getByText(/SIGNING — 1 PHRASE QUEUED/)).toBeInTheDocument());
    expect(gloss.glossPhrase.mock.calls).toEqual([
      ["about", 1],
      ["a cab", 2],
    ]);
    expect(screen.getByRole("heading", { name: "ABOUT" })).toBeInTheDocument();
    // The transcript and gloss lines stay on the phrase being signed; the typed
    // one waiting behind it is shown as NEXT.
    expect(screen.getByText(/NEXT: a cab · QUEUED/)).toBeInTheDocument();
    expect(screen.getByText("GLOSS:").nextSibling).toHaveTextContent("ABOUT");

    // Only after ABOUT has finished does the typed phrase start.
    await waitFor(
      () => expect(screen.getByRole("heading", { name: "FINGERSPELLED: CAB" })).toBeInTheDocument(),
      { timeout: 8000 }
    );
  }, 15000);
});
