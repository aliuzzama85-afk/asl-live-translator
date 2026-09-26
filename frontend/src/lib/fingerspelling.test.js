import { describe, expect, it } from "vitest";

import {
  MAX_FINGERSPELL_LETTERS,
  formatSkippedBanner,
  lettersNeeded,
  planPlayback,
  skipReasonText,
  spellWord,
} from "./fingerspelling.js";

/**
 * All letter data here is SYNTHETIC placeholder data -- tiny stand-in
 * sequences, not real recorded handshapes (the real alphabet isn't
 * recorded yet; see pose_library/FINGERSPELLING_PLAN.md).
 */
function letterResult(letter, status = "ok") {
  return {
    status,
    word: letter,
    sequence: { gloss: letter.toUpperCase(), fps: 30, frames: [], source: "synthetic" },
    manifestEntry: {
      low_confidence: status === "low_confidence",
      quality_notes: status === "low_confidence" ? `synthetic note for ${letter}` : null,
    },
  };
}

function wordResult(word, status = "ok") {
  if (status === "not_found") return { status, word };
  return {
    status,
    word,
    sequence: { gloss: word.toUpperCase(), fps: 30, frames: [], source: "wlasl:x:1" },
    manifestEntry: { low_confidence: status === "low_confidence", quality_notes: "real-ish note" },
  };
}

function readyBatch(letters, overrides = {}) {
  return {
    status: "ready",
    results: letters.map((l) => overrides[l] ?? letterResult(l)),
    fetchMs: 1,
    message: null,
  };
}

const IDLE = { status: "idle", results: [], fetchMs: null, message: null };

describe("spellWord", () => {
  it("splits a plain word into lowercase letters", () => {
    expect(spellWord("Cab")).toEqual({ letters: ["c", "a", "b"], reason: null });
  });

  it("drops apostrophes and hyphens", () => {
    expect(spellWord("don't").letters).toEqual(["d", "o", "n", "t"]);
    expect(spellWord("x-ray").letters).toEqual(["x", "r", "a", "y"]);
  });

  it.each(["abc1", "c@t", "é", "'", ""])("refuses %j as unsupported", (word) => {
    expect(spellWord(word)).toEqual({ letters: null, reason: "unsupported" });
  });

  it("refuses words over the letter cap", () => {
    expect(spellWord("a".repeat(MAX_FINGERSPELL_LETTERS)).letters).toHaveLength(
      MAX_FINGERSPELL_LETTERS
    );
    expect(spellWord("a".repeat(MAX_FINGERSPELL_LETTERS + 1))).toEqual({
      letters: null,
      reason: "too_long",
    });
  });
});

describe("lettersNeeded", () => {
  it("collects each needed letter once, sorted, only from spellable not_found words", () => {
    const results = [
      wordResult("about"),
      wordResult("banana", "not_found"),
      wordResult("cab", "not_found"),
      wordResult("abc1", "not_found"),
    ];
    expect(lettersNeeded(results)).toEqual(["a", "b", "c", "n"]);
  });

  it("is empty when nothing is missing", () => {
    expect(lettersNeeded([wordResult("about")])).toEqual([]);
  });
});

describe("planPlayback", () => {
  it("passes library words through as one unit each, with no letters needed", () => {
    const plan = planPlayback(
      [wordResult("about"), wordResult("phone", "low_confidence")],
      [],
      IDLE
    );
    expect(plan.units.map((u) => [u.word, u.wordIndex, u.letterIndex])).toEqual([
      ["about", 0, null],
      ["phone", 1, null],
    ]);
    expect(plan.words.map((w) => w.kind)).toEqual(["sign", "low_confidence"]);
    expect(plan.lowConfidence).toEqual([{ label: '"PHONE"', notes: "real-ish note" }]);
    expect(plan.pending).toBe(false);
  });

  it("expands a not_found word into one unit per letter, in spelling order, between library words", () => {
    const results = [wordResult("about"), wordResult("cab", "not_found"), wordResult("phone")];
    const needed = lettersNeeded(results);
    const plan = planPlayback(results, needed, readyBatch(needed));

    expect(plan.units.map((u) => [u.word, u.wordIndex, u.letterIndex])).toEqual([
      ["about", 0, null],
      ["c", 1, 0],
      ["a", 1, 1],
      ["b", 1, 2],
      ["phone", 2, null],
    ]);
    // Each unit is shaped like a word result, so stitchTimelines takes it as-is.
    expect(plan.units[1]).toMatchObject({ status: "ok", sequence: { gloss: "C" } });
    expect(plan.words[1]).toEqual({ kind: "fingerspelled", text: "CAB", letters: ["C", "A", "B"] });
  });

  it("reuses one fetched letter for every occurrence", () => {
    const results = [wordResult("banana", "not_found")];
    const needed = lettersNeeded(results);
    const plan = planPlayback(results, needed, readyBatch(needed));
    expect(needed).toEqual(["a", "b", "n"]);
    expect(plan.units.map((u) => u.word).join("")).toBe("banana");
  });

  it("is pending (nothing to stitch yet) while the letter batch hasn't resolved", () => {
    const results = [wordResult("about"), wordResult("cab", "not_found")];
    const needed = lettersNeeded(results);
    for (const batch of [IDLE, { ...IDLE, status: "loading" }]) {
      const plan = planPlayback(results, needed, batch);
      expect(plan.pending).toBe(true);
      expect(plan.words[1].kind).toBe("pending");
    }
  });

  it("treats a batch resolved for *different* letters as still pending", () => {
    const results = [wordResult("cab", "not_found")];
    const stale = readyBatch(["x", "y", "z"]);
    expect(planPlayback(results, lettersNeeded(results), stale).pending).toBe(true);
  });

  it("skips the whole word, naming the missing letters, rather than spell a different word", () => {
    const results = [wordResult("quiz", "not_found")];
    const needed = lettersNeeded(results); // i, q, u, z
    const batch = readyBatch(needed, {
      q: { status: "not_found", word: "q" },
      z: { status: "not_found", word: "z" },
    });
    const plan = planPlayback(results, needed, batch);
    expect(plan.units).toEqual([]);
    expect(plan.words[0]).toMatchObject({
      kind: "skipped",
      skipReason: "missing_letters",
      missingLetters: ["Q", "Z"],
    });
  });

  it("falls back to plain skipping, not an abort, when the alphabet isn't built", () => {
    const results = [wordResult("about"), wordResult("cab", "not_found")];
    const batch = { ...IDLE, status: "error", message: "not built", manifestMissing: true };
    const plan = planPlayback(results, lettersNeeded(results), batch);
    expect(plan.abortMessage).toBeNull();
    expect(plan.fingerspellingUnavailable).toBe(true);
    expect(plan.units.map((u) => u.word)).toEqual(["about"]);
    expect(plan.words[1]).toMatchObject({ kind: "skipped", skipReason: "alphabet_unavailable" });
  });

  it("aborts on a real letter-library fetch failure", () => {
    const results = [wordResult("cab", "not_found")];
    const batch = { ...IDLE, status: "error", message: "HTTP 500", manifestMissing: false };
    expect(planPlayback(results, lettersNeeded(results), batch).abortMessage).toBe("HTTP 500");
  });

  it("aborts when a single letter's fetch errors", () => {
    const results = [wordResult("cab", "not_found")];
    const needed = lettersNeeded(results);
    const batch = readyBatch(needed, { b: { status: "error", message: "HTTP 503" } });
    expect(planPlayback(results, needed, batch).abortMessage).toBe("HTTP 503");
  });

  it("plays low-confidence letters, flagged once each", () => {
    const results = [wordResult("bob", "not_found")];
    const needed = lettersNeeded(results);
    const batch = readyBatch(needed, { b: letterResult("b", "low_confidence") });
    const plan = planPlayback(results, needed, batch);
    expect(plan.units).toHaveLength(3);
    expect(plan.lowConfidence).toEqual([{ label: 'LETTER "B"', notes: "synthetic note for b" }]);
  });

  it("skips unspellable words without fetching letters for them", () => {
    const results = [wordResult("abc1", "not_found")];
    expect(lettersNeeded(results)).toEqual([]);
    const plan = planPlayback(results, [], IDLE);
    expect(plan.pending).toBe(false);
    expect(plan.words[0]).toMatchObject({ kind: "skipped", skipReason: "unsupported" });
  });
});

describe("skip messages", () => {
  it("explains each skip reason", () => {
    expect(skipReasonText({ skipReason: "alphabet_unavailable" })).toMatch(/NOT RECORDED YET/);
    expect(skipReasonText({ skipReason: "unsupported" })).toMatch(/A–Z/);
    expect(skipReasonText({ skipReason: "too_long" })).toMatch(/MAX 20/);
    expect(skipReasonText({ skipReason: "missing_letters", missingLetters: ["Q", "Z"] })).toBe(
      'NO FINGERSPELLING FOR "Q", "Z"'
    );
  });

  it("groups words that share a reason into one clause", () => {
    const banner = formatSkippedBanner([
      { text: "CAB", skipReason: "alphabet_unavailable" },
      { text: "ABC1", skipReason: "unsupported" },
      { text: "DOG", skipReason: "alphabet_unavailable" },
    ]);
    expect(banner).toBe(
      'SKIPPED: "CAB", "DOG" (NOT FOUND — FINGERSPELLING ALPHABET NOT RECORDED YET); ' +
        '"ABC1" (NOT FOUND — ONLY A–Z CAN BE FINGERSPELLED)'
    );
  });
});
