import { describe, expect, it } from "vitest";

import { MAX_PHRASE_WORDS, sanitizeTranscript, splitIntoPhrases } from "./phrases.js";

describe("sanitizeTranscript", () => {
  it("removes control and invisible characters and collapses whitespace", () => {
    const messy = "  where\u0000is\tthe\u200b bath\u00adroom \n\u2028today\ufeff ";
    expect(sanitizeTranscript(messy)).toBe("where is the bath room today");
  });

  it("leaves ordinary text alone", () => {
    expect(sanitizeTranscript("can you help me")).toBe("can you help me");
  });
});

describe("splitIntoPhrases", () => {
  it("keeps a phrase of up to 12 words whole", () => {
    const twelve = Array.from({ length: MAX_PHRASE_WORDS }, (_, i) => `w${i}`).join(" ");
    expect(splitIntoPhrases(twelve)).toEqual([twelve]);
  });

  it("splits a long final into the fewest near-equal chunks, in order", () => {
    // The real 13-word sentence the model stopped early on (STAGE1_2_PLAN.md
    // Section 3) becomes 7 + 6, not 12 + a stray 1.
    const text = "my friend is not coming to the party tonight because she is sick";
    expect(splitIntoPhrases(text)).toEqual([
      "my friend is not coming to the",
      "party tonight because she is sick",
    ]);
  });

  it("never exceeds the cap, and loses no words", () => {
    const words = Array.from({ length: 30 }, (_, i) => `w${i}`);
    const phrases = splitIntoPhrases(words.join(" "));
    expect(phrases.every((p) => p.split(" ").length <= MAX_PHRASE_WORDS)).toBe(true);
    expect(phrases.join(" ").split(" ")).toEqual(words);
  });

  it("drops empty and whitespace-only results", () => {
    expect(splitIntoPhrases("")).toEqual([]);
    expect(splitIntoPhrases("  \n\t \u200b ")).toEqual([]);
  });
});
