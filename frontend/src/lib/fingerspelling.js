/**
 * Fingerspelling fallback: expands a `not_found` word into per-letter
 * playback units, shaped exactly like a word result so the existing
 * `stitchTimelines` pipeline plays them with no second rendering path.
 *
 * See `pose_library/FINGERSPELLING_PLAN.md` Section 3 for the rules and the
 * reasoning behind each failure-handling choice (skip a word whose letters
 * aren't all available rather than spell a different word; fall back to
 * plain skipping when the alphabet isn't recorded; abort on a real fetch
 * error, same as for words).
 */

/** Per-word length cap -- `CLAUDE.md`'s "length-limit all user input" rule
 * applied per word, since one pasted 200-character token would otherwise
 * queue ~2 minutes of letters. */
export const MAX_FINGERSPELL_LETTERS = 20;

/** Punctuation dropped before spelling (not fingerspelled in this v1; see
 * the plan's out-of-scope list). Any other non-letter makes a word
 * unspellable. */
const DROPPED_PUNCTUATION = /['’-]/g;

/**
 * @typedef {"unsupported"|"too_long"|"alphabet_unavailable"|"missing_letters"} SkipReason
 */

/**
 * Splits a word into the letters to fingerspell.
 *
 * @param {string} word
 * @returns {{letters: string[], reason: null} | {letters: null, reason: "unsupported"|"too_long"}}
 *   `unsupported` for digits/symbols (or nothing left after dropping
 *   punctuation); `too_long` past `MAX_FINGERSPELL_LETTERS`.
 */
export function spellWord(word) {
  const cleaned = word.toLowerCase().replace(DROPPED_PUNCTUATION, "");
  if (!/^[a-z]+$/.test(cleaned)) {
    return { letters: null, reason: "unsupported" };
  }
  if (cleaned.length > MAX_FINGERSPELL_LETTERS) {
    return { letters: null, reason: "too_long" };
  }
  return { letters: cleaned.split(""), reason: null };
}

/**
 * The deduplicated letters needed to spell every spellable `not_found` word,
 * in alphabet order -- each letter is fetched once, however often it
 * appears ("banana" needs `a`, `b`, `n`).
 *
 * @param {Array<{status: string, word?: string}>} wordResults - The word
 *   batch's per-word results.
 * @returns {string[]}
 */
export function lettersNeeded(wordResults) {
  const needed = new Set();
  for (const r of wordResults) {
    if (r.status !== "not_found") continue;
    const { letters } = spellWord(r.word);
    letters?.forEach((l) => needed.add(l));
  }
  return [...needed].sort();
}

/**
 * Whether the letter batch is resolved *for this exact set of letters* --
 * for one render after the needed letters change, the hook still returns
 * the previous batch's results, which must not be matched against the new
 * letters.
 */
function letterBatchState(neededLetters, letterBatch) {
  if (neededLetters.length === 0) return "none";
  if (letterBatch.status === "error") {
    return letterBatch.manifestMissing ? "unavailable" : "error";
  }
  if (letterBatch.status !== "ready") return "pending";
  const { results } = letterBatch;
  const matches =
    results.length === neededLetters.length &&
    results.every((r, i) => r.word === undefined || r.word === neededLetters[i]);
  return matches ? "ready" : "pending";
}

/**
 * @typedef {Object} PlaybackUnit
 * @property {string} word - The word, or the letter for a fingerspelled unit.
 * @property {object} sequence - Its `PoseSequence` JSON.
 * @property {object} manifestEntry - Its manifest entry.
 * @property {"ok"|"low_confidence"} status
 * @property {number} wordIndex - Position in the submitted sequence.
 * @property {number|null} letterIndex - Position within a fingerspelled
 *   word, or `null` for a library sign.
 */

/**
 * @typedef {Object} PlannedWord
 * @property {"sign"|"low_confidence"|"fingerspelled"|"skipped"|"pending"|"error"} kind
 * @property {string} text - Uppercase display text.
 * @property {string[]} [letters] - Uppercase letters, for `fingerspelled`.
 * @property {SkipReason} [skipReason] - For `skipped`.
 * @property {string[]} [missingLetters] - Uppercase, for `missing_letters`.
 */

/**
 * Turns the word batch (plus the letter batch, for `not_found` words) into
 * the ordered playback units handed to `stitchTimelines`, and per-word
 * display state for the caption row.
 *
 * @param {Array<object>} wordResults - The word batch's per-word results.
 * @param {string[]} neededLetters - From `lettersNeeded(wordResults)`.
 * @param {{status: string, results: Array<object>, message?: string|null, manifestMissing?: boolean}} letterBatch -
 *   The letter batch (`usePoseSequences(neededLetters, {basePath: "/fingerspelling"})`).
 * @returns {{
 *   units: PlaybackUnit[],
 *   words: PlannedWord[],
 *   pending: boolean,
 *   abortMessage: string|null,
 *   lowConfidence: Array<{label: string, notes: string|null}>,
 *   fingerspellingUnavailable: boolean,
 * }} `pending` is true while a `not_found` word waits on letters;
 *   `abortMessage` is set when a letter fetch failed for real.
 */
export function planPlayback(wordResults, neededLetters, letterBatch) {
  const state = letterBatchState(neededLetters, letterBatch);
  const letterResults = new Map();
  if (state === "ready") {
    neededLetters.forEach((l, i) => letterResults.set(l, letterBatch.results[i]));
  }

  const units = [];
  const words = [];
  const lowConfidence = [];
  const lowConfidenceLetters = new Set();
  let pending = false;
  let abortMessage = state === "error" ? letterBatch.message : null;

  wordResults.forEach((r, wordIndex) => {
    if (r.status === "ok" || r.status === "low_confidence") {
      units.push({ ...r, wordIndex, letterIndex: null });
      words.push({ kind: r.status === "ok" ? "sign" : "low_confidence", text: r.sequence.gloss });
      if (r.status === "low_confidence") {
        lowConfidence.push({
          label: `"${r.word.toUpperCase()}"`,
          notes: r.manifestEntry.quality_notes,
        });
      }
      return;
    }
    if (r.status !== "not_found") {
      words.push({ kind: "error", text: r.word ? r.word.toUpperCase() : "?" });
      return;
    }

    const text = r.word.toUpperCase();
    const { letters, reason } = spellWord(r.word);
    if (!letters) {
      words.push({ kind: "skipped", text, skipReason: reason });
      return;
    }
    if (state === "unavailable") {
      words.push({ kind: "skipped", text, skipReason: "alphabet_unavailable" });
      return;
    }
    if (state !== "ready") {
      pending = pending || state === "pending";
      words.push({ kind: "pending", text });
      return;
    }

    const missing = [];
    for (const l of letters) {
      const lr = letterResults.get(l);
      if (lr?.status === "error") {
        abortMessage = abortMessage ?? lr.message;
      }
      if (lr?.status !== "ok" && lr?.status !== "low_confidence" && !missing.includes(l)) {
        missing.push(l);
      }
    }
    if (missing.length > 0) {
      words.push({
        kind: "skipped",
        text,
        skipReason: "missing_letters",
        missingLetters: missing.map((l) => l.toUpperCase()),
      });
      return;
    }

    letters.forEach((l, letterIndex) => {
      units.push({ ...letterResults.get(l), word: l, wordIndex, letterIndex });
      const lr = letterResults.get(l);
      if (lr.status === "low_confidence" && !lowConfidenceLetters.has(l)) {
        lowConfidenceLetters.add(l);
        lowConfidence.push({
          label: `LETTER "${l.toUpperCase()}"`,
          notes: lr.manifestEntry.quality_notes,
        });
      }
    });
    words.push({ kind: "fingerspelled", text, letters: letters.map((l) => l.toUpperCase()) });
  });

  return {
    units,
    words,
    pending,
    abortMessage,
    lowConfidence,
    fingerspellingUnavailable: state === "unavailable",
  };
}

/**
 * Human-readable reason a word was skipped.
 *
 * @param {PlannedWord} word - A `skipped` planned word.
 * @returns {string}
 */
export function skipReasonText(word) {
  switch (word.skipReason) {
    case "alphabet_unavailable":
      return "FINGERSPELLING ALPHABET NOT RECORDED YET";
    case "unsupported":
      return "ONLY A–Z CAN BE FINGERSPELLED";
    case "too_long":
      return `TOO LONG TO FINGERSPELL (MAX ${MAX_FINGERSPELL_LETTERS} LETTERS)`;
    case "missing_letters":
      return `NO FINGERSPELLING FOR ${word.missingLetters.map((l) => `"${l}"`).join(", ")}`;
    default:
      return "NOT FOUND";
  }
}

/**
 * The "SKIPPED: ..." banner line, with words grouped by shared reason so the
 * common case (alphabet not recorded) reads once, not once per word.
 *
 * @param {PlannedWord[]} skipped - `skipped` planned words, in order.
 * @returns {string}
 */
export function formatSkippedBanner(skipped) {
  const groups = new Map();
  for (const w of skipped) {
    const reason = skipReasonText(w);
    if (!groups.has(reason)) groups.set(reason, []);
    groups.get(reason).push(`"${w.text}"`);
  }
  const parts = [...groups].map(([reason, texts]) => `${texts.join(", ")} (NOT FOUND — ${reason})`);
  return `SKIPPED: ${parts.join("; ")}`;
}
