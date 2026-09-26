/**
 * Turns finished speech-recognition text into phrases for translation.
 *
 * Per `pipeline/STAGE1_2_PLAN.md` Section 2: one final recognizer result is
 * one phrase, with no separate VAD. The one exception is length: the gloss
 * model degrades on long inputs (a 13-word sentence measurably stopped early),
 * so a long final is split before translation.
 */

/** Longest phrase sent to the gloss model, in words (Section 2). Tied to
 * measured degradation, not linguistics; tune from real use. */
export const MAX_PHRASE_WORDS = 12;

// Control characters, zero-width/format characters, and line/paragraph
// separators. The gloss server sanitizes too; this keeps garbage out of the
// transcript display and the request in the first place.
const INVISIBLE =
  // eslint-disable-next-line no-control-regex -- matching control chars is the point
  /[\u0000-\u001f\u007f-\u009f\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff]/g;

/**
 * Removes control/invisible characters and collapses whitespace.
 *
 * @param {string} text
 * @returns {string}
 */
export function sanitizeTranscript(text) {
  return text.replace(INVISIBLE, " ").split(/\s+/).filter(Boolean).join(" ");
}

/**
 * Splits one final result into phrases of at most `maxWords` words.
 *
 * A long result is cut into the fewest near-equal chunks (13 words become
 * 7 + 6, not 12 + a stray 1), at word boundaries, in order.
 *
 * @param {string} text - A final (or flushed partial) recognizer result.
 * @param {number} [maxWords]
 * @returns {string[]} Sanitized phrases; `[]` for empty or whitespace text.
 */
export function splitIntoPhrases(text, maxWords = MAX_PHRASE_WORDS) {
  const words = sanitizeTranscript(text).split(" ").filter(Boolean);
  if (words.length === 0) return [];
  const chunks = Math.ceil(words.length / maxWords);
  const size = Math.ceil(words.length / chunks);
  const phrases = [];
  for (let i = 0; i < words.length; i += size) {
    phrases.push(words.slice(i, i + size).join(" "));
  }
  return phrases;
}
