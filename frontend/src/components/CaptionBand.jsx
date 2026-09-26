import { useState } from "react";

import styles from "./CaptionBand.module.css";

/** Per `frontend/MULTIWORD_PLAN.md` Section 1: a security/sanity length
 * cap on the input, same discipline `gloss_model/inference.py`'s
 * `MAX_INPUT_CHARS` already applies upstream. Words beyond this are
 * dropped, not silently truncated -- the caller is told via
 * `wasTruncated` and must surface it. */
const MAX_WORDS = 20;

/**
 * Splits and normalizes the raw search-box text into gloss words.
 *
 * @param {string} raw
 * @returns {{words: string[], wasTruncated: boolean}}
 */
function parseWords(raw) {
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return { words: [], wasTruncated: false };
  }
  // Duplicate consecutive words are valid, intentionally not deduped --
  // ASL repetition (e.g. signing a word twice for emphasis) is legitimate
  // input, per PLAN.md Section 1.
  const all = normalized.split(/\s+/).filter(Boolean);
  return { words: all.slice(0, MAX_WORDS), wasTruncated: all.length > MAX_WORDS };
}

/**
 * @typedef {Object} CaptionWord
 * @property {string} text - Display text (uppercase gloss for a resolved
 *   word, or the uppercased typed word for a skipped/spelled one).
 * @property {"ok"|"low_confidence"|"fingerspelled"|"pending"|"skipped"} kind
 * @property {boolean} isActive - True for the single word currently playing.
 * @property {string[]} [letters] - Uppercase letters, for `fingerspelled`.
 * @property {number|null} [activeLetterIndex] - The letter being signed,
 *   for the active `fingerspelled` word.
 */

/**
 * A fingerspelled word in gloss notation (`B-A-N-A-N-A`), per
 * `pose_library/FINGERSPELLING_PLAN.md` Section 5. The hyphenated letters
 * are `aria-hidden` and a visually-hidden "FINGERSPELLED: BANANA" carries
 * the accessible name, so screen readers don't read "B dash A dash N...".
 * While playing, the letter currently being signed is highlighted.
 */
function SpelledLetters({ word }) {
  return (
    <>
      <span aria-hidden="true">
        {word.letters.map((letter, i) => (
          <span key={i}>
            {i > 0 ? <span className={styles.letterSeparator}>-</span> : null}
            <span
              className={
                word.isActive
                  ? i === word.activeLetterIndex
                    ? styles.letterActive
                    : styles.letterInactive
                  : undefined
              }
            >
              {letter}
            </span>
          </span>
        ))}
      </span>
      <span className={styles.srOnly}>FINGERSPELLED: {word.text}</span>
    </>
  );
}

/**
 * The bottom band: multi-word search input, playback controls, the current
 * sequence rendered as a row of words with the playing word highlighted,
 * and status banners (low-confidence / skipped / truncated).
 *
 * Per `frontend/MULTIWORD_PLAN.md` Sections 1 and 5: submit splits on
 * whitespace into a sequence of already-glossed words (still no gloss-model/
 * ASR involvement at this stage -- generalizing the original single-word
 * "fed gloss manually" scope to multiple gloss tokens, not changing it).
 * Play/Loop are unchanged real `<button>` elements with `aria-pressed` and
 * a visible custom focus ring.
 *
 * @param {Object} props
 * @param {(words: string[], wasTruncated: boolean) => void} props.onSearch -
 *   Called with the parsed, capped word sequence on submit.
 * @param {CaptionWord[]} props.captionWords - The full submitted sequence,
 *   in order, for the word-progress row. Empty if nothing is loaded.
 * @param {string|null} props.source - WLASL attribution string for the
 *   currently-playing word, or `null`.
 * @param {boolean} props.isPlaying
 * @param {() => void} props.onTogglePlay
 * @param {boolean} props.loop
 * @param {() => void} props.onToggleLoop
 * @param {boolean} props.controlsDisabled - True when there's nothing
 *   playable loaded (nothing resolved / error / loading).
 * @param {Array<{label: string, notes: string|null}>} props.lowConfidenceEntries -
 *   Every currently-low-confidence word (`"PHONE"`) or fingerspelled
 *   letter (`LETTER "Q"`) in the sequence, each with its own real
 *   `quality_notes` text.
 * @param {string|null} props.skippedBanner - The "SKIPPED: ..." line for
 *   words that couldn't be signed or fingerspelled, shown in a small
 *   persistent banner *alongside* still-playing content (the partial-skip
 *   case, PLAN.md Section 4/5). `null` when nothing was skipped, or when
 *   `missingMessage` is set instead (the all-missing case uses that
 *   full-takeover message, not this banner).
 * @param {string|null} props.missingMessage - Set only when *every* word in
 *   the sequence was skipped/not found -- the same full "NO SIGN FOUND
 *   FOR..." message `App.jsx`'s Stage message shows, mirrored here per the
 *   single-word precedent of surfacing it in both places.
 * @param {string|null} props.missingDetail - Why those words couldn't be
 *   fingerspelled either (e.g. "FINGERSPELLING ALPHABET NOT AVAILABLE"),
 *   shown under `missingMessage`.
 * @param {boolean} props.wasTruncated - True when the last submission had
 *   more than `MAX_WORDS` words and was capped.
 */
export function CaptionBand({
  onSearch,
  captionWords,
  source,
  isPlaying,
  onTogglePlay,
  loop,
  onToggleLoop,
  controlsDisabled,
  lowConfidenceEntries,
  skippedBanner,
  missingMessage,
  missingDetail,
  wasTruncated,
}) {
  const [inputValue, setInputValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const { words, wasTruncated: truncated } = parseWords(inputValue);
    if (words.length === 0) return;
    onSearch(words, truncated);
  };

  return (
    <footer className={styles.band}>
      {wasTruncated ? (
        <div className={styles.oovBanner} role="status" aria-live="polite">
          SEQUENCE TRUNCATED TO {MAX_WORDS} WORDS
        </div>
      ) : null}

      {lowConfidenceEntries.length > 0 ? (
        <div className={styles.lowConfidenceBanner} role="status" aria-live="polite">
          LOW-CONFIDENCE: {lowConfidenceEntries.map((e) => e.label).join(", ")} —{" "}
          {lowConfidenceEntries.map((e) => e.notes).join(" | ")}
        </div>
      ) : null}

      {skippedBanner ? (
        <div className={styles.oovBanner} role="status" aria-live="polite">
          {skippedBanner}
        </div>
      ) : null}

      {missingMessage ? (
        <div className={styles.oovBanner} role="status" aria-live="polite">
          {missingMessage}
          <br />
          {missingDetail}
        </div>
      ) : null}

      <div className={styles.mainRow}>
        <div className={styles.glossArea}>
          {captionWords.length > 0 ? (
            <div className={styles.wordRow}>
              {captionWords.map((w, i) => {
                const content = w.kind === "fingerspelled" ? <SpelledLetters word={w} /> : w.text;
                return w.isActive ? (
                  <h1 key={i} className={styles.glossWord}>
                    {content}
                  </h1>
                ) : (
                  <span
                    key={i}
                    className={`${styles.wordChip} ${
                      w.kind === "skipped" ? styles.wordChipSkipped : ""
                    } ${w.kind === "low_confidence" ? styles.wordChipLowConfidence : ""} ${
                      w.kind === "fingerspelled" ? styles.wordChipFingerspelled : ""
                    }`}
                  >
                    {content}
                  </span>
                );
              })}
            </div>
          ) : (
            <p className={styles.glossPlaceholder}>ENTER A WORD BELOW TO BEGIN</p>
          )}
          {source ? <p className={styles.source}>SOURCE: {source}</p> : null}
        </div>

        <div className={styles.controls}>
          <button
            type="button"
            className={styles.controlButton}
            onClick={onTogglePlay}
            disabled={controlsDisabled}
            aria-pressed={isPlaying}
          >
            {isPlaying ? "PAUSE" : "PLAY"}
          </button>
          <button
            type="button"
            className={styles.controlButton}
            onClick={onToggleLoop}
            disabled={controlsDisabled}
            aria-pressed={loop}
          >
            LOOP
          </button>
        </div>
      </div>

      <form className={styles.searchForm} onSubmit={handleSubmit}>
        <label className={styles.searchLabel} htmlFor="gloss-word-search">
          WORD
        </label>
        <input
          id="gloss-word-search"
          type="text"
          className={styles.searchInput}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="e.g. about, phone, thanks"
          autoComplete="off"
          spellCheck={false}
        />
        <button type="submit" className={styles.searchSubmit}>
          LOOKUP
        </button>
      </form>
    </footer>
  );
}
