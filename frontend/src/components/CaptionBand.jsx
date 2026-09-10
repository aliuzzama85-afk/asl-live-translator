import { useState } from "react";

import styles from "./CaptionBand.module.css";

/**
 * The bottom band: word-search input, playback controls, the current gloss
 * word in the display font, and status banners (low-confidence / OOV).
 *
 * Per PLAN.md Section 3: submit is a lowercase-normalized lookup, no
 * gloss-model/ASR involvement at this stage. Play/Loop are real `<button>`
 * elements with `aria-pressed` reflecting toggle state and a visible custom
 * focus ring (never `outline: none` with nothing in its place).
 *
 * @param {Object} props
 * @param {(word: string) => void} props.onSearch - Called with the
 *   lowercase-normalized word on submit.
 * @param {string|null} props.glossWord - The current sequence's `gloss`
 *   (already uppercase), or `null` if nothing is loaded.
 * @param {string|null} props.source - WLASL attribution string, or `null`.
 * @param {boolean} props.isPlaying
 * @param {() => void} props.onTogglePlay
 * @param {boolean} props.loop
 * @param {() => void} props.onToggleLoop
 * @param {boolean} props.controlsDisabled - True when there's no playable
 *   sequence loaded (nothing found / error / loading).
 * @param {string|null} props.lowConfidenceNotes - `quality_notes` text when
 *   the current word is low-confidence, else `null`.
 * @param {string|null} props.notFoundWord - The word that produced a
 *   not-found result, or `null`.
 */
export function CaptionBand({
  onSearch,
  glossWord,
  source,
  isPlaying,
  onTogglePlay,
  loop,
  onToggleLoop,
  controlsDisabled,
  lowConfidenceNotes,
  notFoundWord,
}) {
  const [inputValue, setInputValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const normalized = inputValue.trim().toLowerCase();
    if (normalized) {
      onSearch(normalized);
    }
  };

  return (
    <footer className={styles.band}>
      {lowConfidenceNotes ? (
        <div className={styles.lowConfidenceBanner} role="status" aria-live="polite">
          ● LOW-CONFIDENCE SIGN — {lowConfidenceNotes}
        </div>
      ) : null}

      {notFoundWord ? (
        <div className={styles.oovBanner} role="status" aria-live="polite">
          NO SIGN FOUND FOR &quot;{notFoundWord.toUpperCase()}&quot; — FINGERSPELLING NOT YET
          AVAILABLE
        </div>
      ) : null}

      <div className={styles.mainRow}>
        <div className={styles.glossArea}>
          {glossWord ? (
            <h1 className={styles.glossWord}>{glossWord}</h1>
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
