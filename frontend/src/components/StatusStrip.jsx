import styles from "./StatusStrip.module.css";

/**
 * @typedef {"idle"|"loading"|"ready"|"low_confidence"|"not_found"|"error"} StripState
 */

const STATE_CONFIG = {
  idle: { dotClass: "dotMuted", label: "STANDBY" },
  loading: { dotClass: "dotAmber", label: "LOADING…" },
  ready: { dotClass: "dotReady", label: "READY" },
  low_confidence: { dotClass: "dotError", label: "LOW-CONFIDENCE SIGN" },
  not_found: { dotClass: "dotError", label: "NO SIGN FOUND" },
  error: { dotClass: "dotError", label: "ERROR" },
};

/**
 * The top band: console-telemetry-styled status strip. A square LED-style
 * status dot + text label (color is never the only signal, per PLAN.md
 * Section 0) on the left, a monospace latency readout on the right.
 *
 * @param {Object} props
 * @param {StripState} props.state - Current overall app state.
 * @param {number|null} props.latencyMs - This word's own fetch-to-first-
 *   frame time in ms, or `null` if nothing has loaded yet. Scoped down to
 *   this stage's own fetch time, per PLAN.md -- never a simulated
 *   end-to-end pipeline number.
 * @param {boolean} props.reducedMotion - Disables the LED pulse animation.
 * @param {number} [props.wordCount] - Words in the current sequence, for
 *   the "N/M WORDS (K SKIPPED)" label suffix (per
 *   `frontend/MULTIWORD_PLAN.md` Section 5). Only shown when there's more
 *   than one word or at least one skip, so a plain single clean or
 *   low-confidence word still reads as the original bare "READY"/
 *   "LOW-CONFIDENCE SIGN" label, unchanged.
 * @param {number} [props.skippedCount] - Words skipped (not found) in the
 *   current sequence.
 */
export function StatusStrip({ state, latencyMs, reducedMotion, wordCount = 0, skippedCount = 0 }) {
  const config = STATE_CONFIG[state] ?? STATE_CONFIG.idle;
  const pulsing = state === "loading" && !reducedMotion;

  // The composite "N/M WORDS (K SKIPPED)" count is independent of which
  // dot color/state is showing -- a low-confidence word among an otherwise
  // playing sequence still gets an informative "READY — ..." label (the
  // low-confidence nuance itself is surfaced separately, in detail, by
  // CaptionBand's own low-confidence banner) rather than silently losing
  // the count info just because `state` isn't literally "ready". Only a
  // genuinely single, unremarkable word (no skips) keeps the original bare
  // per-state label.
  let label = config.label;
  const isSingleUnremarkableWord = wordCount <= 1 && skippedCount === 0;
  if ((state === "ready" || state === "low_confidence") && !isSingleUnremarkableWord) {
    const playable = wordCount - skippedCount;
    label = `READY — ${playable}/${wordCount} WORDS${skippedCount > 0 ? ` (${skippedCount} SKIPPED)` : ""}`;
  }

  return (
    <header className={styles.strip}>
      <div className={styles.statusGroup} role="status" aria-live="polite">
        <span
          className={`${styles.dot} ${styles[config.dotClass]} ${pulsing ? styles.pulse : ""}`}
          aria-hidden="true"
        />
        <span className={styles.label}>{label}</span>
      </div>
      <div className={styles.latency}>
        <span className={styles.latencyLabel}>FETCH</span>
        <span className={styles.latencyValue}>
          {latencyMs === null ? "--" : `${Math.round(latencyMs)}ms`}
        </span>
      </div>
    </header>
  );
}
