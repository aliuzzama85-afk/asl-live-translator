import styles from "./StatusStrip.module.css";

/**
 * @typedef {"idle"|"loading"|"ready"|"low_confidence"|"not_found"|"error"
 *   |"connecting"|"service_starting"|"mic_starting"|"listening"|"translating"|"signing"} StripState
 * The last six are live-speech states (`pipeline/STAGE1_2_PLAN.md` Section 5).
 */

const STATE_CONFIG = {
  idle: { dotClass: "dotMuted", label: "STANDBY" },
  loading: { dotClass: "dotAmber", label: "LOADING…", pulse: true },
  ready: { dotClass: "dotReady", label: "READY" },
  low_confidence: { dotClass: "dotError", label: "LOW-CONFIDENCE SIGN" },
  not_found: { dotClass: "dotError", label: "NO SIGN FOUND" },
  error: { dotClass: "dotError", label: "ERROR" },
  connecting: { dotClass: "dotAmber", label: "CONNECTING…", pulse: true },
  service_starting: { dotClass: "dotAmber", label: "TRANSLATION SERVICE STARTING…", pulse: true },
  mic_starting: { dotClass: "dotAmber", label: "STARTING MIC…", pulse: true },
  listening: { dotClass: "dotReady", label: "LISTENING" },
  translating: { dotClass: "dotAmber", label: "TRANSLATING…", pulse: true },
  signing: { dotClass: "dotReady", label: "SIGNING" },
};

const LIVE_STATES = new Set([
  "connecting",
  "service_starting",
  "mic_starting",
  "listening",
  "translating",
  "signing",
]);

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
 * @param {number} [props.skippedCount] - Words skipped (neither signed nor
 *   fingerspelled) in the current sequence.
 * @param {number} [props.fingerspelledCount] - Words fingerspelled instead
 *   of signed. They count as playable, not skipped
 *   (`pose_library/FINGERSPELLING_PLAN.md` Section 5).
 * @param {string} [props.latencyLabel] - `"FETCH"` (typed input: this
 *   stage's own fetch time) or `"SPEECH→SIGN"` (live: from the recognizer
 *   finalizing a phrase to its signing starting). Both are measured.
 * @param {number} [props.queuedCount] - Live phrases waiting (being
 *   translated or queued for the avatar), shown as "— N PHRASES QUEUED".
 */
export function StatusStrip({
  state,
  latencyMs,
  reducedMotion,
  wordCount = 0,
  skippedCount = 0,
  fingerspelledCount = 0,
  latencyLabel = "FETCH",
  queuedCount = 0,
}) {
  const config = STATE_CONFIG[state] ?? STATE_CONFIG.idle;
  const pulsing = Boolean(config.pulse) && !reducedMotion;

  // The composite "N/M WORDS (K SKIPPED)" count is independent of which
  // dot color/state is showing -- a low-confidence word among an otherwise
  // playing sequence still gets an informative "READY — ..." label (the
  // low-confidence nuance itself is surfaced separately, in detail, by
  // CaptionBand's own low-confidence banner) rather than silently losing
  // the count info just because `state` isn't literally "ready". Only a
  // genuinely single, unremarkable word (no skips) keeps the original bare
  // per-state label.
  let label = config.label;
  const isSingleUnremarkableWord = wordCount <= 1 && skippedCount === 0 && fingerspelledCount === 0;
  if ((state === "ready" || state === "low_confidence") && !isSingleUnremarkableWord) {
    const playable = wordCount - skippedCount;
    const notes = [
      fingerspelledCount > 0 ? `${fingerspelledCount} FINGERSPELLED` : null,
      skippedCount > 0 ? `${skippedCount} SKIPPED` : null,
    ].filter(Boolean);
    label = `READY — ${playable}/${wordCount} WORDS${notes.length > 0 ? ` (${notes.join(", ")})` : ""}`;
  }
  if (LIVE_STATES.has(state) && queuedCount > 0) {
    label = `${config.label} — ${queuedCount} PHRASE${queuedCount === 1 ? "" : "S"} QUEUED`;
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
        <span className={styles.latencyLabel}>{latencyLabel}</span>
        <span className={styles.latencyValue}>
          {latencyMs === null ? "--" : `${Math.round(latencyMs)}ms`}
        </span>
      </div>
    </header>
  );
}
