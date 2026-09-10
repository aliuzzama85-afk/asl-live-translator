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
 */
export function StatusStrip({ state, latencyMs, reducedMotion }) {
  const config = STATE_CONFIG[state] ?? STATE_CONFIG.idle;
  const pulsing = state === "loading" && !reducedMotion;

  return (
    <header className={styles.strip}>
      <div className={styles.statusGroup} role="status" aria-live="polite">
        <span
          className={`${styles.dot} ${styles[config.dotClass]} ${pulsing ? styles.pulse : ""}`}
          aria-hidden="true"
        />
        <span className={styles.label}>{config.label}</span>
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
