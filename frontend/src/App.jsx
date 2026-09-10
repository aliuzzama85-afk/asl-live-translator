import { useCallback, useEffect, useMemo, useState } from "react";

import { StatusStrip } from "./components/StatusStrip.jsx";
import { SkeletonCanvas } from "./components/SkeletonCanvas.jsx";
import { CaptionBand } from "./components/CaptionBand.jsx";
import { usePoseSequence } from "./hooks/usePoseSequence.js";
import { usePrefersReducedMotion } from "./hooks/usePrefersReducedMotion.js";
import { reconstructTimeline } from "./lib/reconstructTimeline.js";
import styles from "./App.module.css";

/**
 * App shell: the three fixed horizontal bands from PLAN.md Section 0 --
 * StatusStrip (top), the skeleton Stage (middle), CaptionBand (bottom).
 * Owns the word-lookup state and wires it through `usePoseSequence` into
 * the reconstructed timeline the canvas actually plays.
 */
export function App() {
  const [submittedWord, setSubmittedWord] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const reducedMotion = usePrefersReducedMotion();

  const result = usePoseSequence(submittedWord);

  // Auto-play a freshly loaded sequence -- never leave a newly loaded word
  // sitting frozen, per CLAUDE.md's "never a silently frozen avatar" rule.
  useEffect(() => {
    if (result.status === "ok" || result.status === "low_confidence") {
      setIsPlaying(true);
    }
  }, [result]);

  const timeline = useMemo(() => {
    if (result.status === "ok" || result.status === "low_confidence") {
      return reconstructTimeline(result.sequence, result.manifestEntry);
    }
    return null;
  }, [result]);

  const handleSearch = useCallback((word) => {
    setSubmittedWord(word);
  }, []);

  const handleTogglePlay = useCallback(() => setIsPlaying((p) => !p), []);
  const handleToggleLoop = useCallback(() => setLoop((l) => !l), []);
  const handleEnded = useCallback(() => setIsPlaying(false), []);

  let stripState = "idle";
  if (result.status === "loading") stripState = "loading";
  else if (result.status === "ok") stripState = "ready";
  else if (result.status === "low_confidence") stripState = "low_confidence";
  else if (result.status === "not_found") stripState = "not_found";
  else if (result.status === "error") stripState = "error";

  const latencyMs =
    result.status === "ok" || result.status === "low_confidence" ? result.fetchMs : null;

  const isLoaded = result.status === "ok" || result.status === "low_confidence";

  return (
    <div className={styles.app}>
      <StatusStrip state={stripState} latencyMs={latencyMs} reducedMotion={reducedMotion} />

      <main className={styles.stage}>
        {isLoaded ? (
          <SkeletonCanvas
            landmarkNames={result.sequence.landmark_names}
            timeline={timeline}
            isPlaying={isPlaying}
            loop={loop}
            reducedMotion={reducedMotion}
            onEnded={handleEnded}
          />
        ) : (
          <StageMessage
            status={result.status}
            word={result.status === "not_found" ? result.word : submittedWord}
            message={result.status === "error" ? result.message : null}
          />
        )}
      </main>

      <CaptionBand
        onSearch={handleSearch}
        glossWord={isLoaded ? result.sequence.gloss : null}
        source={isLoaded ? result.sequence.source : null}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlay}
        loop={loop}
        onToggleLoop={handleToggleLoop}
        controlsDisabled={!isLoaded}
        lowConfidenceNotes={
          result.status === "low_confidence" ? result.manifestEntry.quality_notes : null
        }
        notFoundWord={result.status === "not_found" ? result.word : null}
      />
    </div>
  );
}

/** The Stage band's non-canvas states (idle/loading/not_found/error) --
 * never a blank canvas or a frozen last-played skeleton with no
 * explanation, per PLAN.md Section 2/6. */
function StageMessage({ status, word, message }) {
  if (status === "loading") {
    return (
      <p className={styles.stageMessage} role="status" aria-live="polite">
        LOADING &quot;{word.toUpperCase()}&quot;…
      </p>
    );
  }
  if (status === "not_found") {
    return (
      <p
        className={`${styles.stageMessage} ${styles.stageMessageError}`}
        role="status"
        aria-live="polite"
      >
        NO SIGN FOUND FOR &quot;{word.toUpperCase()}&quot;
        <br />
        FINGERSPELLING NOT YET AVAILABLE
      </p>
    );
  }
  if (status === "error") {
    return (
      <p
        className={`${styles.stageMessage} ${styles.stageMessageError}`}
        role="status"
        aria-live="polite"
      >
        ERROR: {message}
      </p>
    );
  }
  return <p className={styles.stageMessage}>ENTER A WORD BELOW TO BEGIN</p>;
}
