import { useCallback, useEffect, useMemo, useState } from "react";

import { StatusStrip } from "./components/StatusStrip.jsx";
import { SkeletonCanvas } from "./components/SkeletonCanvas.jsx";
import { CaptionBand } from "./components/CaptionBand.jsx";
import { usePoseSequences } from "./hooks/usePoseSequences.js";
import { usePrefersReducedMotion } from "./hooks/usePrefersReducedMotion.js";
import { stitchTimelines } from "./lib/stitchTimelines.js";
import styles from "./App.module.css";

/** Composes a "no sign found" message for one or more missing words, using
 * the same wording the original single-word app used (`NO SIGN FOUND FOR
 * "X"`) so a single-word miss reads identically to before, generalized to
 * a comma-separated list for multiple misses. */
function formatMissingWordsMessage(words) {
  return `NO SIGN FOUND FOR ${words.map((w) => `"${w.toUpperCase()}"`).join(", ")}`;
}

/**
 * App shell: the three fixed horizontal bands from PLAN.md Section 0 --
 * StatusStrip (top), the skeleton Stage (middle), CaptionBand (bottom).
 *
 * Generalizes the original single-word App to always treat its input as a
 * *sequence* of gloss words (a single typed word is simply a sequence of
 * length 1) -- per `frontend/MULTIWORD_PLAN.md` Section 2. Owns the
 * word-lookup state via `usePoseSequences`, filters out `not_found` words
 * before handing the rest to `stitchTimelines` (Section 3/4), and tracks
 * which word is currently playing via `SkeletonCanvas`'s `onFrameChange`
 * callback and the returned `wordBoundaries` table.
 */
export function App() {
  const [submittedWords, setSubmittedWords] = useState([]);
  const [wasTruncated, setWasTruncated] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const reducedMotion = usePrefersReducedMotion();

  const batch = usePoseSequences(submittedWords);
  const results = batch.results;

  // Per PLAN.md Section 4: `not_found` words are filtered out before
  // stitching (they never need special transition handling -- the real
  // words on either side simply become adjacent), while `originalIndices`
  // remembers each surviving word's position in the full submitted
  // sequence, duplicate-word-safe (no string-identity matching), so the
  // caption row can highlight the right chip even when words repeat.
  const resolvable = useMemo(() => {
    const list = [];
    const originalIndices = [];
    results.forEach((r, i) => {
      if (r.status === "ok" || r.status === "low_confidence") {
        list.push(r);
        originalIndices.push(i);
      }
    });
    return { list, originalIndices };
  }, [results]);

  // A per-word fetch `error` (as opposed to `not_found`) is a technical
  // failure, not a vocabulary gap -- it aborts the whole sequence rather
  // than being silently skipped, per PLAN.md Section 4's stated asymmetry.
  const erroredResult = results.find((r) => r.status === "error");
  const sequenceAborted = batch.status === "error" || Boolean(erroredResult);
  const abortMessage = batch.status === "error" ? batch.message : erroredResult?.message;

  const stitched = useMemo(() => {
    if (sequenceAborted || resolvable.list.length === 0) {
      return null;
    }
    return stitchTimelines(resolvable.list);
  }, [resolvable.list, sequenceAborted]);

  const timeline = stitched?.timeline ?? null;
  const wordBoundaries = stitched?.wordBoundaries ?? [];
  const isLoaded = Boolean(timeline) && timeline.frames.length > 0;

  // Auto-play a freshly loaded sequence -- never leave it sitting frozen,
  // per CLAUDE.md's "never a silently frozen avatar" rule.
  useEffect(() => {
    if (isLoaded) {
      setIsPlaying(true);
    }
  }, [isLoaded]);

  // A new timeline always starts at frame 0 -- reset here rather than
  // waiting for SkeletonCanvas's first tick, so the caption row doesn't
  // briefly show the previous sequence's last-playing word.
  useEffect(() => {
    setCurrentFrameIndex(0);
  }, [timeline]);

  const currentBoundary = useMemo(
    () =>
      wordBoundaries.find(
        (b) => currentFrameIndex >= b.startFrameIndex && currentFrameIndex <= b.endFrameIndex
      ) ?? null,
    [wordBoundaries, currentFrameIndex]
  );
  const currentOriginalIndex =
    currentBoundary !== null ? resolvable.originalIndices[currentBoundary.wordIndex] : null;
  const currentSource =
    currentBoundary !== null ? resolvable.list[currentBoundary.wordIndex].sequence.source : null;

  const skippedWordsAll = results.filter((r) => r.status === "not_found").map((r) => r.word);
  const lowConfidenceEntries = results
    .filter((r) => r.status === "low_confidence")
    .map((r) => ({ word: r.word, notes: r.manifestEntry.quality_notes }));

  // Every submitted word was skipped/not found (and nothing errored) --
  // falls back to the same full-stage "no sign found" treatment the
  // original single-word app used, generalized to list every missing word.
  const allMissing = submittedWords.length > 0 && !sequenceAborted && resolvable.list.length === 0;
  const missingMessage = allMissing ? formatMissingWordsMessage(skippedWordsAll) : null;

  // The small persistent "SKIPPED" banner only applies to the *partial*
  // case (some words play, others didn't resolve) -- the all-missing case
  // uses `missingMessage`'s full-takeover treatment instead, so the two
  // never render at once.
  const skippedWordsForBanner = allMissing ? [] : skippedWordsAll;

  const captionWords = useMemo(
    () =>
      results.map((r, origIdx) => {
        const isActive = currentOriginalIndex === origIdx;
        if (r.status === "ok" || r.status === "low_confidence") {
          return { text: r.sequence.gloss, kind: r.status, isActive };
        }
        if (r.status === "not_found") {
          return { text: r.word.toUpperCase(), kind: "skipped", isActive: false };
        }
        return { text: r.word ? r.word.toUpperCase() : "?", kind: "skipped", isActive: false };
      }),
    [results, currentOriginalIndex]
  );

  const handleSearch = useCallback((words, truncated) => {
    setSubmittedWords(words);
    setWasTruncated(truncated);
  }, []);

  const handleTogglePlay = useCallback(() => setIsPlaying((p) => !p), []);
  const handleToggleLoop = useCallback(() => setLoop((l) => !l), []);
  const handleEnded = useCallback(() => setIsPlaying(false), []);
  const handleFrameChange = useCallback((frameIndex) => setCurrentFrameIndex(frameIndex), []);

  let stripState = "idle";
  if (batch.status === "loading") stripState = "loading";
  else if (sequenceAborted) stripState = "error";
  else if (allMissing) stripState = "not_found";
  else if (lowConfidenceEntries.length > 0) stripState = "low_confidence";
  else if (isLoaded) stripState = "ready";

  const latencyMs = batch.status === "ready" ? batch.fetchMs : null;

  return (
    <div className={styles.app}>
      <StatusStrip
        state={stripState}
        latencyMs={latencyMs}
        reducedMotion={reducedMotion}
        wordCount={submittedWords.length}
        skippedCount={skippedWordsAll.length}
      />

      <main className={styles.stage}>
        {isLoaded ? (
          <SkeletonCanvas
            landmarkNames={resolvable.list[0].sequence.landmark_names}
            timeline={timeline}
            isPlaying={isPlaying}
            loop={loop}
            reducedMotion={reducedMotion}
            onEnded={handleEnded}
            onFrameChange={handleFrameChange}
          />
        ) : (
          <StageMessage
            status={batch.status}
            sequenceAborted={sequenceAborted}
            abortMessage={abortMessage}
            missingMessage={missingMessage}
          />
        )}
      </main>

      <CaptionBand
        onSearch={handleSearch}
        captionWords={captionWords}
        source={currentSource}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlay}
        loop={loop}
        onToggleLoop={handleToggleLoop}
        controlsDisabled={!isLoaded}
        lowConfidenceEntries={lowConfidenceEntries}
        skippedWords={skippedWordsForBanner}
        missingMessage={missingMessage}
        wasTruncated={wasTruncated}
      />
    </div>
  );
}

/** The Stage band's non-canvas states (idle/loading/aborted/all-missing) --
 * never a blank canvas or a frozen last-played skeleton with no
 * explanation, per PLAN.md Section 2/6. `missingMessage` is always set
 * whenever words were submitted, nothing resolved, and nothing errored
 * (see `allMissing`'s derivation in `App`), so no separate "submitted but
 * unaccounted for" fallback branch is needed here. */
function StageMessage({ status, sequenceAborted, abortMessage, missingMessage }) {
  if (status === "loading") {
    return (
      <p className={styles.stageMessage} role="status" aria-live="polite">
        LOADING…
      </p>
    );
  }
  if (sequenceAborted) {
    return (
      <p
        className={`${styles.stageMessage} ${styles.stageMessageError}`}
        role="status"
        aria-live="polite"
      >
        ERROR: {abortMessage}
      </p>
    );
  }
  if (missingMessage) {
    return (
      <p
        className={`${styles.stageMessage} ${styles.stageMessageError}`}
        role="status"
        aria-live="polite"
      >
        {missingMessage}
        <br />
        FINGERSPELLING NOT YET AVAILABLE
      </p>
    );
  }
  return <p className={styles.stageMessage}>ENTER A WORD BELOW TO BEGIN</p>;
}
