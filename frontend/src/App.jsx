import { useCallback, useEffect, useMemo, useState } from "react";

import { StatusStrip } from "./components/StatusStrip.jsx";
import { SkeletonCanvas } from "./components/SkeletonCanvas.jsx";
import { CaptionBand } from "./components/CaptionBand.jsx";
import { usePoseSequences } from "./hooks/usePoseSequences.js";
import { usePrefersReducedMotion } from "./hooks/usePrefersReducedMotion.js";
import { stitchTimelines } from "./lib/stitchTimelines.js";
import {
  formatSkippedBanner,
  lettersNeeded,
  planPlayback,
  skipReasonText,
} from "./lib/fingerspelling.js";
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
 *
 * Fingerspelling (`pose_library/FINGERSPELLING_PLAN.md` Section 3): a
 * `not_found` word is expanded into letters, looked up with the *same*
 * `usePoseSequences` hook against the letter library, and each letter
 * becomes one more unit in the same `stitchTimelines` call -- a
 * fingerspelled word plays through exactly the path a sentence of words does.
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

  // Letters for every `not_found` word, deduplicated, fetched with the same
  // hook against the letter library. Empty (hook idle) until the word batch
  // resolves, since only then is it known which words are missing.
  const neededLettersKey = lettersNeeded(results).join("");
  const neededLetters = useMemo(
    () => (neededLettersKey ? neededLettersKey.split("") : []),
    [neededLettersKey]
  );
  const letterBatch = usePoseSequences(neededLetters, { basePath: "/fingerspelling" });

  // Per PLAN.md Section 4, unresolvable words never reach stitching (the
  // playable words on either side simply become adjacent). Each unit keeps
  // its position in the full submitted sequence (`wordIndex`, plus
  // `letterIndex` for a spelled word), duplicate-word-safe, so the caption
  // row can highlight the right chip -- and letter -- even when words repeat.
  const plan = useMemo(
    () => planPlayback(results, neededLetters, letterBatch),
    [results, neededLetters, letterBatch]
  );

  // A per-word or per-letter fetch `error` (as opposed to `not_found`) is
  // a technical failure, not a vocabulary gap -- it aborts the whole
  // sequence rather than being silently skipped, per PLAN.md Section 4's
  // stated asymmetry.
  const erroredResult = results.find((r) => r.status === "error");
  const sequenceAborted =
    batch.status === "error" || Boolean(erroredResult) || Boolean(plan.abortMessage);
  const abortMessage =
    batch.status === "error" ? batch.message : (erroredResult?.message ?? plan.abortMessage);
  const isLoading = batch.status === "loading" || plan.pending;

  const stitched = useMemo(() => {
    if (sequenceAborted || plan.pending || plan.units.length === 0) {
      return null;
    }
    return stitchTimelines(plan.units);
  }, [plan, sequenceAborted]);

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
  const currentUnit = currentBoundary !== null ? plan.units[currentBoundary.wordIndex] : null;
  const currentSource = currentUnit ? currentUnit.sequence.source : null;

  const skippedWords = plan.words.filter((w) => w.kind === "skipped");
  const fingerspelledCount = plan.words.filter((w) => w.kind === "fingerspelled").length;
  const lowConfidenceEntries = plan.lowConfidence;

  // Nothing playable (every word skipped/not found, and nothing errored) --
  // falls back to the same full-stage "no sign found" treatment the
  // original single-word app used, generalized to list every missing word,
  // plus why each couldn't be fingerspelled either.
  const allMissing =
    submittedWords.length > 0 &&
    batch.status === "ready" &&
    !isLoading &&
    !sequenceAborted &&
    plan.units.length === 0;
  const missingMessage = allMissing
    ? formatMissingWordsMessage(skippedWords.map((w) => w.text))
    : null;
  const missingDetail = allMissing
    ? [...new Set(skippedWords.map((w) => skipReasonText(w)))].join(" · ")
    : null;

  // The small persistent "SKIPPED" banner only applies to the *partial*
  // case (some words play, others didn't resolve) -- the all-missing case
  // uses `missingMessage`'s full-takeover treatment instead, so the two
  // never render at once.
  const skippedBanner =
    !allMissing && skippedWords.length > 0 ? formatSkippedBanner(skippedWords) : null;

  const captionWords = useMemo(
    () =>
      plan.words.map((w, wordIndex) => {
        const isActive = currentUnit !== null && currentUnit.wordIndex === wordIndex;
        if (w.kind === "fingerspelled") {
          return {
            text: w.text,
            kind: "fingerspelled",
            letters: w.letters,
            isActive,
            activeLetterIndex: isActive ? currentUnit.letterIndex : null,
          };
        }
        if (w.kind === "sign") return { text: w.text, kind: "ok", isActive };
        if (w.kind === "low_confidence") return { text: w.text, kind: "low_confidence", isActive };
        if (w.kind === "pending") return { text: w.text, kind: "pending", isActive: false };
        return { text: w.text, kind: "skipped", isActive: false };
      }),
    [plan.words, currentUnit]
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
  if (isLoading) stripState = "loading";
  else if (sequenceAborted) stripState = "error";
  else if (allMissing) stripState = "not_found";
  else if (lowConfidenceEntries.length > 0) stripState = "low_confidence";
  else if (isLoaded) stripState = "ready";

  // Word and letter lookups run one after the other (letters need the word
  // results first), so a spelled sentence's fetch time is both batches.
  const latencyMs =
    batch.status === "ready" && !isLoading
      ? batch.fetchMs + (letterBatch.status === "ready" ? letterBatch.fetchMs : 0)
      : null;

  return (
    <div className={styles.app}>
      <StatusStrip
        state={stripState}
        latencyMs={latencyMs}
        reducedMotion={reducedMotion}
        wordCount={submittedWords.length}
        skippedCount={skippedWords.length}
        fingerspelledCount={fingerspelledCount}
      />

      <main className={styles.stage}>
        {isLoaded ? (
          <SkeletonCanvas
            landmarkNames={plan.units[0].sequence.landmark_names}
            timeline={timeline}
            isPlaying={isPlaying}
            loop={loop}
            reducedMotion={reducedMotion}
            onEnded={handleEnded}
            onFrameChange={handleFrameChange}
          />
        ) : (
          <StageMessage
            isLoading={isLoading}
            sequenceAborted={sequenceAborted}
            abortMessage={abortMessage}
            missingMessage={missingMessage}
            missingDetail={missingDetail}
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
        skippedBanner={skippedBanner}
        missingMessage={missingMessage}
        missingDetail={missingDetail}
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
function StageMessage({ isLoading, sequenceAborted, abortMessage, missingMessage, missingDetail }) {
  if (isLoading) {
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
        {missingDetail}
      </p>
    );
  }
  return <p className={styles.stageMessage}>ENTER A WORD BELOW TO BEGIN</p>;
}
