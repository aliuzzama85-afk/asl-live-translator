import { useEffect, useRef, useState } from "react";

/**
 * @typedef {
 *   { status: "idle" } |
 *   { status: "loading" } |
 *   { status: "ok", word: string, sequence: object, manifestEntry: object, fetchMs: number } |
 *   { status: "low_confidence", word: string, sequence: object, manifestEntry: object, fetchMs: number } |
 *   { status: "not_found", word: string } |
 *   { status: "error", message: string }
 * } PoseSequenceResult
 */

/**
 * Looks up a single gloss word against the pose library, over the dev-server
 * `/poses/*` stopgap (see `vite.config.js`).
 *
 * Per `frontend/PLAN.md` Section 2/5, fetches *both* `/poses/manifest.json`
 * (for `low_confidence`/`quality_notes`/`dropped_frame_indices` -- data that
 * only lives in the manifest, never in the pose JSON itself) and
 * `/poses/<word>.json` (the actual `PoseSequence`), and returns a
 * discriminated union so callers don't need to separately check "found" vs.
 * "degraded" vs. "missing" vs. "network error".
 *
 * @param {string} word - The gloss word to look up. Lowercased/trimmed
 *   internally before use as a lookup key/URL segment. Pass an empty string
 *   to reset to `{status: "idle"}` (no active lookup).
 * @returns {PoseSequenceResult} The current lookup result. `fetchMs` is the
 *   wall-clock time from fetch start to both responses having arrived and
 *   been parsed (this stage's own fetch-to-first-frame latency, per
 *   PLAN.md's explicit scoping -- not a simulated pipeline number).
 */
export function usePoseSequence(word) {
  const [result, setResult] = useState({ status: "idle" });
  const requestIdRef = useRef(0);

  useEffect(() => {
    const normalizedWord = (word ?? "").trim().toLowerCase();
    if (!normalizedWord) {
      setResult({ status: "idle" });
      return undefined;
    }

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setResult({ status: "loading" });

    const startedAt = performance.now();

    async function run() {
      try {
        const [manifestRes, sequenceRes] = await Promise.all([
          fetch("/poses/manifest.json"),
          fetch(`/poses/${encodeURIComponent(normalizedWord)}.json`),
        ]);

        if (requestIdRef.current !== requestId) {
          return; // A newer lookup superseded this one; drop the result.
        }

        if (manifestRes.status === 404) {
          // The manifest itself is missing -- the library hasn't been
          // built on this machine yet. Per PLAN.md's gotchas section, this
          // must look identical to "word not found", not a crash.
          setResult({ status: "not_found", word: normalizedWord });
          return;
        }
        if (!manifestRes.ok) {
          setResult({
            status: "error",
            message: `Manifest fetch failed (HTTP ${manifestRes.status}).`,
          });
          return;
        }

        const manifest = await manifestRes.json();
        const manifestEntry = manifest[normalizedWord];

        if (sequenceRes.status === 404 || !manifestEntry) {
          setResult({ status: "not_found", word: normalizedWord });
          return;
        }
        if (!sequenceRes.ok) {
          setResult({
            status: "error",
            message: `Pose data fetch failed (HTTP ${sequenceRes.status}).`,
          });
          return;
        }

        const sequence = await sequenceRes.json();
        const fetchMs = performance.now() - startedAt;

        if (requestIdRef.current !== requestId) {
          return;
        }

        setResult({
          status: manifestEntry.low_confidence ? "low_confidence" : "ok",
          word: normalizedWord,
          sequence,
          manifestEntry,
          fetchMs,
        });
      } catch (err) {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setResult({
          status: "error",
          message: err instanceof Error ? err.message : "Unknown fetch error.",
        });
      }
    }

    run();

    return undefined;
  }, [word]);

  return result;
}
