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
 * Resolves one word's already-in-flight `/poses/<word>.json` fetch against
 * an already-parsed manifest, into the shared ok/low_confidence/not_found/
 * error union (without `fetchMs` -- timing is a caller concern, since a
 * single lookup times itself while a batch times the whole batch as one
 * number, see `usePoseSequences.js`).
 *
 * Factored out of the original single-word fetch body (per
 * `frontend/MULTIWORD_PLAN.md` Section 2) so `usePoseSequence` and the
 * multi-word `usePoseSequences` share one implementation of "given a word's
 * response and a manifest, resolve ok/low_confidence/not_found/error"
 * instead of duplicating it. Deliberately does **not** fetch or classify
 * the *manifest* itself, and takes the sequence fetch as an
 * already-started `Promise` rather than calling `fetch` itself -- both so
 * each hook keeps full control of its own fetch timing/parallelism (this
 * hook still dispatches the manifest and word-JSON requests
 * simultaneously, exactly as before this extraction) and so a
 * manifest-fetch failure can be classified differently by each caller (this
 * hook folds it into the one word's own not_found/error; the batch hook
 * treats it as a sequence-level failure -- see PLAN Section 4).
 *
 * @param {string} word - Already-trimmed, already-lowercased.
 * @param {object} manifest - The already-parsed `/poses/manifest.json` body.
 * @param {Promise<Response>} sequenceFetch - An already-started `fetch()`
 *   call for `/poses/<word>.json`.
 * @returns {Promise<Omit<PoseSequenceResult, "fetchMs">>} A result without
 *   `fetchMs` (never `idle`/`loading`, which are caller-only states).
 */
export async function fetchWordPoseResult(word, manifest, sequenceFetch) {
  const sequenceRes = await sequenceFetch;
  const manifestEntry = manifest[word];

  if (sequenceRes.status === 404 || !manifestEntry) {
    return { status: "not_found", word };
  }
  if (!sequenceRes.ok) {
    return {
      status: "error",
      message: `Pose data fetch failed (HTTP ${sequenceRes.status}).`,
    };
  }

  const sequence = await sequenceRes.json();
  return {
    status: manifestEntry.low_confidence ? "low_confidence" : "ok",
    word,
    sequence,
    manifestEntry,
  };
}

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
        // Both requests are dispatched here, simultaneously -- awaiting the
        // manifest first below costs no extra network round-trip, since
        // `sequenceFetch` is already in flight.
        const manifestFetch = fetch("/poses/manifest.json");
        const sequenceFetch = fetch(`/poses/${encodeURIComponent(normalizedWord)}.json`);

        const manifestRes = await manifestFetch;

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
        const wordResult = await fetchWordPoseResult(normalizedWord, manifest, sequenceFetch);

        if (requestIdRef.current !== requestId) {
          return;
        }

        if (wordResult.status === "ok" || wordResult.status === "low_confidence") {
          setResult({ ...wordResult, fetchMs: performance.now() - startedAt });
        } else {
          setResult(wordResult);
        }
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
