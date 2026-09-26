import { useEffect, useRef, useState } from "react";

import { fetchWordPoseResult } from "./usePoseSequence.js";

/**
 * @typedef {import("./usePoseSequence.js").PoseSequenceResult} PoseSequenceResult
 */

/**
 * @typedef {Object} PoseSequencesResult
 * @property {"idle"|"loading"|"ready"|"error"} status - `"error"` here
 *   means the *manifest itself* failed to fetch -- a sequence-level
 *   failure, distinct from any one word's own `not_found`/`error` entry in
 *   `results` (see `frontend/MULTIWORD_PLAN.md` Section 2/4).
 * @property {Omit<PoseSequenceResult, "fetchMs">[]} results - One entry per
 *   input word, in order, using the same discriminated union
 *   `usePoseSequence` returns (minus `idle`/`loading`, which don't apply to
 *   a single already-resolved word within a batch).
 * @property {number|null} fetchMs - Wall-clock time from the first fetch
 *   dispatched to every response having arrived and been parsed, or `null`
 *   before that's happened.
 * @property {string|null} message - Set only when `status: "error"`.
 * @property {boolean} [manifestMissing] - Present only when `status:
 *   "error"`: `true` if the manifest itself 404'd (that library isn't built
 *   on this machine -- for the fingerspelling alphabet, "not available"),
 *   `false` for any other failure. Lets a caller treat a library that
 *   doesn't exist differently from one that exists but failed
 *   (`pose_library/FINGERSPELLING_PLAN.md` Section 3).
 */

/**
 * Looks up a short sequence of gloss words against the pose library in one
 * batch, over the same dev-server `/poses/*` stopgap `usePoseSequence` uses.
 *
 * Per `frontend/MULTIWORD_PLAN.md` Section 2: fetches `/poses/manifest.json`
 * **once** for the whole sequence (not once per word), then fetches each
 * word's own `/poses/<word>.json` in parallel -- every request (the
 * manifest and all N word lookups) is dispatched simultaneously, same
 * maximal-parallelism approach `usePoseSequence` already uses for its one
 * word, just generalized to N.
 *
 * The same hook serves the fingerspelling alphabet: `basePath:
 * "/fingerspelling"` looks letters up against
 * `pose_library/fingerspelling/poses/` instead (see `vite.config.js`) --
 * a letter is just a very short "word" with the same JSON shape, per
 * `pose_library/FINGERSPELLING_PLAN.md` Section 3.
 *
 * @param {string[]} words - Already-normalized (trimmed/lowercased) gloss
 *   words (or letters), in order. Pass an empty array to reset to
 *   `{status: "idle"}`.
 * @param {{basePath?: string}} [options] - `basePath` is the URL prefix
 *   serving `manifest.json` and `<word>.json`; defaults to `"/poses"`.
 * @returns {PoseSequencesResult} The current batch lookup result.
 */
export function usePoseSequences(words, { basePath = "/poses" } = {}) {
  const [result, setResult] = useState({
    status: "idle",
    results: [],
    fetchMs: null,
    message: null,
  });
  const requestIdRef = useRef(0);

  // Depend on the words joined into one string -- an array literal from the
  // caller is a new reference every render, which would otherwise re-run
  // this effect (and re-fetch) even when the actual words haven't changed.
  // NUL can't occur inside a word, so the join is collision-free; written
  // as an escape, not a raw byte, so git still treats this file as text.
  const wordsKey = words.join("\u0000");

  useEffect(() => {
    if (words.length === 0) {
      setResult({ status: "idle", results: [], fetchMs: null, message: null });
      return undefined;
    }

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setResult({ status: "loading", results: [], fetchMs: null, message: null });

    const startedAt = performance.now();

    async function run() {
      try {
        // Every request -- the manifest and all N word lookups -- is
        // dispatched here, simultaneously.
        const manifestFetch = fetch(`${basePath}/manifest.json`);
        const sequenceFetches = words.map((word) =>
          fetch(`${basePath}/${encodeURIComponent(word)}.json`)
        );

        const manifestRes = await manifestFetch;

        if (requestIdRef.current !== requestId) {
          return; // A newer batch superseded this one; drop the result.
        }

        if (manifestRes.status === 404) {
          setResult({
            status: "error",
            results: [],
            fetchMs: null,
            message: "Pose library not found -- it may not be built on this machine yet.",
            manifestMissing: true,
          });
          return;
        }
        if (!manifestRes.ok) {
          setResult({
            status: "error",
            results: [],
            fetchMs: null,
            message: `Manifest fetch failed (HTTP ${manifestRes.status}).`,
            manifestMissing: false,
          });
          return;
        }

        const manifest = await manifestRes.json();
        const results = await Promise.all(
          words.map((word, i) => fetchWordPoseResult(word, manifest, sequenceFetches[i]))
        );

        if (requestIdRef.current !== requestId) {
          return;
        }

        setResult({
          status: "ready",
          results,
          fetchMs: performance.now() - startedAt,
          message: null,
        });
      } catch (err) {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setResult({
          status: "error",
          results: [],
          fetchMs: null,
          message: err instanceof Error ? err.message : "Unknown fetch error.",
          manifestMissing: false,
        });
      }
    }

    run();

    return undefined;
    // `wordsKey` (plus `basePath`) is the intentional, stable dependency;
    // `words` itself is a new array reference every render (see above).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wordsKey, basePath]);

  return result;
}
