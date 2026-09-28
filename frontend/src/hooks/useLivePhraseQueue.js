import { useCallback, useEffect, useRef, useState } from "react";

import { splitIntoPhrases } from "../lib/phrases.js";

/** Most glossed phrases allowed to wait for the avatar. Past this, the oldest
 * waiting phrase is dropped, with a visible notice. An interpreter who falls
 * behind drops content rather than falling further behind for good
 * (`pipeline/STAGE1_2_PLAN.md` Section 4). */
export const MAX_BACKLOG = 3;

/** Phrases kept in the on-screen history (the transcript line shows the
 * newest; older ones only matter for status bookkeeping). */
const HISTORY_LIMIT = 20;

/**
 * @typedef {"waiting"|"translating"|"queued"|"signing"|"done"|"not_translated"|"nothing_to_sign"|"dropped"} PhraseStatus
 */

/**
 * @typedef {Object} PhraseEntry
 * @property {number} phraseId
 * @property {string} text - What was heard or typed (sanitized).
 * @property {"speech"|"typed"} source
 * @property {PhraseStatus} status
 * @property {string} [reason] - For `not_translated`: the gloss client's error code.
 * @property {string} [gloss] - The model's raw gloss, once translated.
 * @property {string[]} [dropped] - Gloss tokens that won't be signed (pronoun
 *   markers, stop-list words, punctuation), once translated.
 */

/**
 * @typedef {Object} GlossedPhrase
 * @property {number} phraseId
 * @property {string} text
 * @property {string} gloss
 * @property {string[]} words - Lookup words, ready for `setSubmittedWords`.
 * @property {string[]} dropped
 * @property {number} finalAt - When the recognizer finalized it, or when it
 *   was submitted if typed (performance.now()).
 * @property {"speech"|"typed"} source
 */

/**
 * The live phrase queue: final transcripts (spoken, or typed in TRANSLATE
 * mode) in, glossed phrases out, in the order they were given.
 *
 * Phrases are translated one at a time, in order (the server is
 * single-threaded anyway), so results arrive in speaking order without any
 * reordering. The owner takes the next glossed phrase with `takeNext()` when
 * the avatar is idle, then reports `markDone()` when that phrase has
 * finished.
 *
 * @param {{glossPhrase: (text: string, phraseId: number) => Promise<
 *   {ok: true, result: object} | {ok: false, code: string, message: string}>}} options
 * @returns {{
 *   entries: PhraseEntry[],
 *   readyCount: number,
 *   translating: boolean,
 *   waitingCount: number,
 *   droppedCount: number,
 *   enqueueFinal: (final: {text: string, at: number, source?: "speech"|"typed"}) => void,
 *   takeNext: () => GlossedPhrase | null,
 *   markDone: (phraseId: number) => void,
 *   resetNotices: () => void,
 * }}
 */
export function useLivePhraseQueue({ glossPhrase }) {
  const [entries, setEntries] = useState([]);
  const [readyCount, setReadyCount] = useState(0);
  const [translating, setTranslating] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [droppedCount, setDroppedCount] = useState(0);

  const nextIdRef = useRef(1);
  const pendingRef = useRef([]); // awaiting translation, in order
  const readyRef = useRef([]); // glossed, waiting for the avatar
  const busyRef = useRef(false);
  const mountedRef = useRef(true);
  const glossRef = useRef(glossPhrase);
  glossRef.current = glossPhrase;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const setStatus = useCallback((phraseId, status, fields = {}) => {
    setEntries((prev) =>
      prev.map((e) => (e.phraseId === phraseId ? { ...e, ...fields, status } : e))
    );
  }, []);

  const pump = useCallback(async () => {
    if (busyRef.current) return;
    const next = pendingRef.current.shift();
    setPendingCount(pendingRef.current.length);
    if (!next) return;
    busyRef.current = true;
    setTranslating(true);
    setStatus(next.phraseId, "translating");

    const response = await glossRef.current(next.text, next.phraseId);
    if (!mountedRef.current) return;

    const translated = response.ok
      ? { gloss: response.result.gloss, dropped: response.result.dropped ?? [] }
      : null;
    if (!response.ok) {
      setStatus(next.phraseId, "not_translated", { reason: response.code });
    } else if (response.result.words.length === 0) {
      setStatus(next.phraseId, "nothing_to_sign", translated);
    } else {
      readyRef.current.push({
        phraseId: next.phraseId,
        text: next.text,
        gloss: response.result.gloss,
        words: response.result.words,
        dropped: response.result.dropped,
        finalAt: next.finalAt,
        source: next.source,
      });
      setStatus(next.phraseId, "queued", translated);
      const overflow = readyRef.current.length - MAX_BACKLOG;
      if (overflow > 0) {
        const droppedPhrases = readyRef.current.splice(0, overflow);
        droppedPhrases.forEach((p) => setStatus(p.phraseId, "dropped"));
        setDroppedCount((n) => n + overflow);
      }
      setReadyCount(readyRef.current.length);
    }

    busyRef.current = false;
    setTranslating(false);
    pump();
  }, [setStatus]);

  const enqueueFinal = useCallback(
    ({ text, at, source = "speech" }) => {
      const phrases = splitIntoPhrases(text);
      if (phrases.length === 0) return; // empty/whitespace: nothing to sign
      const added = phrases.map((phraseText) => ({
        phraseId: nextIdRef.current++,
        text: phraseText,
        finalAt: at,
        source,
      }));
      pendingRef.current.push(...added);
      setPendingCount(pendingRef.current.length);
      setEntries((prev) =>
        [
          ...prev,
          ...added.map((p) => ({
            phraseId: p.phraseId,
            text: p.text,
            source: p.source,
            status: "waiting",
          })),
        ].slice(-HISTORY_LIMIT)
      );
      pump();
    },
    [pump]
  );

  const takeNext = useCallback(() => {
    const next = readyRef.current.shift() ?? null;
    setReadyCount(readyRef.current.length);
    if (next) setStatus(next.phraseId, "signing");
    return next;
  }, [setStatus]);

  const markDone = useCallback((phraseId) => setStatus(phraseId, "done"), [setStatus]);
  const resetNotices = useCallback(() => setDroppedCount(0), []);

  return {
    entries,
    readyCount,
    translating,
    waitingCount: readyCount + pendingCount + (translating ? 1 : 0),
    droppedCount,
    enqueueFinal,
    takeNext,
    markDone,
    resetNotices,
  };
}
