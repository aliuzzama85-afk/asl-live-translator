import { useCallback, useEffect, useRef, useState } from "react";

/**
 * User-facing messages for recognizer failures (`pipeline/STAGE1_2_PLAN.md`
 * Section 6). Keyed by the adapter's error codes (Web Speech API names), plus
 * `restart_storm` and `start_failed`, raised by this hook itself.
 */
export const LIVE_ERROR_MESSAGES = {
  "not-allowed": "MICROPHONE BLOCKED — ALLOW IT IN YOUR BROWSER'S SITE SETTINGS, THEN TRY AGAIN",
  "service-not-allowed":
    "MICROPHONE BLOCKED — ALLOW IT IN YOUR BROWSER'S SITE SETTINGS, THEN TRY AGAIN",
  "audio-capture": "NO MICROPHONE FOUND (OR IT'S IN USE BY ANOTHER APP)",
  network: "SPEECH SERVICE UNREACHABLE — CHECK YOUR INTERNET CONNECTION",
  "language-not-supported": "THIS BROWSER CAN'T RECOGNIZE US ENGLISH SPEECH",
  restart_storm: "SPEECH RECOGNITION KEEPS STOPPING — TURN THE MIC OFF AND ON TO RETRY",
  start_failed: "SPEECH RECOGNITION COULDN'T START — TURN THE MIC OFF AND ON TO RETRY",
};

const GENERIC_MESSAGE = "SPEECH RECOGNITION STOPPED UNEXPECTEDLY — TURN THE MIC ON TO RETRY";

/** Not failures: `no-speech` is silence (the session just restarts), and
 * `aborted` is what our own `stop()` produces. */
const IGNORED_ERRORS = new Set(["no-speech", "aborted"]);

/** More than this many automatic restarts within `RESTART_WINDOW_MS`, with no
 * result in between, means the recognizer is failing, not idling (Section 2). */
export const MAX_RESTARTS = 3;
export const RESTART_WINDOW_MS = 10000;

/**
 * @typedef {"off"|"starting"|"listening"|"error"} TranscriptionStatus
 */

/**
 * Runs a recognizer (any `RecognizerFactory`; see
 * `lib/asr/webSpeechRecognizer.js`) for as long as live mode wants it.
 *
 * The browser ends continuous sessions by itself after silence or a time
 * limit, so an `end` while still active is answered with an immediate
 * restart, unless restarts pile up with no results (see `MAX_RESTARTS`).
 * `stop()` flushes whatever partial text was showing as one last final
 * result, so words spoken just before the mic goes off aren't lost.
 *
 * @param {{
 *   createRecognizer: import("../lib/asr/webSpeechRecognizer.js").RecognizerFactory | null,
 *   onFinal: (final: {text: string, at: number, flushed?: boolean}) => void,
 *   now?: () => number,
 * }} options
 * @returns {{
 *   status: TranscriptionStatus,
 *   partial: string,
 *   error: {code: string, message: string} | null,
 *   start: () => void,
 *   stop: () => void,
 *   clearError: () => void,
 * }}
 */
export function useLiveTranscription({ createRecognizer, onFinal, now = () => performance.now() }) {
  const [status, setStatus] = useState("off");
  const [partial, setPartial] = useState("");
  const [error, setError] = useState(null);

  const recognizerRef = useRef(null);
  const activeRef = useRef(false);
  const partialRef = useRef("");
  const restartsRef = useRef([]);
  const onFinalRef = useRef(onFinal);
  const nowRef = useRef(now);
  onFinalRef.current = onFinal;
  nowRef.current = now;

  const fail = useCallback((code) => {
    activeRef.current = false;
    partialRef.current = "";
    recognizerRef.current?.stop();
    recognizerRef.current = null;
    setPartial("");
    setStatus("error");
    setError({ code, message: LIVE_ERROR_MESSAGES[code] ?? GENERIC_MESSAGE });
  }, []);

  const handleEvent = useCallback(
    (event) => {
      if (!activeRef.current) return; // late events after stop()/fail()
      switch (event.type) {
        case "listening":
          setStatus("listening");
          break;
        case "partial":
          partialRef.current = event.text;
          setPartial(event.text);
          break;
        case "final":
          restartsRef.current = [];
          partialRef.current = "";
          setPartial("");
          onFinalRef.current({ text: event.text, at: event.at });
          break;
        case "error":
          if (!IGNORED_ERRORS.has(event.code)) fail(event.code);
          break;
        case "end": {
          const t = nowRef.current();
          const recent = [...restartsRef.current, t].filter((r) => t - r <= RESTART_WINDOW_MS);
          restartsRef.current = recent;
          if (recent.length > MAX_RESTARTS) {
            fail("restart_storm");
            return;
          }
          try {
            recognizerRef.current?.start();
          } catch {
            fail("start_failed");
          }
          break;
        }
        default:
          break;
      }
    },
    [fail]
  );

  const start = useCallback(() => {
    if (activeRef.current || !createRecognizer) return;
    activeRef.current = true;
    restartsRef.current = [];
    partialRef.current = "";
    setError(null);
    setPartial("");
    setStatus("starting");
    try {
      recognizerRef.current = createRecognizer(handleEvent);
      recognizerRef.current.start();
    } catch {
      fail("start_failed");
    }
  }, [createRecognizer, handleEvent, fail]);

  const stop = useCallback(() => {
    if (!activeRef.current) {
      setStatus((s) => (s === "error" ? s : "off"));
      return;
    }
    activeRef.current = false;
    const pending = partialRef.current;
    partialRef.current = "";
    recognizerRef.current?.stop();
    recognizerRef.current = null;
    setPartial("");
    setStatus("off");
    if (pending.trim()) {
      onFinalRef.current({ text: pending, at: nowRef.current(), flushed: true });
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
    setStatus((s) => (s === "error" ? "off" : s));
  }, []);

  useEffect(
    () => () => {
      activeRef.current = false;
      recognizerRef.current?.stop();
    },
    []
  );

  return { status, partial, error, start, stop, clearError };
}
