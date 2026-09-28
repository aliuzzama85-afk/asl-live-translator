import { useCallback, useEffect, useRef, useState } from "react";

import * as defaultGlossClient from "../lib/glossClient.js";
import { useLivePhraseQueue } from "./useLivePhraseQueue.js";
import { useLiveTranscription } from "./useLiveTranscription.js";

/** localStorage key recording that the speech-service privacy notice was
 * acknowledged (`pipeline/STAGE1_2_PLAN.md` Section 5). */
export const PRIVACY_ACK_KEY = "asl-live-translator.speechPrivacyAck";

/** Appended to the "service down" messages: typed input still works in
 * EXACT WORDS mode, which never calls the gloss model. */
const EXACT_WORDS_HINT = "EXACT WORDS MODE WORKS WITHOUT IT";

export const SERVICE_MESSAGES = {
  connecting: "CONNECTING TO TRANSLATION SERVICE…",
  loading: "TRANSLATION SERVICE STARTING…",
  unreachable: `TRANSLATION SERVICE NOT RUNNING — START IT WITH: python -m pipeline.gloss_server (OR npm run dev:live). ${EXACT_WORDS_HINT}`,
  failed: `TRANSLATION SERVICE FAILED TO LOAD ITS MODEL — SEE ITS LOG. ${EXACT_WORDS_HINT}`,
};

function readAck(storage) {
  try {
    return storage?.getItem(PRIVACY_ACK_KEY) === "1";
  } catch {
    return false; // storage blocked: ask again, which is the safe default
  }
}

function writeAck(storage) {
  try {
    storage?.setItem(PRIVACY_ACK_KEY, "1");
  } catch {
    // Not remembered; the notice will simply be shown again next time.
  }
}

/**
 * Live-mode orchestration: the privacy notice, waiting for the gloss server,
 * running the recognizer, and the phrase queue. `App` only has to take
 * glossed phrases from `queue` and feed them to the existing playback.
 *
 * Mic-on sequence: privacy acknowledged? → gloss server ready (retrying
 * while it starts; see `waitForGlossService`) → recognizer starts. Turning
 * the mic off at any point cancels the rest.
 *
 * Typed text in TRANSLATE mode (`submitTyped`) goes through the same gate
 * and the same queue: gloss server ready → `queue.enqueueFinal`, exactly as
 * a spoken final result does. If the server isn't there, the same service
 * banner shows and nothing is signed -- typed text is never played
 * untranslated as if it had been translated.
 *
 * @param {{
 *   asr: import("../lib/asr/index.js").AsrChoice,
 *   glossClient?: {glossPhrase: Function, waitForGlossService: Function},
 *   storage?: Storage | null,
 * }} options
 */
export function useLiveMode({
  asr,
  glossClient = defaultGlossClient,
  storage = globalThis.localStorage ?? null,
}) {
  const [micRequested, setMicRequested] = useState(false);
  const [privacyPrompt, setPrivacyPrompt] = useState(false);
  const [serviceState, setServiceState] = useState("idle"); // idle|connecting|loading|ready|unreachable|failed
  const [typedWaiting, setTypedWaiting] = useState(false);
  // The one in-flight wait for the gloss server, shared by the mic and typed
  // input so they never run two health-poll loops at once.
  const serviceWaitRef = useRef(null); // {controller, promise}
  const serviceSeenReadyRef = useRef(false);
  const micWantedRef = useRef(false);
  const typedPendingRef = useRef([]); // typed text waiting for the service, in order

  const glossPhrase = useCallback(
    (text, phraseId) => glossClient.glossPhrase(text, phraseId),
    [glossClient]
  );
  const queue = useLivePhraseQueue({ glossPhrase });
  const transcription = useLiveTranscription({
    createRecognizer: asr.createRecognizer,
    onFinal: queue.enqueueFinal,
  });

  const cancelServiceWait = useCallback(() => {
    serviceWaitRef.current?.controller.abort();
    serviceWaitRef.current = null;
  }, []);

  /** Resolves when the gloss server is ready (or definitely isn't), joining
   * the wait already in flight, if any. Once the server has been seen ready,
   * a later "no answer" means it was stopped, not that it's still starting,
   * so that's reported at once instead of after the startup grace period. */
  const waitForService = useCallback(() => {
    if (serviceWaitRef.current) return serviceWaitRef.current.promise;
    const controller = new AbortController();
    setServiceState("connecting");
    const promise = glossClient
      .waitForGlossService({
        signal: controller.signal,
        onProgress: (state) => {
          if (!controller.signal.aborted) setServiceState(state);
        },
        ...(serviceSeenReadyRef.current ? { graceMs: 0 } : {}),
      })
      .then((result) => {
        if (serviceWaitRef.current?.controller === controller) serviceWaitRef.current = null;
        if (controller.signal.aborted || result.state === "cancelled") {
          return { state: "cancelled" };
        }
        if (result.state === "ready") serviceSeenReadyRef.current = true;
        setServiceState(result.state);
        return result;
      });
    serviceWaitRef.current = { controller, promise };
    return promise;
  }, [glossClient]);

  const beginStartup = useCallback(async () => {
    const result = await waitForService();
    if (!micWantedRef.current || result.state === "cancelled") return;
    if (result.state === "ready") {
      transcription.start();
    } else {
      micWantedRef.current = false;
      setMicRequested(false);
    }
  }, [transcription, waitForService]);

  const micOn = useCallback(() => {
    if (!asr.supported) return;
    transcription.clearError();
    queue.resetNotices();
    micWantedRef.current = true;
    setMicRequested(true);
    if (!readAck(storage)) {
      setPrivacyPrompt(true);
      return;
    }
    beginStartup();
  }, [asr.supported, beginStartup, queue, storage, transcription]);

  const micOff = useCallback(() => {
    micWantedRef.current = false;
    // Typed text may be waiting on the same service check; let it finish.
    if (typedPendingRef.current.length === 0) {
      cancelServiceWait();
      setServiceState((s) => (s === "ready" ? s : "idle"));
    }
    setPrivacyPrompt(false);
    setMicRequested(false);
    transcription.stop();
  }, [cancelServiceWait, transcription]);

  const toggleMic = useCallback(() => {
    if (micRequested) micOff();
    else micOn();
  }, [micOff, micOn, micRequested]);

  const acknowledgePrivacy = useCallback(() => {
    writeAck(storage);
    setPrivacyPrompt(false);
    beginStartup();
  }, [beginStartup, storage]);

  /**
   * Translates typed text (TRANSLATE mode). One submit is one final result,
   * split and sanitized by the queue exactly like speech. Waits for the gloss
   * server first (the same check the mic uses); if it isn't available, the
   * service banner says so and nothing is queued or signed.
   *
   * @param {string} text
   * @returns {Promise<void>}
   */
  const submitTyped = useCallback(
    async (text) => {
      if (!text.trim()) return;
      queue.resetNotices();
      typedPendingRef.current.push({ text, at: performance.now() });
      setTypedWaiting(true);
      const result = await waitForService();
      // Everything typed while this check ran is released together, in submit
      // order; a later caller joining the same check finds the list empty.
      const pending = typedPendingRef.current;
      typedPendingRef.current = [];
      setTypedWaiting(false);
      if (result.state !== "ready") return;
      pending.forEach((p) => queue.enqueueFinal({ ...p, source: "typed" }));
    },
    [queue, waitForService]
  );

  // A recognizer failure turns the mic off (the error stays on screen).
  useEffect(() => {
    if (transcription.status === "error") {
      micWantedRef.current = false;
      setMicRequested(false);
    }
  }, [transcription.status]);

  useEffect(() => cancelServiceWait, [cancelServiceWait]);

  const serviceProblem =
    serviceState === "unreachable" || serviceState === "failed"
      ? SERVICE_MESSAGES[serviceState]
      : null;

  return {
    supported: asr.supported,
    micRequested,
    listening: transcription.status === "listening",
    privacyPrompt,
    acknowledgePrivacy,
    cancelPrivacy: micOff,
    toggleMic,
    submitTyped,
    serviceState,
    serviceStartupMessage:
      (micRequested || typedWaiting) &&
      (serviceState === "connecting" || serviceState === "loading")
        ? SERVICE_MESSAGES[serviceState]
        : null,
    errorMessage: transcription.error?.message ?? serviceProblem,
    partial: transcription.partial,
    queue,
  };
}
