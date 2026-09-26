import { useCallback, useEffect, useRef, useState } from "react";

import * as defaultGlossClient from "../lib/glossClient.js";
import { useLivePhraseQueue } from "./useLivePhraseQueue.js";
import { useLiveTranscription } from "./useLiveTranscription.js";

/** localStorage key recording that the speech-service privacy notice was
 * acknowledged (`pipeline/STAGE1_2_PLAN.md` Section 5). */
export const PRIVACY_ACK_KEY = "asl-live-translator.speechPrivacyAck";

export const SERVICE_MESSAGES = {
  connecting: "CONNECTING TO TRANSLATION SERVICE…",
  loading: "TRANSLATION SERVICE STARTING…",
  unreachable:
    "TRANSLATION SERVICE NOT RUNNING — START IT WITH: python -m pipeline.gloss_server (OR npm run dev:live)",
  failed: "TRANSLATION SERVICE FAILED TO LOAD ITS MODEL — SEE ITS LOG",
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
  const startupRef = useRef(null);

  const glossPhrase = useCallback(
    (text, phraseId) => glossClient.glossPhrase(text, phraseId),
    [glossClient]
  );
  const queue = useLivePhraseQueue({ glossPhrase });
  const transcription = useLiveTranscription({
    createRecognizer: asr.createRecognizer,
    onFinal: queue.enqueueFinal,
  });

  const cancelStartup = useCallback(() => {
    startupRef.current?.abort();
    startupRef.current = null;
  }, []);

  const beginStartup = useCallback(async () => {
    cancelStartup();
    const controller = new AbortController();
    startupRef.current = controller;
    setServiceState("connecting");
    const result = await glossClient.waitForGlossService({
      signal: controller.signal,
      onProgress: (state) => setServiceState(state),
    });
    if (controller.signal.aborted || result.state === "cancelled") return;
    startupRef.current = null;
    setServiceState(result.state);
    if (result.state === "ready") {
      transcription.start();
    } else {
      setMicRequested(false);
    }
  }, [cancelStartup, glossClient, transcription]);

  const micOn = useCallback(() => {
    if (!asr.supported) return;
    transcription.clearError();
    queue.resetNotices();
    setMicRequested(true);
    if (!readAck(storage)) {
      setPrivacyPrompt(true);
      return;
    }
    beginStartup();
  }, [asr.supported, beginStartup, queue, storage, transcription]);

  const micOff = useCallback(() => {
    cancelStartup();
    setPrivacyPrompt(false);
    setMicRequested(false);
    setServiceState((s) => (s === "ready" ? s : "idle"));
    transcription.stop();
  }, [cancelStartup, transcription]);

  const toggleMic = useCallback(() => {
    if (micRequested) micOff();
    else micOn();
  }, [micOff, micOn, micRequested]);

  const acknowledgePrivacy = useCallback(() => {
    writeAck(storage);
    setPrivacyPrompt(false);
    beginStartup();
  }, [beginStartup, storage]);

  // A recognizer failure turns the mic off (the error stays on screen).
  useEffect(() => {
    if (transcription.status === "error") setMicRequested(false);
  }, [transcription.status]);

  useEffect(() => cancelStartup, [cancelStartup]);

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
    serviceState,
    serviceStartupMessage:
      micRequested && (serviceState === "connecting" || serviceState === "loading")
        ? SERVICE_MESSAGES[serviceState]
        : null,
    errorMessage: transcription.error?.message ?? serviceProblem,
    partial: transcription.partial,
    queue,
  };
}
