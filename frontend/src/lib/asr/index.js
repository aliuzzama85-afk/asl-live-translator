/**
 * Picks the ASR implementation for this page load.
 *
 * The one place a future provider (Deepgram, AssemblyAI) gets wired in: add
 * its factory here. Nothing downstream changes (`pipeline/STAGE1_2_PLAN.md`
 * Section 1).
 */

import { createWebSpeechRecognizer, getSpeechRecognitionCtor } from "./webSpeechRecognizer.js";
import { scriptedFactoryFromParams } from "./scriptedRecognizer.js";

/**
 * @typedef {Object} AsrChoice
 * @property {boolean} supported - Whether live speech can work at all here.
 * @property {"web-speech"|"scripted"|"none"} kind
 * @property {import("./webSpeechRecognizer.js").RecognizerFactory|null} createRecognizer
 */

/**
 * @param {{win?: object, dev?: boolean}} [options]
 * @returns {AsrChoice} The dev-only scripted recognizer when `?asr=fake` is
 *   set in a dev build; otherwise the Web Speech API if the browser has it;
 *   otherwise unsupported.
 */
export function selectAsr({ win = globalThis.window, dev = import.meta.env.DEV } = {}) {
  const params = new URLSearchParams(win?.location?.search ?? "");
  if (dev && params.get("asr") === "fake") {
    return {
      supported: true,
      kind: "scripted",
      createRecognizer: scriptedFactoryFromParams(params),
    };
  }
  const Ctor = getSpeechRecognitionCtor(win);
  if (!Ctor) {
    return { supported: false, kind: "none", createRecognizer: null };
  }
  return {
    supported: true,
    kind: "web-speech",
    createRecognizer: (onEvent) => createWebSpeechRecognizer(onEvent, { Ctor }),
  };
}
