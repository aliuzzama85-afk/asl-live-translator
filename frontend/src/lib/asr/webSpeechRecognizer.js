/**
 * The ASR adapter interface, and its Web Speech API implementation.
 *
 * Everything above this file (phrase chunking, translation, the queue,
 * playback) depends only on the interface below, never on the Web Speech API
 * itself. Moving to a streaming provider (Deepgram, AssemblyAI) means writing
 * one more factory with the same shape, per `pipeline/STAGE1_2_PLAN.md`
 * Section 1. The dev-only scripted recognizer (`scriptedRecognizer.js`) and
 * the tests' fake recognizer are two more implementations, which is what
 * shows the boundary holds.
 */

/**
 * @typedef {{type: "listening"}
 *   | {type: "partial", text: string}
 *   | {type: "final", text: string, at: number}
 *   | {type: "error", code: string, message: string}
 *   | {type: "end"}} AsrEvent
 * `listening`: audio capture has started. `partial`: the current
 * not-yet-final text ("" clears it). `final`: one finished utterance, with
 * `at` = `performance.now()` when it went final. `error`: `code` uses the Web
 * Speech API's names (`not-allowed`, `audio-capture`, `network`, `no-speech`,
 * `aborted`, ...); other adapters should map onto the same names. `end`: the
 * session stopped (it may be restarted with `start()`).
 */

/**
 * @typedef {Object} Recognizer
 * @property {() => void} start - Starts (or restarts, after `end`) listening.
 * @property {() => void} stop - Stops immediately and discards anything
 *   not yet final. The caller flushes the last partial text itself, so
 *   nothing arrives twice.
 */

/**
 * @typedef {(onEvent: (event: AsrEvent) => void) => Recognizer} RecognizerFactory
 */

/**
 * Finds the browser's speech recognition constructor.
 *
 * @param {object} [win] - The global object to look on (default `window`).
 * @returns {Function|null} `SpeechRecognition` or `webkitSpeechRecognition`,
 *   or `null` where the API doesn't exist (e.g. Firefox).
 */
export function getSpeechRecognitionCtor(win = globalThis.window) {
  return win?.SpeechRecognition ?? win?.webkitSpeechRecognition ?? null;
}

/**
 * Creates a recognizer backed by the Web Speech API.
 *
 * Continuous, with interim results, `en-US`. The browser owns the
 * microphone. Note: in Chrome (and Edge) recognition runs on the browser
 * vendor's speech service, not on-device; the UI discloses this before
 * the first use (STAGE1_2_PLAN.md Section 3).
 *
 * @param {(event: AsrEvent) => void} onEvent
 * @param {{lang?: string, Ctor?: Function|null, now?: () => number}} [options]
 * @returns {Recognizer}
 * @throws {Error} If the Web Speech API isn't available.
 */
export function createWebSpeechRecognizer(
  onEvent,
  { lang = "en-US", Ctor = getSpeechRecognitionCtor(), now = () => performance.now() } = {}
) {
  if (!Ctor) {
    throw new Error("The Web Speech API isn't available in this browser.");
  }
  let recognition = null;

  function build() {
    const r = new Ctor();
    r.lang = lang;
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;
    r.onstart = () => onEvent({ type: "listening" });
    r.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) {
          onEvent({ type: "final", text, at: now() });
        } else {
          interim += text;
        }
      }
      onEvent({ type: "partial", text: interim });
    };
    r.onerror = (event) =>
      onEvent({ type: "error", code: event.error ?? "unknown", message: event.message ?? "" });
    r.onend = () => onEvent({ type: "end" });
    return r;
  }

  return {
    start() {
      recognition = recognition ?? build();
      recognition.start();
    },
    stop() {
      // abort(), not stop(): stop() would still deliver a final result for
      // audio already heard, duplicating the partial text the caller
      // flushes itself.
      recognition?.abort();
    },
  };
}
