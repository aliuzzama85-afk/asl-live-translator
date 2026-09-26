/**
 * DEV-ONLY scripted recognizer: "speaks" a fixed list of phrases on a timer,
 * through the same adapter interface as the Web Speech recognizer. Selected
 * with the `?asr=fake` URL flag, and only in dev builds (`selectAsr` checks
 * `import.meta.env.DEV`), so the whole live path can be exercised in a real
 * browser without a microphone or a speech service
 * (`pipeline/STAGE1_2_PLAN.md` Section 8).
 *
 * URL options: `fakePhrases=a|b|c` replaces the default script; `burst=1`
 * shortens the pause between phrases, to exercise the 3-phrase backlog.
 */

/** Real phrases whose checkpoints_v2 gloss is known (Section 3), plus one with
 * an out-of-library word, so fingerspelling is exercised too. */
export const DEFAULT_SCRIPT = [
  "where is the bathroom",
  "can you help me find my phone",
  "I need a taxi",
];

/**
 * @param {(event: import("./webSpeechRecognizer.js").AsrEvent) => void} onEvent
 * @param {{phrases?: string[], wordMs?: number, pauseMs?: number,
 *   setTimeoutFn?: typeof setTimeout, clearTimeoutFn?: typeof clearTimeout,
 *   now?: () => number}} [options]
 * @returns {import("./webSpeechRecognizer.js").Recognizer}
 */
export function createScriptedRecognizer(
  onEvent,
  {
    phrases = DEFAULT_SCRIPT,
    wordMs = 180,
    pauseMs = 2500,
    setTimeoutFn = setTimeout,
    clearTimeoutFn = clearTimeout,
    now = () => performance.now(),
  } = {}
) {
  let timers = [];
  let started = false;

  function schedule(fn, at) {
    timers.push(setTimeoutFn(fn, at));
  }

  return {
    start() {
      if (started) return; // a restart after `end` never happens: this never ends
      started = true;
      let t = 300;
      schedule(() => onEvent({ type: "listening" }), t);
      for (const phrase of phrases) {
        const words = phrase.split(/\s+/).filter(Boolean);
        words.forEach((_, i) => {
          t += wordMs;
          const partial = words.slice(0, i + 1).join(" ");
          schedule(() => onEvent({ type: "partial", text: partial }), t);
        });
        t += wordMs;
        schedule(() => {
          onEvent({ type: "final", text: phrase, at: now() });
          onEvent({ type: "partial", text: "" });
        }, t);
        t += pauseMs;
      }
    },
    stop() {
      timers.forEach((id) => clearTimeoutFn(id));
      timers = [];
      started = false;
    },
  };
}

/**
 * Builds the scripted-recognizer factory from URL options.
 *
 * @param {URLSearchParams} params
 * @returns {import("./webSpeechRecognizer.js").RecognizerFactory}
 */
export function scriptedFactoryFromParams(params) {
  const custom = params.get("fakePhrases");
  const phrases = custom
    ? custom
        .split("|")
        .map((p) => p.trim())
        .filter(Boolean)
    : DEFAULT_SCRIPT;
  const pauseMs = params.get("burst") === "1" ? 150 : 2500;
  return (onEvent) => createScriptedRecognizer(onEvent, { phrases, pauseMs });
}
