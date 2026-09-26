/**
 * Test double for the ASR adapter interface (`lib/asr/webSpeechRecognizer.js`).
 *
 * The code under test gets this factory in place of the Web Speech API and
 * can't tell the difference. Tests drive it by emitting adapter events
 * directly (`emit`), so every recognizer behavior (results, errors, the
 * browser ending a session) is reproducible without a microphone.
 */
export function createFakeAsr() {
  const instances = [];
  const createRecognizer = (onEvent) => {
    const recognizer = {
      onEvent,
      starts: 0,
      stops: 0,
      start() {
        recognizer.starts += 1;
      },
      stop() {
        recognizer.stops += 1;
      },
    };
    instances.push(recognizer);
    return recognizer;
  };
  return {
    asr: { supported: true, kind: "fake", createRecognizer },
    instances,
    get last() {
      return instances[instances.length - 1];
    },
    emit(event) {
      instances[instances.length - 1].onEvent(event);
    },
  };
}
