import { describe, expect, it } from "vitest";

import { createWebSpeechRecognizer, getSpeechRecognitionCtor } from "./webSpeechRecognizer.js";
import { selectAsr } from "./index.js";

/** A stand-in for the browser's SpeechRecognition class. */
class FakeSpeechRecognition {
  static instances = [];

  constructor() {
    this.started = 0;
    this.aborted = 0;
    FakeSpeechRecognition.instances.push(this);
  }

  start() {
    this.started += 1;
  }

  abort() {
    this.aborted += 1;
  }
}

function result(transcript, isFinal) {
  return Object.assign([{ transcript }], { isFinal });
}

function setup() {
  FakeSpeechRecognition.instances = [];
  const events = [];
  const recognizer = createWebSpeechRecognizer((e) => events.push(e), {
    Ctor: FakeSpeechRecognition,
    now: () => 123,
  });
  recognizer.start();
  return { recognizer, events, native: FakeSpeechRecognition.instances[0] };
}

describe("createWebSpeechRecognizer", () => {
  it("configures continuous US-English recognition with interim results", () => {
    const { native } = setup();
    expect(native).toMatchObject({ lang: "en-US", continuous: true, interimResults: true });
    expect(native.started).toBe(1);
  });

  it("maps browser events onto the adapter's events", () => {
    const { native, events } = setup();
    native.onstart();
    native.onresult({
      resultIndex: 0,
      results: [result("where is the bathroom", true), result("can you", false)],
    });
    native.onerror({ error: "not-allowed" });
    native.onend();
    expect(events).toEqual([
      { type: "listening" },
      { type: "final", text: "where is the bathroom", at: 123 },
      { type: "partial", text: "can you" },
      { type: "error", code: "not-allowed", message: "" },
      { type: "end" },
    ]);
  });

  it("only reads results from resultIndex on", () => {
    const { native, events } = setup();
    native.onresult({
      resultIndex: 1,
      results: [result("already sent", true), result("new words", false)],
    });
    expect(events).toEqual([{ type: "partial", text: "new words" }]);
  });

  it("restarts on the same instance, and stop() aborts (no duplicate final)", () => {
    const { recognizer, native } = setup();
    recognizer.start();
    expect(FakeSpeechRecognition.instances).toHaveLength(1);
    expect(native.started).toBe(2);
    recognizer.stop();
    expect(native.aborted).toBe(1);
  });

  it("refuses to be created without the API", () => {
    expect(() => createWebSpeechRecognizer(() => {}, { Ctor: null })).toThrow(/isn't available/);
  });
});

describe("getSpeechRecognitionCtor / selectAsr", () => {
  it("finds the standard or webkit-prefixed constructor", () => {
    const A = function A() {};
    expect(getSpeechRecognitionCtor({ SpeechRecognition: A })).toBe(A);
    expect(getSpeechRecognitionCtor({ webkitSpeechRecognition: A })).toBe(A);
    expect(getSpeechRecognitionCtor({})).toBeNull();
  });

  it("is unsupported without the API (e.g. Firefox)", () => {
    expect(selectAsr({ win: { location: { search: "" } }, dev: true })).toMatchObject({
      supported: false,
      createRecognizer: null,
    });
  });

  it("uses the scripted recognizer for ?asr=fake only in dev builds", () => {
    const win = { location: { search: "?asr=fake" } };
    expect(selectAsr({ win, dev: true }).kind).toBe("scripted");
    expect(selectAsr({ win, dev: false }).kind).toBe("none");
  });

  it("uses the Web Speech API when present", () => {
    const win = { location: { search: "" }, webkitSpeechRecognition: FakeSpeechRecognition };
    expect(selectAsr({ win, dev: false }).kind).toBe("web-speech");
  });
});
