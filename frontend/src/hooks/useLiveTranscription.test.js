import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";

import {
  LIVE_ERROR_MESSAGES,
  MAX_RESTARTS,
  RESTART_WINDOW_MS,
  useLiveTranscription,
} from "./useLiveTranscription.js";
import { createFakeAsr } from "../test/fakeRecognizer.js";

function setup() {
  const fake = createFakeAsr();
  const onFinal = vi.fn();
  const clock = { t: 1000 };
  const hook = renderHook(() =>
    useLiveTranscription({
      createRecognizer: fake.asr.createRecognizer,
      onFinal,
      now: () => clock.t,
    })
  );
  const emit = (event) => act(() => fake.emit(event));
  return { fake, onFinal, clock, hook, emit };
}

describe("useLiveTranscription", () => {
  it("starts the recognizer and reports listening once audio starts", () => {
    const { fake, hook, emit } = setup();
    act(() => hook.result.current.start());
    expect(hook.result.current.status).toBe("starting");
    expect(fake.last.starts).toBe(1);
    emit({ type: "listening" });
    expect(hook.result.current.status).toBe("listening");
  });

  it("shows partial text, and passes each final result on", () => {
    const { onFinal, hook, emit } = setup();
    act(() => hook.result.current.start());
    emit({ type: "partial", text: "where is" });
    expect(hook.result.current.partial).toBe("where is");
    emit({ type: "final", text: "where is the bathroom", at: 42 });
    expect(onFinal).toHaveBeenCalledWith({ text: "where is the bathroom", at: 42 });
    expect(hook.result.current.partial).toBe("");
  });

  it("restarts the recognizer when the browser ends the session by itself", () => {
    const { fake, hook, emit } = setup();
    act(() => hook.result.current.start());
    emit({ type: "end" });
    expect(fake.last.starts).toBe(2);
    expect(hook.result.current.status).toBe("starting");
  });

  it("treats silence (no-speech) as normal, not an error", () => {
    const { hook, emit } = setup();
    act(() => hook.result.current.start());
    emit({ type: "error", code: "no-speech", message: "" });
    expect(hook.result.current.error).toBeNull();
  });

  it("stops with an error after more than 3 restarts in 10s with no result", () => {
    const { fake, clock, hook, emit } = setup();
    act(() => hook.result.current.start());
    for (let i = 0; i < MAX_RESTARTS; i += 1) {
      clock.t += 1000;
      emit({ type: "end" });
    }
    expect(hook.result.current.status).not.toBe("error");
    clock.t += 1000;
    emit({ type: "end" });
    expect(hook.result.current.status).toBe("error");
    expect(hook.result.current.error.message).toBe(LIVE_ERROR_MESSAGES.restart_storm);
    expect(fake.last.stops).toBe(1);
  });

  it("a result resets the restart count, and old restarts age out", () => {
    const { clock, hook, emit } = setup();
    act(() => hook.result.current.start());
    for (let i = 0; i < MAX_RESTARTS; i += 1) emit({ type: "end" });
    emit({ type: "final", text: "hello", at: 1 });
    for (let i = 0; i < MAX_RESTARTS; i += 1) emit({ type: "end" });
    expect(hook.result.current.status).not.toBe("error");
    clock.t += RESTART_WINDOW_MS + 1;
    emit({ type: "end" });
    expect(hook.result.current.status).not.toBe("error");
  });

  it.each([
    ["not-allowed", LIVE_ERROR_MESSAGES["not-allowed"]],
    ["service-not-allowed", LIVE_ERROR_MESSAGES["not-allowed"]],
    ["audio-capture", LIVE_ERROR_MESSAGES["audio-capture"]],
    ["network", LIVE_ERROR_MESSAGES.network],
    ["something-new", "SPEECH RECOGNITION STOPPED UNEXPECTEDLY — TURN THE MIC ON TO RETRY"],
  ])("maps the %s error to its message and turns the mic off", (code, message) => {
    const { fake, hook, emit } = setup();
    act(() => hook.result.current.start());
    emit({ type: "error", code, message: "" });
    expect(hook.result.current.status).toBe("error");
    expect(hook.result.current.error).toEqual({ code, message });
    expect(fake.last.stops).toBe(1);
    // No restart loop after a real error.
    emit({ type: "end" });
    expect(fake.last.starts).toBe(1);
  });

  it("stop() flushes the words still showing as one last final result", () => {
    const { onFinal, clock, hook, emit } = setup();
    act(() => hook.result.current.start());
    emit({ type: "partial", text: "see you tomorrow" });
    clock.t = 5000;
    act(() => hook.result.current.stop());
    expect(onFinal).toHaveBeenCalledWith({ text: "see you tomorrow", at: 5000, flushed: true });
    expect(hook.result.current.status).toBe("off");
  });

  it("ignores events that arrive after stop()", () => {
    const { onFinal, fake, hook, emit } = setup();
    act(() => hook.result.current.start());
    act(() => hook.result.current.stop());
    emit({ type: "final", text: "late", at: 1 });
    emit({ type: "end" });
    expect(onFinal).not.toHaveBeenCalled();
    expect(fake.last.starts).toBe(1);
  });

  it("reports a recognizer that throws on start", () => {
    const onFinal = vi.fn();
    const hook = renderHook(() =>
      useLiveTranscription({
        createRecognizer: () => ({
          start() {
            throw new Error("InvalidStateError");
          },
          stop() {},
        }),
        onFinal,
      })
    );
    act(() => hook.result.current.start());
    expect(hook.result.current.error.code).toBe("start_failed");
  });
});
