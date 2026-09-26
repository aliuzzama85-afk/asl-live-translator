import { describe, expect, it } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import { MAX_BACKLOG, useLivePhraseQueue } from "./useLivePhraseQueue.js";

/** A gloss client whose responses the test releases one at a time. */
function controllableGloss() {
  const calls = [];
  const glossPhrase = (text, phraseId) =>
    new Promise((resolve) => calls.push({ text, phraseId, resolve }));
  const respond = async (index, response) => {
    await act(async () => {
      calls[index].resolve(response);
    });
  };
  const ok = (words) => ({
    ok: true,
    result: { gloss: words.join(" ").toUpperCase(), words, dropped: [] },
  });
  return { calls, glossPhrase, respond, ok };
}

function statusOf(result, text) {
  return result.current.entries.find((e) => e.text === text)?.status;
}

describe("useLivePhraseQueue", () => {
  it("translates phrases one at a time, in speaking order", async () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    act(() => {
      result.current.enqueueFinal({ text: "first phrase", at: 1 });
      result.current.enqueueFinal({ text: "second phrase", at: 2 });
    });
    expect(gloss.calls.map((c) => c.text)).toEqual(["first phrase"]); // one in flight
    expect(statusOf(result, "second phrase")).toBe("waiting");

    await gloss.respond(0, gloss.ok(["first"]));
    await waitFor(() => expect(gloss.calls).toHaveLength(2));
    await gloss.respond(1, gloss.ok(["second"]));

    let a;
    let b;
    act(() => {
      a = result.current.takeNext();
      b = result.current.takeNext();
    });
    expect([a.words, b.words]).toEqual([["first"], ["second"]]);
    expect(a.finalAt).toBe(1);
  });

  it("splits a long final into several phrases", async () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    act(() =>
      result.current.enqueueFinal({
        text: "my friend is not coming to the party tonight because she is sick",
        at: 1,
      })
    );
    expect(result.current.entries.map((e) => e.text)).toEqual([
      "my friend is not coming to the",
      "party tonight because she is sick",
    ]);
  });

  it("ignores empty results", () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    act(() => result.current.enqueueFinal({ text: "   ", at: 1 }));
    expect(result.current.entries).toEqual([]);
    expect(gloss.calls).toHaveLength(0);
  });

  it(`keeps at most ${MAX_BACKLOG} phrases waiting, dropping the oldest with a notice`, async () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    const texts = ["one", "two", "three", "four", "five"];
    act(() => texts.forEach((text, i) => result.current.enqueueFinal({ text, at: i })));
    for (let i = 0; i < texts.length; i += 1) {
      await waitFor(() => expect(gloss.calls).toHaveLength(i + 1));
      await gloss.respond(i, gloss.ok([texts[i]]));
    }
    expect(result.current.readyCount).toBe(MAX_BACKLOG);
    expect(result.current.droppedCount).toBe(2);
    expect(statusOf(result, "one")).toBe("dropped");
    expect(statusOf(result, "two")).toBe("dropped");
    let next;
    act(() => {
      next = result.current.takeNext();
    });
    expect(next.words).toEqual(["three"]); // the oldest phrase still waiting
  });

  it("marks failures and empty glosses, and never queues them for signing", async () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    act(() => {
      result.current.enqueueFinal({ text: "server down", at: 1 });
      result.current.enqueueFinal({ text: "just pronouns", at: 2 });
    });
    await gloss.respond(0, { ok: false, code: "unreachable", message: "" });
    await waitFor(() => expect(gloss.calls).toHaveLength(2));
    await gloss.respond(1, gloss.ok([]));

    expect(result.current.entries.find((e) => e.text === "server down")).toMatchObject({
      status: "not_translated",
      reason: "unreachable",
    });
    expect(statusOf(result, "just pronouns")).toBe("nothing_to_sign");
    expect(result.current.readyCount).toBe(0);
  });

  it("tracks signing and done, and counts everything still waiting", async () => {
    const gloss = controllableGloss();
    const { result } = renderHook(() => useLivePhraseQueue({ glossPhrase: gloss.glossPhrase }));
    act(() => result.current.enqueueFinal({ text: "hello", at: 1 }));
    expect(result.current.waitingCount).toBe(1); // translating
    await gloss.respond(0, gloss.ok(["hello"]));
    expect(result.current.waitingCount).toBe(1); // queued
    let next;
    act(() => {
      next = result.current.takeNext();
    });
    expect(statusOf(result, "hello")).toBe("signing");
    expect(result.current.waitingCount).toBe(0);
    act(() => result.current.markDone(next.phraseId));
    expect(statusOf(result, "hello")).toBe("done");
  });
});
