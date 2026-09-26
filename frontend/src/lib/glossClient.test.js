import { describe, expect, it, vi } from "vitest";

import { checkHealth, glossPhrase, waitForGlossService } from "./glossClient.js";

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new SyntaxError("not JSON");
      return body;
    },
  };
}

describe("checkHealth", () => {
  it.each([
    ["ready", { status: "ready", checkpoint: "checkpoints_v2" }],
    ["loading", { status: "loading" }],
    ["failed", { status: "failed", message: "OSError" }],
  ])("reports %s", async (state, body) => {
    const result = await checkHealth({ fetchImpl: async () => response(body) });
    expect(result.state).toBe(state);
  });

  it("treats a network error or the dev proxy's non-JSON error page as unreachable", async () => {
    const down = await checkHealth({
      fetchImpl: async () => {
        throw new TypeError("Failed to fetch");
      },
    });
    const proxyError = await checkHealth({ fetchImpl: async () => response(undefined, 500) });
    expect([down.state, proxyError.state]).toEqual(["unreachable", "unreachable"]);
  });
});

describe("glossPhrase", () => {
  it("posts the phrase and returns the result", async () => {
    const fetchImpl = vi.fn(async () =>
      response({ phrase_id: 3, text: "hi", gloss: "HALF", words: ["half"], dropped: [] })
    );
    const result = await glossPhrase("hi", 3, { fetchImpl });
    expect(result).toMatchObject({ ok: true, result: { words: ["half"] } });
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe("/api/gloss");
    expect(JSON.parse(init.body)).toEqual({ text: "hi", phrase_id: 3 });
  });

  it("passes through the server's own error code", async () => {
    const result = await glossPhrase("hi", 1, {
      fetchImpl: async () => response({ error: "rate_limited", message: "slow down" }, 429),
    });
    expect(result).toEqual({ ok: false, code: "rate_limited", message: "slow down" });
  });

  it("maps a timeout and a missing server", async () => {
    const hang = (url, { signal }) =>
      new Promise((_, reject) =>
        signal.addEventListener("abort", () =>
          reject(Object.assign(new Error("aborted"), { name: "AbortError" }))
        )
      );
    expect((await glossPhrase("hi", 1, { fetchImpl: hang, timeoutMs: 10 })).code).toBe("timeout");
    const proxyError = await glossPhrase("hi", 1, {
      fetchImpl: async () => response(undefined, 502),
    });
    expect(proxyError.code).toBe("unreachable");
  });
});

describe("waitForGlossService", () => {
  const noSleep = async () => {};

  it("keeps waiting through 'no answer' and 'loading', then reports ready", async () => {
    const states = ["unreachable", "loading", "loading", "ready"];
    const onProgress = vi.fn();
    const result = await waitForGlossService({
      check: async () => ({ state: states.shift() }),
      onProgress,
      sleep: noSleep,
    });
    expect(result.state).toBe("ready");
    expect(onProgress.mock.calls.map((c) => c[0])).toEqual(["connecting", "loading", "loading"]);
  });

  it("gives up on 'no answer' after the grace period", async () => {
    let t = 0;
    const result = await waitForGlossService({
      check: async () => ({ state: "unreachable" }),
      sleep: async (ms) => {
        t += ms;
      },
      now: () => t,
      graceMs: 5000,
      pollMs: 2000,
    });
    expect(result.state).toBe("unreachable");
    expect(t).toBeGreaterThanOrEqual(4000);
  });

  it("stops when cancelled", async () => {
    const controller = new AbortController();
    const result = await waitForGlossService({
      check: async () => {
        controller.abort();
        return { state: "loading" };
      },
      signal: controller.signal,
      sleep: noSleep,
    });
    expect(result.state).toBe("cancelled");
  });
});
