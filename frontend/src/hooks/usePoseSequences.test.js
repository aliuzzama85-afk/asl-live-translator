import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

import { usePoseSequences } from "./usePoseSequences.js";

const MANIFEST = {
  about: { low_confidence: false, quality_notes: null },
  phone: { low_confidence: true, quality_notes: "some real quality note" },
};

const SEQUENCES = {
  about: {
    gloss: "ABOUT",
    fps: 30,
    landmark_names: ["p"],
    frames: [[[0, 0, 0]]],
    source: "test:about",
  },
  phone: {
    gloss: "PHONE",
    fps: 30,
    landmark_names: ["p"],
    frames: [[[1, 1, 1]]],
    source: "test:phone",
  },
};

function jsonResponse(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

let fetchMock;

beforeEach(() => {
  fetchMock = vi.fn(async (url) => {
    const path = String(url);
    if (path.endsWith("/poses/manifest.json")) {
      return jsonResponse(MANIFEST);
    }
    const match = path.match(/\/poses\/([^/]+)\.json$/);
    const word = match?.[1];
    if (word && SEQUENCES[word]) {
      return jsonResponse(SEQUENCES[word]);
    }
    return jsonResponse({ error: "not_found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("usePoseSequences", () => {
  it("returns idle with no results for an empty word list", () => {
    const { result } = renderHook(() => usePoseSequences([]));
    expect(result.current).toEqual({ status: "idle", results: [], fetchMs: null, message: null });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches the manifest exactly once regardless of sequence length, and resolves one result per word", async () => {
    const { result } = renderHook(() => usePoseSequences(["about", "phone", "about"]));

    await waitFor(() => {
      expect(result.current.status).toBe("ready");
    });

    const manifestCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/poses/manifest.json")
    );
    expect(manifestCalls).toHaveLength(1);

    expect(result.current.results).toHaveLength(3);
    expect(result.current.results[0]).toMatchObject({ status: "ok", word: "about" });
    expect(result.current.results[1]).toMatchObject({ status: "low_confidence", word: "phone" });
    expect(result.current.results[2]).toMatchObject({ status: "ok", word: "about" });
    expect(typeof result.current.fetchMs).toBe("number");
  });

  it("a not_found word's entry doesn't block the other words from resolving", async () => {
    const { result } = renderHook(() => usePoseSequences(["about", "xyzzynotasign", "phone"]));

    await waitFor(() => {
      expect(result.current.status).toBe("ready");
    });

    expect(result.current.results).toHaveLength(3);
    expect(result.current.results[0].status).toBe("ok");
    expect(result.current.results[1]).toEqual({ status: "not_found", word: "xyzzynotasign" });
    expect(result.current.results[2].status).toBe("low_confidence");
  });

  it("a manifest-fetch 404 produces a sequence-level error, not per-word results", async () => {
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith("/poses/manifest.json")) {
        return jsonResponse({ error: "not found" }, 404);
      }
      return jsonResponse(SEQUENCES.about);
    });

    const { result } = renderHook(() => usePoseSequences(["about"]));

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });

    expect(result.current.results).toEqual([]);
    expect(result.current.fetchMs).toBeNull();
    expect(result.current.message).toMatch(/not.*built/i);
  });

  it("a manifest-fetch non-404 failure also produces a sequence-level error", async () => {
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith("/poses/manifest.json")) {
        return jsonResponse({ error: "server error" }, 500);
      }
      return jsonResponse(SEQUENCES.about);
    });

    const { result } = renderHook(() => usePoseSequences(["about"]));

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });

    expect(result.current.results).toEqual([]);
    expect(result.current.message).toMatch(/HTTP 500/);
  });

  it("resets to idle when the word list becomes empty again", async () => {
    const { result, rerender } = renderHook(({ words }) => usePoseSequences(words), {
      initialProps: { words: ["about"] },
    });

    await waitFor(() => {
      expect(result.current.status).toBe("ready");
    });

    rerender({ words: [] });

    await waitFor(() => {
      expect(result.current.status).toBe("idle");
    });
    expect(result.current.results).toEqual([]);
  });
});
