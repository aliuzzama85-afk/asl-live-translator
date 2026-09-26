/**
 * Client for the local gloss server (`pipeline/gloss_server.py`), reached
 * through the Vite dev proxy at `/api/*`. See `pipeline/STAGE1_2_PLAN.md`
 * Sections 3 and 6 for the contract and the error behavior.
 */

export const GLOSS_TIMEOUT_MS = 5000;
export const HEALTH_TIMEOUT_MS = 5000;

/** How long "no answer at all" is retried before reporting the service as not
 * running. On a cold start, PyTorch's import keeps the server from answering
 * for ~5s (measured), and `npm run dev:live` starts both processes at once, so
 * the first seconds of "no answer" are expected, not a failure. */
export const CONNECT_GRACE_MS = 15000;
export const HEALTH_POLL_MS = 2000;

/**
 * @typedef {"ready"|"loading"|"failed"|"unreachable"} ServiceState
 */

async function requestJson(url, init, timeoutMs, fetchImpl) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, { ...init, signal: controller.signal });
    let body = null;
    try {
      body = await res.json();
    } catch {
      body = null; // e.g. the Vite proxy's own error page when the server is down
    }
    return { res, body };
  } catch (err) {
    return { error: err?.name === "AbortError" ? "timeout" : "unreachable" };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Checks the gloss server's health once.
 *
 * @param {{fetchImpl?: typeof fetch, timeoutMs?: number}} [options]
 * @returns {Promise<{state: ServiceState, message?: string}>} `unreachable`
 *   covers a network error, a timeout, and any non-JSON answer (the dev
 *   proxy's error response when nothing is listening).
 */
export async function checkHealth({ fetchImpl = fetch, timeoutMs = HEALTH_TIMEOUT_MS } = {}) {
  const { res, body, error } = await requestJson("/api/health", {}, timeoutMs, fetchImpl);
  if (error || !res.ok || !body || typeof body.status !== "string") {
    return { state: "unreachable" };
  }
  if (body.status === "ready" || body.status === "loading" || body.status === "failed") {
    return { state: body.status, message: body.message };
  }
  return { state: "unreachable" };
}

/**
 * Waits for the gloss server to be ready, reporting progress.
 *
 * `loading` is waited on for as long as it lasts. "No answer" is retried
 * for `graceMs`, then reported as `unreachable`. `failed` is final.
 *
 * @param {{
 *   check?: () => Promise<{state: ServiceState, message?: string}>,
 *   onProgress?: (state: "connecting"|"loading") => void,
 *   signal?: AbortSignal,
 *   graceMs?: number,
 *   pollMs?: number,
 *   sleep?: (ms: number) => Promise<void>,
 *   now?: () => number,
 * }} [options]
 * @returns {Promise<{state: ServiceState|"cancelled", message?: string}>}
 */
export async function waitForGlossService({
  check = checkHealth,
  onProgress = () => {},
  signal,
  graceMs = CONNECT_GRACE_MS,
  pollMs = HEALTH_POLL_MS,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  now = () => Date.now(),
} = {}) {
  const started = now();
  for (;;) {
    if (signal?.aborted) return { state: "cancelled" };
    const result = await check();
    if (signal?.aborted) return { state: "cancelled" };
    if (result.state === "ready" || result.state === "failed") return result;
    if (result.state === "unreachable" && now() - started >= graceMs) return result;
    onProgress(result.state === "loading" ? "loading" : "connecting");
    await sleep(pollMs);
  }
}

/**
 * Translates one phrase.
 *
 * @param {string} text - A sanitized phrase.
 * @param {number} phraseId
 * @param {{fetchImpl?: typeof fetch, timeoutMs?: number}} [options]
 * @returns {Promise<{ok: true, result: {phrase_id: number, text: string, gloss: string,
 *   words: string[], dropped: string[], inference_ms: number}}
 *   | {ok: false, code: string, message: string}>} On failure, `code` is the
 *   server's own error code (`rate_limited`, `not_ready`, `input_too_long`,
 *   ...) when it answered, else `unreachable` or `timeout`.
 */
export async function glossPhrase(
  text,
  phraseId,
  { fetchImpl = fetch, timeoutMs = GLOSS_TIMEOUT_MS } = {}
) {
  const { res, body, error } = await requestJson(
    "/api/gloss",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, phrase_id: phraseId }),
    },
    timeoutMs,
    fetchImpl
  );
  if (error) {
    return { ok: false, code: error, message: `Translation service ${error}.` };
  }
  if (res.ok && body && Array.isArray(body.words)) {
    return { ok: true, result: body };
  }
  if (body && typeof body.error === "string") {
    return { ok: false, code: body.error, message: body.message ?? "" };
  }
  return { ok: false, code: "unreachable", message: `HTTP ${res.status}` };
}
