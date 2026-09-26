# Stage 1–2 Plan (v1): Live Speech Input

Scope per `CLAUDE.md`'s build order, **step 4: "Live speech input (Stage 1 + 2)
wired to the working core."** A person speaks into their microphone; the
speech is transcribed live, cut into phrases, translated to ASL gloss by the
existing `gloss_model`, and played through the **already-built** multi-word
pipeline (`usePoseSequences` → `stitchTimelines` → `SkeletonCanvas`),
fingerspelling included. This document is a plan only: no code is written or
modified yet.

**Decision already made (by the user, not revisited here)**: v1 ASR is the
browser's built-in **Web Speech API** (`SpeechRecognition` /
`webkitSpeechRecognition`), with no external API, signup, or cost. The design
keeps ASR behind one small interface, so a later move to a streaming provider
(Deepgram, AssemblyAI) replaces one module and leaves chunking, gloss, and
rendering alone.

Baseline checked directly before writing this (not assumed): `CLAUDE.md`
(architecture, security, build order), `PROJECT_STATUS.md` (Sections 1, 5, 6,
9, 12–14), `frontend/MULTIWORD_PLAN.md`, `gloss_model/inference.py`,
`gloss_model/config.py`, `pose_library/wlasl_metadata.py`
(`extract_vocab_stems`), `frontend/src/{App.jsx,components/CaptionBand.jsx,
hooks/usePoseSequences.js}`, `frontend/vite.config.js`, `requirements.txt`,
`.env.example`. The model was run for real on this machine (Section 3), and
its output was checked against the real pose library (Section 4).

---

## 0. What CLAUDE.md originally proposed, and what changes

`CLAUDE.md`'s architecture section names:

1. **ASR**: "streaming, via Deepgram or AssemblyAI. Lives in `pipeline/asr.py`."
2. **VAD chunking**: "Silero VAD detects pauses, emits complete phrases. Lives
   in `pipeline/vad.py`."
3. An **in-memory queue** "so playback never blocks on translation — the
   avatar always stays a phrase or two 'behind' live speech."

The Web Speech API decision changes where (1) and (2) live, because that API
**only exists in the browser**: it takes the microphone itself and hands back
text. No audio ever reaches Python. So in v1:

| Piece | Original plan | v1 |
|---|---|---|
| ASR | `pipeline/asr.py` (Python, Deepgram/AssemblyAI) | `frontend/src/lib/asr/` (JS, Web Speech API), behind an interface a Deepgram/AssemblyAI adapter can implement later |
| Phrase chunking | `pipeline/vad.py` (Silero VAD) | The recognizer's own end-of-utterance ("final result") segmentation, plus a length guard. **No separate VAD** (Section 2) |
| Gloss translation | `gloss_model` (Python) | Unchanged, served live by a small local HTTP wrapper in `pipeline/` (Section 3) |
| Queue | in-memory | In-memory, in the frontend, one phrase per playback (Section 4) |
| Playback | built | Unchanged |

`pipeline/asr.py` and `pipeline/vad.py` are therefore **not created**. Once
this plan is approved, `CLAUDE.md`'s architecture section needs updating to
say so. That's flagged here rather than changed silently, and not done in
this planning pass.

---

## 1. Architecture overview

```
 Browser (frontend/, JS)                                       Local Python (pipeline/)
 ─────────────────────────────────────────────────────────     ─────────────────────────
 mic ─► Web Speech API ─► ASR adapter ─► phrase chunker ─┐
        (browser-owned)    lib/asr/       lib/phrases.js │  POST /api/gloss {text}
                                                         ├───────────────────────────► gloss_server.py
                                                         │   (Vite dev proxy,           │ model loaded ONCE
                                                         │    same origin)              │ translate(text)
                                                         │                              │ gloss → lookup words
                                           words: [...] ◄┘◄──────────────────────────── ┘ {gloss, words, ...}
                                                │
                              live phrase queue (hooks/useLivePhraseQueue.js)
                                                │  one phrase at a time, when the avatar is idle
                                                ▼
                         EXISTING: setSubmittedWords(words) → usePoseSequences → planPlayback
                                   (fingerspelling) → stitchTimelines → SkeletonCanvas
```

### How the browser and Python talk: one local HTTP endpoint
`gloss_model` is Python and the Web Speech API is browser JS, so they need a
boundary. Options considered:

- **Run the model in the browser** (export T5 to ONNX and run it with
  `onnxruntime-web` or transformers.js): no server at all, but it means a new
  export pipeline, a ~240MB model download into the page, and re-validating
  the model's output after conversion (a new source of bugs of exactly the
  kind Section 9 of `PROJECT_STATUS.md` spent a retrain fixing). Too much for
  v1.
- **WebSocket server**: only useful for streaming audio or partial results.
  v1 sends one short, finished text phrase and gets one answer back; that's
  request/response, not a stream.
- **One HTTP endpoint (chosen)**: `POST /api/gloss` on a small local server
  that loads the model once and calls the existing `translate()`. That's the
  simplest thing that works.

**`gloss_model/inference.py` is already callable this way, unchanged.** Checked
directly: it exposes `load_model(checkpoint_dir)` (load once, eval mode) and
`translate(text, model, tokenizer)`. It already enforces
`MAX_INPUT_CHARS = 500` and raises `EmptyInputError`/`InputTooLongError`; its
own docstring says it is "the eventual callee for that live endpoint." The
server is a thin wrapper around it with no changes to `inference.py`. One
gotcha: `config.CHECKPOINT_DIR` still defaults to **v1**
(`gloss_model/checkpoints/`), so the server must pass
`gloss_model/checkpoints_v2/` explicitly (configurable via `.env`,
`GLOSS_CHECKPOINT_DIR`).

### The server: Python standard library, no new dependency
`pipeline/gloss_server.py`, run as `python -m pipeline.gloss_server`, built on
`http.server` from the standard library (no Flask, FastAPI, or uvicorn):

- **Why not FastAPI/Flask**: the endpoint is one route, one JSON body, and
  one local user. FastAPI would add four packages (fastapi, starlette,
  pydantic, uvicorn) to audit and pin for validation this route can do in a
  few lines. `CLAUDE.md` requires `pip-audit` before any new dependency, and
  the cheapest dependency to audit is none. **Revisit** if a later provider
  swap needs WebSockets or streaming, or if more routes appear.
- **Single-threaded on purpose** (`HTTPServer`, not `ThreadingHTTPServer`):
  requests are served one at a time. With one speaker producing a phrase
  every few seconds, and inference at 80–480ms (Section 3), there's no
  contention worth handling. It also means the PyTorch model is never
  called from two threads at once, so no locking needed.
- **Binds `127.0.0.1` only**, never `0.0.0.0`. It's a local dev service, in
  line with the project's local/dev-only decision (`PROJECT_STATUS.md`
  Section 5 item 6).
- **No CORS needed**: Vite's dev server proxies `/api/*` to the Python server
  (`server.proxy` in `vite.config.js`), so the browser sees one origin. The
  server still rejects a request whose `Origin` header is present and isn't
  in `ALLOWED_ORIGINS`, as defense in depth. Note that `.env.example` has
  `ALLOWED_ORIGINS=http://localhost:3000`, which is stale (Vite runs on
  5173) and gets fixed in the build.

### Running it: two processes, one command
Live speech is the first stage that needs **two processes at once**: the
Vite dev server (the app) and `python -m pipeline.gloss_server` (the model).
Every earlier stage ran from `npm run dev` alone. Options considered:

- **`concurrently`** (npm): the usual answer, but a new dependency, which
  `CLAUDE.md` says must be `npm audit`ed and pinned, for a job the standard
  library already does.
- **Have Vite start Python itself** (from the `configureServer` hook the
  pose middleware already uses): one command, but it would make plain
  `npm run dev` load the model (~4.6s, several hundred MB of memory) every
  time, even when nobody uses live mode. Typed-input development shouldn't
  pay for live mode.
- **A small Node script using only the standard library (chosen)**:
  `frontend/scripts/dev-live.mjs`, run as **`npm run dev:live`**. It spawns
  the gloss server with the project's venv Python (`.venv/Scripts/python.exe`
  on Windows, `.venv/bin/python` elsewhere, failing with a clear message if
  neither exists), then spawns Vite. It prefixes each process's output
  (`[gloss]`, `[vite]`), and if either one exits it stops the other, so
  nothing is left running in the background. No new dependency, about 60
  lines.

`npm run dev` is unchanged and still starts the app alone. Live mode then
shows its "translation service not running" message (Section 6), with the
exact command to start it. The two ways to run are documented in a new
"Running it" section of the repo `README.md`, which currently has no setup
instructions at all, and the script is listed in "Planned files" below.

---

## 2. VAD / phrase chunking: no separate VAD in v1

### What the browser already gives us
With `continuous = true` and `interimResults = true`, the recognizer emits a
stream of results. Each result is **interim** (still changing) until the
recognizer decides the utterance is over, typically after a short pause, and
marks it **final** (`isFinal`). That final result *is* voice-activity-based
phrase segmentation: the browser's recognizer has already detected speech
start, speech end, and the pause between phrases, on the same audio it
transcribed.

### Why a separate Silero VAD would be redundant (and harmful) here
- **We never get the audio.** The Web Speech API reads the mic itself and
  doesn't expose the sound. Running Silero would mean opening the mic a
  second time (`getUserMedia`), running Silero in the browser (ONNX +
  `onnxruntime-web`) or streaming audio to Python, all just to find pause
  boundaries the recognizer already found.
- **Two segmenters would disagree.** Silero's "phrase ended" and the
  recognizer's "result final" come from separate systems with different
  timing. We'd have to reconcile a boundary in one with text from the other,
  which creates problems instead of solving one.
- **Silero was chosen for a raw-audio pipeline.** It's the right tool when
  you own an audio stream and a streaming ASR that doesn't segment for you.
  Web Speech API isn't that. (Deepgram and AssemblyAI both ship their own
  endpointing too, so whether Silero is ever needed is an open question for
  the swap, not something to build now.)

**Decision: in v1, one final result = one phrase.** Two guards on top:

1. **A length guard.** Measured in Section 3: the model's output degrades on
   long inputs. A 13-word sentence stopped early at `…BECAUSE X-`, the same
   early-stop behavior Section 9 of `PROJECT_STATUS.md` documents. A fast
   talker who doesn't pause produces long final results. So a final longer
   than **`MAX_PHRASE_WORDS = 12`** words is split into consecutive chunks of
   at most 12, at word boundaries, before translation. That's a blunt
   heuristic tied to the measured degradation, not linguistics
   (punctuation-aware chunking is out of scope, Section 7). The number is
   tunable after real use.
2. **A flush on stop.** If the user turns the mic off mid-utterance, the last
   interim text is treated as final and sent, instead of being lost.

**Interim results are shown, never translated.** They change word by word,
and translating them would waste inference and make the avatar start signs
it would then have to retract.

### Recognizer restarts
Chrome ends a continuous session on its own after a period of silence or a
time limit (the `end` event fires even though nobody pressed stop). While
the mic toggle is on, the ASR adapter restarts it immediately on `end` and
counts restarts. More than 3 restarts in 10 seconds without any result means
something is wrong, and it stops with an error instead of looping forever
(Section 6).

---

## 3. gloss_model as a live service

### Measured, not assumed (this machine: CPU only, 10 threads, `checkpoints_v2`)

| | Time |
|---|---|
| Python + transformers import (once, at server start) | 4.3 s |
| `load_model(checkpoints_v2)` (once, at server start) | 0.3 s |
| `translate("hello")` → `HALF` | 78 ms |
| `translate("where is the bathroom")` → `WHERE BE BATHROOM` | 137 ms |
| 8 words → `X-I NEED TO SEE DOCTOR TOMORROW MORNING` | 276 ms |
| 7 words → `CAN X-YOU HELP X-I FIND X-MY PHONE` | 405 ms |
| 13 words → `X-MY FRIEND BE DESC-NOT COME TO PARTY DESC-TONIGHT BECAUSE X-` (stopped early) | 480 ms |

(Warm, median of 3 runs each.) So:

- **The model must be loaded once and kept in memory.** Loading per request
  would add ~4.6s to every phrase. The server loads it at startup and
  reports "ready" only after a warm-up `translate()` has run, so the first
  real phrase doesn't pay a first-call cost either.
- **Per-phrase inference is 80–480ms on CPU**, well within budget. No GPU,
  batching, caching, or model changes are needed for v1.

### Latency budget: from "speaker stops" to "avatar starts"

| Step | Expected |
|---|---|
| Recognizer marks the result final after the pause | ~0.5–1 s (browser-dependent, not controllable) |
| `POST /api/gloss` round trip (local), including inference | ~0.1–0.5 s |
| Pose fetch (`usePoseSequences`, measured in Stage 5 at ~20–80 ms) plus stitching | < 0.1 s |
| **Total, avatar idle** | **~0.7–1.6 s** |
| Plus, if the avatar is still signing the previous phrase | wait until it finishes |

That fits the `CLAUDE.md` framing, "a phrase or two behind, like a human
interpreter." The status strip reports the measured **speech-to-sign**
latency for each phrase (Section 5), not a simulated number.

### The endpoint contract
`POST /api/gloss`, request:
```json
{ "text": "where is the bathroom", "phrase_id": 7 }
```
`200` response:
```json
{
  "phrase_id": 7,
  "text": "where is the bathroom",
  "gloss": "WHERE BE BATHROOM",
  "words": ["where", "bathroom"],
  "dropped": ["BE"],
  "inference_ms": 137
}
```
`GET /api/health` → `{"status": "ready", "checkpoint": "checkpoints_v2"}` (or
`"loading"` during startup). The frontend calls it when the mic is switched
on, so an unreachable backend is caught before anyone speaks (Section 6).

Errors are JSON `{ "error": "<code>", "message": "..." }` with codes:
`empty_input` (400), `input_too_long` (400, from `InputTooLongError`),
`bad_request` (400: malformed JSON, wrong types, body over 4KB),
`rate_limited` (429), `not_ready` (503, model still loading),
`internal` (500; logged server-side, no stack trace sent to the client).

### Security, per CLAUDE.md's non-negotiable rules
- **Sanitize and length-limit before the model call**: strip control
  characters, collapse whitespace, cap the body at 4KB. `translate()` itself
  still enforces `MAX_INPUT_CHARS = 500`, the single source of truth. The
  `.env.example` value `MAX_INPUT_CHARS=1000` conflicts with it and gets
  removed in the build rather than creating a second, looser limit.
- **Rate-limit the endpoint**: a per-client token bucket, sized from the
  existing `.env` `RATE_LIMIT_PER_MINUTE=60`. That's far above what one
  speaker produces (a phrase every few seconds), low enough to stop a runaway
  loop.
- **No credentials** are involved (Web Speech API needs no key), so nothing
  new goes in `.env` except `GLOSS_CHECKPOINT_DIR` and `GLOSS_SERVER_PORT`.
  `gitleaks` keeps running as the pre-commit hook.
- **Privacy, stated plainly**: the Web Speech API is not on-device in
  general. Chrome sends the audio to Google's speech service, and Edge to
  Microsoft's. For an assistive tool that's a real disclosure, not fine
  print: the first time the mic is switched on, the UI says so in one line,
  and the user has to acknowledge it (Section 5). The gloss server itself
  only ever sees text, and only on `localhost`.

---

## 4. Data flow contract

The whole point is a thin layer that produces `words: string[]` and hands it
to code that already exists.

### Stage by stage
1. **ASR adapter → chunker.** The adapter emits events:
   `{type: "partial", text}`, `{type: "final", text, at}` (`at` =
   `performance.now()` when the result went final), `{type: "state", state}`
   (`idle | starting | listening | stopped`), `{type: "error", code,
   message}`.
2. **Chunker → gloss client.** Each final result is trimmed, sanitized, and
   split by the 12-word guard into one or more **phrases**
   `{phraseId, text, finalAt}`. Empty results are dropped (Section 6).
3. **Gloss client → queue.** `POST /api/gloss` per phrase, in order. The
   response becomes a **glossed phrase** `{phraseId, text, gloss, words,
   dropped, finalAt, glossedAt}`.
4. **Queue → existing pipeline.** When the avatar is idle (nothing loaded,
   or the last sequence ended via `SkeletonCanvas`'s existing `onEnded`), the
   queue takes the next glossed phrase and calls **the same
   `setSubmittedWords(words)`** that typing a sentence calls today via
   `CaptionBand`'s `onSearch` → `App.handleSearch`. Everything after that
   (`usePoseSequences`, fingerspelling's `planPlayback`, `stitchTimelines`,
   `SkeletonCanvas`) is **unchanged**.

The only change to existing playback: while live mode is on, `App` feeds
`submittedWords` from the queue instead of from the search box, and forces
`loop` off (a looping phrase would block the queue forever).

### Gloss tokens → lookup words: reuse the existing rule
The model's output isn't lookup-ready. Checked against real output and the
real 118-word library:
- **`X-` pronoun markers** (`X-I`, `X-YOU`, `X-MY`) show up in most everyday
  sentences, and **the library has no pronoun signs** (checked: none of
  i/me/my/you/your/we/he/she/they). In ASL, pronouns are made by pointing,
  not fingerspelled, so spelling `X-MY` as M-Y would be wrong, not just
  noisy.
- **`DESC-` descriptors** (`DESC-NOT`, `DESC-TONIGHT`) wrap real words that
  *are* in the library (`not`, `tonight`).
- **Punctuation tokens** (`.`, `?`) aren't signs.

The project already has this exact rule: `pose_library/wlasl_metadata.py`'s
`extract_vocab_stems` drops `X-` tokens, strips `DESC-`, drops punctuation,
and lowercases. It's used to build the pose vocabulary, so applying it at
playback means live gloss maps onto that same vocabulary. But it's written
inline inside a CSV reader. Per `CLAUDE.md`'s reuse-over-duplication rule,
the per-token rule gets **extracted into one shared function**,
`normalize_gloss_token(token) -> str | None`, in a new
`pose_library/gloss_tokens.py`. `extract_vocab_stems` then calls it (its
tests still pass), and the gloss server calls it. That is also why
normalization runs **server-side in Python**: a JavaScript copy would be a
second implementation of the same rule. The response's `dropped` list shows
exactly what was removed, so nothing disappears silently.

**One addition, flagged as a judgment call**: the corpus's rule-generated
gloss keeps English function words as tokens (`BE`, `TO`), none of which is
in the library. Left alone, every `BE` would be fingerspelled B-E, which
isn't how ASL works (`pose_library/PLAN.md` Section 1 already lists
"be, do, at, too, would, well, and" as words ASL grammar drops). So a small,
documented **stop-list** of those tokens goes into `gloss_tokens.py` and
they're dropped like `X-` markers. It starts with exactly that PLAN.md list
plus `to` (seen in real output above), and only changes on evidence.
**This needs a check by someone who knows ASL**, the same open item as the
fingerspelling review.

Worked example, all real: "can you help me find my phone" → gloss `CAN X-YOU
HELP X-I FIND X-MY PHONE` → words `["can", "help", "find", "phone"]` (all
four in the library), dropped `["X-YOU", "X-I", "X-MY"]`.

### The live phrase queue (`hooks/useLivePhraseQueue.js`)
An in-memory FIFO of glossed phrases, as `CLAUDE.md` describes, with one rule
beyond first-in-first-out: **a maximum backlog of 3 phrases**. If speech
outruns the avatar by more than that, the oldest waiting phrase is dropped
and the UI says so ("SKIPPED 1 PHRASE — FELL BEHIND"). A human interpreter
who falls behind drops content rather than falling further behind for good,
and a queue that grows without limit would put the avatar minutes behind the
speaker. Translation requests go out **in order, one at a time** (the server
is single-threaded anyway), so glossed phrases arrive in speaking order
without reordering logic.

---

## 5. UI

Per `CLAUDE.md`: minimal chrome, a live-captioning feel, strong contrast,
visible state at all times. `CaptionBand.jsx` was checked. It already has a
controls row (PLAY/LOOP buttons with `aria-pressed`), the word-progress row,
status banners with `role="status"`, and the search form. Live mode extends
it rather than adding a new band:

- **Mic toggle**: a third control button, `MIC`, next to PLAY/LOOP, the same
  `controlButton` style and `aria-pressed`, labelled `LISTENING` while on.
  Keyboard-operable like the others. When the Web Speech API isn't available
  it's disabled, with the reason as visible text (Section 6), never just
  greyed out.
- **Live transcript line**: a new line above the word row, like a real-time
  caption. Final text is `--color-text-primary`; interim text (the words
  still being recognized) follows in `--color-text-muted`, italic. Only
  **final** text goes in an `aria-live="polite"` region, since announcing
  every interim change would flood a screen reader.
- **Gloss/latency state**: the word row keeps showing the phrase currently
  being signed, exactly as today. Two small additions:
  - **Status strip** gains `LISTENING` (ready dot) and `TRANSLATING…` (amber
    pulsing dot, the existing `loading` style) while a phrase is at the gloss
    server, plus a queue count when phrases are waiting
    (`● SIGNING — 2 PHRASES QUEUED`).
  - **Latency readout**: the status strip's `FETCH` readout becomes
    `SPEECH→SIGN` in live mode: the measured time from the result going final
    to the first frame of that phrase. That's a real number, per Stage 5's
    rule of never showing simulated latency.
- **Privacy notice**: the first time the mic is switched on, a one-line
  banner: `SPEECH IS PROCESSED BY YOUR BROWSER'S SPEECH SERVICE (E.G.
  GOOGLE IN CHROME), NOT ON THIS DEVICE` with an `OK` button. Recognition
  starts only after OK. The acknowledgement is remembered in `localStorage`
  (wrapped in try/catch, and asked again if storage is unavailable).
- **LOOP is visibly disabled in live mode, never silently ignored.** Live
  mode forces loop off (Section 4), because a looping phrase would block the
  queue forever. The LOOP button shows that: it gets the `disabled`
  attribute and the existing disabled `controlButton` style (dimmed,
  `not-allowed` cursor), the same treatment PLAY/LOOP already get when
  nothing is loaded. A short visible line under the controls,
  `LOOP IS OFF DURING LIVE SPEECH`, is linked to the button with
  `aria-describedby`. That's visible text, not just a hover tooltip, so it
  works for keyboard and touch users and screen readers, and it matches how
  the MIC button states its own disabled reason. If loop was on when the
  mic was switched on, it's turned off and stays off when the mic goes off.
  The user turns it back on, rather than the app flipping it back unasked.
- **Typed input stays**: the search box still works. While the mic is on it's
  disabled, with the placeholder "Mic is on — turn it off to type", so the two
  inputs never compete for the same queue.
- **Never frozen**: at every moment exactly one of these is showing:
  listening (and what's being heard), translating, signing, an error with
  its reason, or the idle prompt.

---

## 6. Error handling

Consistent with the project rule of never leaving a silently frozen state:
every failure shows a specific message, the mic state stays truthful, and
nothing that already works (typed input, playback of already-queued phrases)
breaks.

| Situation | Detected by | Behavior |
|---|---|---|
| **Browser has no Web Speech API** (Firefox) | `window.SpeechRecognition ?? window.webkitSpeechRecognition` is undefined at load | MIC button disabled, with the text `LIVE SPEECH NEEDS CHROME, EDGE, OR SAFARI`. Typed input works as today. |
| **Mic permission denied** | recognizer error `not-allowed` / `service-not-allowed` | Mic toggles back off. Banner: `MICROPHONE BLOCKED — ALLOW IT IN YOUR BROWSER'S SITE SETTINGS, THEN TRY AGAIN`. No retry loop. |
| **No microphone / mic in use** | error `audio-capture` | Mic off. Banner: `NO MICROPHONE FOUND (OR IT'S IN USE BY ANOTHER APP)`. |
| **Speech service unreachable** (it needs internet) | error `network` | Mic off. Banner: `SPEECH SERVICE UNREACHABLE — CHECK YOUR INTERNET CONNECTION`. |
| **No speech detected** | error `no-speech`, or the silence timeout | Not an error. The recognizer is quietly restarted (Section 2), and the strip keeps saying `LISTENING`. |
| **Recognizer keeps ending with no results** | more than 3 restarts in 10 s with no result | Mic off. Banner: `SPEECH RECOGNITION KEEPS STOPPING — TURN THE MIC OFF AND ON TO RETRY`. |
| **Gloss backend unreachable** | `/api/health` fails when the mic is switched on, or `/api/gloss` fails (network, timeout after 5 s) | On mic-on: the mic doesn't start. Banner: `TRANSLATION SERVICE NOT RUNNING — START IT WITH: python -m pipeline.gloss_server`. Mid-session: the phrase's transcript still shows, marked `NOT TRANSLATED (SERVICE UNREACHABLE)`, and listening continues so a restarted server picks up from the next phrase. Phrases are never retried silently later, out of order. |
| **Backend still loading** | `503 not_ready` | `TRANSLATION SERVICE STARTING…`, re-checked every 2 s. The mic starts when the server is ready. |
| **Rate-limited** | `429` | That phrase shows `NOT TRANSLATED (TOO MANY REQUESTS)`; later phrases continue. |
| **Empty / whitespace transcription** | chunker | Dropped before any request, since there's nothing to sign. Not shown as an error. |
| **Gibberish or mis-heard speech** | can't be detected reliably | Handled honestly, not "fixed": the live transcript line shows exactly what was heard, so a bad recognition is visible as such. Out-of-vocabulary words go through the existing fingerspelling fallback. Words it can't spell are skipped with the existing messages. No confidence filtering in v1 (the API's confidence scores are inconsistent across browsers). |
| **Phrase glosses to nothing playable** (e.g. only pronouns, all dropped) | `words` is empty | The transcript shows, marked `NOTHING TO SIGN`, and the queue moves on. The avatar never sits on a stale sign as if it were the new phrase. |

---

## 7. Explicitly out of scope for v1

- **Multiple speakers / diarization**: one speaker, one mic.
- **Punctuation-aware or grammar-aware chunking**: phrases are whatever the
  recognizer finalizes, split only by the 12-word guard.
- **Non-English speech**: `lang = "en-US"`, fixed. The gloss model is
  English → ASL only.
- **Offline / no-internet operation**: the Web Speech API (in Chrome and
  Edge) needs its network speech service. The UI says so when it's
  unreachable (Section 6), but v1 has no offline mode.
- **A UI for choosing ASR providers**: the adapter interface makes a later
  swap a code change, and choosing a provider would be a config value then,
  not a v1 control.
- **A Deepgram/AssemblyAI adapter itself**: designed for, not built.
- **Showing pronouns** (`X-` markers are dropped; pointing/indexing signs
  don't exist in the library) and **ASL grammar/facial markers**: same
  limits as Stage 3/5.
- **Retracting or revising a phrase once signed**, or live-editing the
  queue.
- **Running beyond `localhost`**: no deployment, TLS, or auth, per the
  local/dev-only decision.

---

## 8. Test bar / sign-off

Per `CLAUDE.md`: every new module in `pipeline/` gets a test in `tests/`, and
this stage isn't done until its tests pass.

### Automated
- **`tests/test_gloss_tokens.py`**: `normalize_gloss_token` (`X-` dropped,
  `DESC-` stripped, punctuation dropped, stop-list dropped, lowercasing) and
  `gloss_to_lookup_words` on the real outputs in Sections 3–4. The existing
  `test_wlasl_metadata.py` tests keep passing unchanged, which proves the
  extraction didn't change `extract_vocab_stems`.
- **`tests/test_gloss_server.py`**, with the model mocked (the project's
  established pattern): request validation and every error code in
  Section 3, sanitization, the 4KB cap, the rate limiter (fake clock), the
  `Origin` check, `/api/health` before and after warm-up, and that the model
  loads **once**. Plus one real-model test that loads `checkpoints_v2` and
  checks `translate` → `words` end to end, **skipped when the checkpoint
  isn't present** (it's gitignored), so a fresh clone's suite still passes.
- **`frontend/src/lib/phrases.test.js`**: sanitizing, the 12-word split,
  empty-final dropping, flush-on-stop.
- **`frontend/src/hooks/useLiveTranscription.test.js`**: driven by a **fake
  recognizer implementing the same adapter interface**. That's also the
  proof that the interface really is the swap boundary. It covers restart on
  `end`, the restart-storm cutoff, and every Section 6 error code mapping to
  its state and message.
- **`frontend/src/hooks/useLivePhraseQueue.test.js`**: FIFO order, the
  3-phrase backlog cap and its "fell behind" notice, and that the queue
  advances only on `onEnded`.
- **`frontend/src/App.test.jsx`**: fake recognizer plus mocked `/api/gloss`,
  covering speech → transcript → `TRANSLATING…` → words through the **real,
  unchanged** `usePoseSequences`/fingerspelling/stitch path → skeleton, a
  phrase with a fingerspelled word, backend-unreachable,
  Web-Speech-unsupported (MIC disabled with its reason), and LOOP shown
  disabled with its visible reason while the mic is on.
- Full suites, `ruff`, `black`, `eslint`, `prettier`, `gitleaks` clean. No
  new dependency is planned; if one becomes necessary, `pip-audit`/`npm
  audit` it first.

### Manual (real browser, per `CLAUDE.md`'s happy-dom lesson)
- **Scripted, without a mic**: a `?asr=fake` dev-only URL flag swaps in a
  scripted recognizer that "speaks" a fixed list of phrases on a timer. That
  lets the whole live path be checked in a real browser, and rerun, with
  no microphone or speech service. Dev builds only, never shipped.
- **Real mic, by a person** (it can't be automated): in Chrome, speak the
  Section 3 phrases and a few of your own, and confirm transcript →
  translating → signing, the speech→sign latency reads roughly 1–2s, the
  privacy notice appears once, and deny-permission / server-stopped behave
  as in Section 6. Repeat in Firefox to confirm the "needs Chrome, Edge, or
  Safari" state.

### Sign-off criteria
- Everything above passing, and the real-mic check done by a person.
- Measured speech→sign latency, avatar idle, **under ~2s** on this machine
  for phrases of 12 words or fewer.
- `CLAUDE.md`'s architecture section updated to match reality
  (`pipeline/asr.py`/`vad.py` not created, and why) — flagged in Section 0.
- Still open afterwards, as it already is: a review by someone who knows ASL
  (the gloss output's quality, the function-word stop-list, the
  fingerspelling), which no automated test covers.

---

## Planned files (for scope; nothing is created yet)

| New | Purpose |
|---|---|
| `pipeline/gloss_server.py` | Local HTTP wrapper around `gloss_model.inference` (stdlib, `127.0.0.1`) |
| `pose_library/gloss_tokens.py` | `normalize_gloss_token` (extracted from `extract_vocab_stems`), stop-list, `gloss_to_lookup_words` |
| `frontend/src/lib/asr/webSpeechRecognizer.js` | Web Speech API adapter behind the ASR interface |
| `frontend/src/lib/phrases.js` | Sanitize, 12-word split, flush-on-stop |
| `frontend/src/lib/glossClient.js` | `/api/health`, `/api/gloss` with timeout and error mapping |
| `frontend/src/hooks/useLiveTranscription.js` | Recognizer lifecycle, restarts, error states |
| `frontend/src/hooks/useLivePhraseQueue.js` | In-memory FIFO, 3-phrase backlog, advances on `onEnded` |
| `frontend/scripts/dev-live.mjs` | **Two-process dev workflow** (Section 1): starts the gloss server and Vite together, prefixes output, stops both together. Node stdlib only, no new dependency |
| Tests for each (Section 8) | |

| Changed (small) | Change |
|---|---|
| `pose_library/wlasl_metadata.py` | `extract_vocab_stems` calls the shared `normalize_gloss_token` |
| `frontend/src/App.jsx` | Live mode feeds `setSubmittedWords` from the queue; forces loop off |
| `frontend/src/components/CaptionBand.jsx` + CSS | MIC button, live transcript line, privacy notice, LOOP visibly disabled with its reason in live mode |
| `frontend/package.json` | Add the `dev:live` script (no new dependencies) |
| `README.md` | New "Running it" section: `npm run dev` vs `npm run dev:live`, and what each needs |
| `frontend/src/components/StatusStrip.jsx` | `LISTENING` / `TRANSLATING…` / queue count, `SPEECH→SIGN` readout |
| `frontend/vite.config.js` | `server.proxy` for `/api` → the gloss server |
| `.env.example` | Add `GLOSS_CHECKPOINT_DIR`, `GLOSS_SERVER_PORT`; fix `ALLOWED_ORIGINS` (5173); remove the conflicting `MAX_INPUT_CHARS` |
| `CLAUDE.md` | Architecture section: ASR/VAD in v1 (after approval) |

Unchanged: `gloss_model/inference.py`, `usePoseSequences`,
`fingerspelling.js`, `stitchTimelines`, `reconstructTimeline`,
`SkeletonCanvas`.

---

## Known gotchas / open questions (carried into implementation)

- **`config.CHECKPOINT_DIR` defaults to v1.** The server must be pointed at
  `checkpoints_v2` explicitly, or it would silently serve the older,
  truncation-affected model.
- **The first request after startup**: the ~4.3s import is paid before the
  server reports `ready` (Section 3), never on a user's phrase.
- **Model quality is now visible in real time.** Measured: "hello" →
  `HALF`, and long sentences stop early. The live transcript line lets
  users see when the mistake is in translation rather than recognition.
  Improving the model is Stage 3's concern, not this stage's.
- **The 12-word guard and the 3-phrase backlog** are informed guesses (the
  first from measured degradation, the second from the "phrase or two
  behind" framing), to be tuned from real use.
- **Web Speech API behavior differs across browsers** (restart timing,
  when results go final, Safari's quirks). v1 targets Chrome first; Edge and
  Safari are expected to work through the same interface but must be
  checked, not assumed.
- **The function-word stop-list** is a judgment call that needs a check by
  someone who knows ASL (Section 4).

---

## Implementation notes (2026-09-26)

Built as planned, with these differences, each recorded in
`PROJECT_STATUS.md` Section 15:

- **Extra files**: `frontend/src/hooks/useLiveMode.js` (privacy notice →
  service ready → recognizer, so `App.jsx` stays readable),
  `frontend/src/lib/asr/index.js` (the one place an ASR adapter is chosen),
  `frontend/src/lib/asr/scriptedRecognizer.js` (the `?asr=fake` recognizer),
  and `frontend/src/test/fakeRecognizer.js`.
- **Extra error code**: `forbidden_origin` (403) for the `Origin` check, and
  `not_found` (404) for unknown routes.
- **Startup**: the server binds and logs *before* importing
  torch/transformers (measured at 4-55s on this machine), and that import
  runs on the model-load thread while `/api/health` answers `loading`. The
  frontend retries "no answer" for 15s, which only has to cover Python
  starting up (~2s), and waits on `loading` with no cap. (Fixed 2026-09-27:
  importing first had kept the port closed for the whole import.)
- **MIC button label**: `MIC ON` rather than `LISTENING` while on, because it
  is "on" before it is actually listening (connecting, privacy notice); the
  status strip shows the true state.
- **Typing is disabled** while live phrases are still playing after the mic
  goes off, not only while the mic is on (the queue owns `submittedWords`
  until it drains).
