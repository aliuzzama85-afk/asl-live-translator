"""Local HTTP service: live English phrase in, ASL gloss + lookup words out.

The Stage 1-2 bridge between the browser (Web Speech API, JS) and
`gloss_model` (Python). See `pipeline/STAGE1_2_PLAN.md` Sections 1 and 3 for
the design; the short version:

- Python standard library only (`http.server`), no web framework: one
  route, one local user, nothing new to audit.
- Single-threaded on purpose, so the PyTorch model is never called from two
  threads at once. The model loads once, on a background thread at
  startup, so `GET /api/health` can report `"loading"` while it loads.
- Binds `127.0.0.1` only. The Vite dev server proxies `/api/*` here, so the
  browser sees one origin.
- Sanitizes and length-limits input, rate-limits requests, checks `Origin`,
  and never logs phrase text (it's the user's speech).

Run (from the repo root, with the venv active):

    python -m pipeline.gloss_server

or `npm run dev:live` in `frontend/`, which starts this and Vite together.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from pose_library.gloss_tokens import gloss_to_lookup_words

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# checkpoints_v2 explicitly: gloss_model.config.CHECKPOINT_DIR still defaults
# to the older v1 checkpoint (STAGE1_2_PLAN.md "Known gotchas").
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "gloss_model" / "checkpoints_v2"
DEFAULT_PORT = 8765
DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

MAX_BODY_BYTES = 4096
"""Request-body cap. A phrase is at most a few hundred characters (translate()
itself caps text at MAX_INPUT_CHARS = 500), so 4KB is generous."""

DRAIN_LIMIT_BYTES = 64 * 1024
"""An over-cap body up to this size is read and discarded before the error
reply, so the client gets the JSON error rather than a reset connection."""

WARMUP_TEXT = "hello"


@dataclass(frozen=True)
class ServerConfig:
    """Runtime settings, from the environment (see `.env.example`).

    Attributes:
        checkpoint_dir: The fine-tuned model to serve.
        port: Port on 127.0.0.1.
        allowed_origins: Origins a browser request may come from.
        rate_limit_per_minute: Requests allowed per client per minute.
        host: Always 127.0.0.1 -- local only, by design.
    """

    checkpoint_dir: Path
    port: int
    allowed_origins: frozenset[str]
    rate_limit_per_minute: int
    host: str = "127.0.0.1"


def _positive_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def load_config(env: dict[str, str] | None = None) -> ServerConfig:
    """Reads `ServerConfig` from environment variables.

    Args:
        env: Variables to read (defaults to `os.environ`). Recognized:
            `GLOSS_CHECKPOINT_DIR` (relative paths resolve against the repo
            root), `GLOSS_SERVER_PORT`, `ALLOWED_ORIGINS` (comma-separated),
            `RATE_LIMIT_PER_MINUTE`.

    Returns:
        The resolved config.

    Raises:
        ValueError: If a numeric variable isn't a positive integer.
    """
    env = dict(os.environ if env is None else env)
    checkpoint = env.get("GLOSS_CHECKPOINT_DIR", "").strip()
    checkpoint_dir = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT_DIR
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    origins_raw = env.get("ALLOWED_ORIGINS", "").strip()
    origins = (
        frozenset(o.strip() for o in origins_raw.split(",") if o.strip())
        if origins_raw
        else frozenset(DEFAULT_ALLOWED_ORIGINS)
    )
    return ServerConfig(
        checkpoint_dir=checkpoint_dir,
        port=_positive_int(env, "GLOSS_SERVER_PORT", DEFAULT_PORT),
        allowed_origins=origins,
        rate_limit_per_minute=_positive_int(
            env, "RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE
        ),
    )


def sanitize_text(text: str) -> str:
    """Strips control/format characters and collapses whitespace.

    Args:
        text: Raw phrase text from the browser.

    Returns:
        The text with every control or invisible format character (Unicode
        categories Cc, Cf, Zl, Zp: newlines, NUL, zero-width characters,
        etc.) replaced by a space, whitespace runs collapsed to one space,
        and the ends trimmed.
    """
    cleaned = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch
        for ch in text
    )
    return " ".join(cleaned.split())


class RateLimiter:
    """A token bucket per client: `per_minute` requests, refilled continuously.

    Args:
        per_minute: Bucket size and refill rate.
        clock: Monotonic seconds; injectable for tests.
    """

    def __init__(
        self, per_minute: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._capacity = float(per_minute)
        self._refill_per_second = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client: str) -> bool:
        """Takes one token for `client` if one is available.

        Args:
            client: The client key (its IP address).

        Returns:
            Whether the request may proceed.
        """
        now = self._clock()
        tokens, last = self._buckets.get(client, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill_per_second)
        if tokens < 1.0:
            self._buckets[client] = (tokens, now)
            return False
        self._buckets[client] = (tokens - 1.0, now)
        return True


Loader = Callable[[], tuple[Any, Any]]
TranslateFn = Callable[[str, Any, Any], str]


def default_loader(checkpoint_dir: Path) -> Loader:
    """Returns a loader for a real checkpoint via `gloss_model.inference`.

    Args:
        checkpoint_dir: The checkpoint to load.

    Returns:
        A zero-argument callable returning `(model, tokenizer)`.
    """

    def load() -> tuple[Any, Any]:
        from gloss_model.inference import load_model

        return load_model(str(checkpoint_dir))

    return load


class GlossService:
    """Holds the model (loaded once) and turns a phrase into a response body.

    Args:
        loader: Returns `(model, tokenizer)`; called exactly once, by `load`.
        translate_fn: `gloss_model.inference.translate`, or a test double.
        checkpoint_label: Name reported by `/api/health` (not a full path).
    """

    def __init__(
        self, loader: Loader, translate_fn: TranslateFn, checkpoint_label: str
    ) -> None:
        self._loader = loader
        self._translate = translate_fn
        self.checkpoint_label = checkpoint_label
        self.status = "loading"
        self.failure: str | None = None
        self.load_count = 0
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Loads the model and runs one warm-up translation.

        Sets `status` to `"ready"`, or `"failed"` (with `failure`) if loading
        raises. Never raises itself: it runs on a background thread.
        """
        try:
            self.load_count += 1
            self._model, self._tokenizer = self._loader()
            self._translate(WARMUP_TEXT, self._model, self._tokenizer)
        except Exception as exc:
            logger.exception("Loading the gloss model failed")
            self.failure = f"{type(exc).__name__}: {exc}"
            self.status = "failed"
            return
        self.status = "ready"

    def gloss(self, text: str) -> dict[str, Any]:
        """Translates one sanitized phrase.

        Args:
            text: Sanitized phrase text.

        Returns:
            `{"gloss", "words", "dropped", "inference_ms"}`.

        Raises:
            gloss_model.inference.EmptyInputError, InputTooLongError: From
                `translate()`.
        """
        started = time.perf_counter()
        gloss = self._translate(text, self._model, self._tokenizer)
        inference_ms = round((time.perf_counter() - started) * 1000)
        words, dropped = gloss_to_lookup_words(gloss)
        return {
            "gloss": gloss,
            "words": words,
            "dropped": dropped,
            "inference_ms": inference_ms,
        }


def make_handler(
    service: GlossService, config: ServerConfig, limiter: RateLimiter
) -> type[BaseHTTPRequestHandler]:
    """Builds the request-handler class bound to one service/config/limiter.

    Args:
        service: The gloss service.
        config: Server settings (for `allowed_origins`).
        limiter: The rate limiter.

    Returns:
        A `BaseHTTPRequestHandler` subclass for `HTTPServer`.
    """

    class GlossRequestHandler(BaseHTTPRequestHandler):
        server_version = "gloss-server"
        sys_version = ""  # don't advertise the Python version

        def log_message(self, format: str, *args: Any) -> None:
            # Method, path, and status only; request bodies (speech) are
            # never logged.
            logger.info("%s %s", self.address_string(), format % args)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, code: str, message: str) -> None:
            self._send_json(status, {"error": code, "message": message})

        def do_GET(self) -> None:
            if self.path != "/api/health":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "No such route.")
                return
            payload = {"status": service.status, "checkpoint": service.checkpoint_label}
            if service.failure:
                payload["message"] = service.failure
            self._send_json(HTTPStatus.OK, payload)

        def _read_body(self) -> bytes | None:
            """Reads the request body *before* any response is sent.

            Replying while the client's body is still unread makes the OS
            reset the connection on close (seen on Windows), so the client
            gets a connection error instead of the JSON error. So every POST
            reads its body first, then decides. A body over the cap is read
            and discarded up to `DRAIN_LIMIT_BYTES`; past that the connection
            is closed without draining.

            Returns:
                The body, or `None` if an error response was already sent.
            """
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length < 0:
                self._error(
                    HTTPStatus.BAD_REQUEST, "bad_request", "Content-Length required."
                )
                return None
            if length > MAX_BODY_BYTES:
                if length <= DRAIN_LIMIT_BYTES:
                    self.rfile.read(length)
                self.close_connection = True
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "bad_request",
                    f"Body over {MAX_BODY_BYTES} bytes.",
                )
                return None
            return self.rfile.read(length)

        def do_POST(self) -> None:
            raw = self._read_body()
            if raw is None:
                return

            if self.path != "/api/gloss":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "No such route.")
                return

            origin = self.headers.get("Origin")
            if origin is not None and origin not in config.allowed_origins:
                self._error(
                    HTTPStatus.FORBIDDEN, "forbidden_origin", "Origin not allowed."
                )
                return

            if not limiter.allow(self.client_address[0]):
                self._error(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "rate_limited",
                    "Too many requests; slow down.",
                )
                return

            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(HTTPStatus.BAD_REQUEST, "bad_request", "Body must be JSON.")
                return
            phrase_id = data.get("phrase_id") if isinstance(data, dict) else None
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("text"), str)
                or (
                    phrase_id is not None
                    and (not isinstance(phrase_id, int) or isinstance(phrase_id, bool))
                )
            ):
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    "bad_request",
                    'Expected {"text": string, "phrase_id": integer (optional)}.',
                )
                return

            if service.status != "ready":
                message = (
                    "The translation model failed to load."
                    if service.status == "failed"
                    else "The translation model is still loading."
                )
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "not_ready", message)
                return

            # Imported here, not at module top: gloss_model.inference pulls in
            # transformers (~4s), and a top-level import would delay the
            # server from listening at all, so /api/health couldn't report
            # "loading" during startup. By now the load thread has imported it.
            from gloss_model.inference import EmptyInputError, InputTooLongError

            text = sanitize_text(data["text"])
            try:
                result = service.gloss(text)
            except EmptyInputError:
                self._error(HTTPStatus.BAD_REQUEST, "empty_input", "Text is empty.")
                return
            except InputTooLongError as exc:
                self._error(HTTPStatus.BAD_REQUEST, "input_too_long", str(exc))
                return
            except Exception:
                logger.exception("Translation failed")
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal",
                    "Translation failed; see the server log.",
                )
                return

            self._send_json(
                HTTPStatus.OK, {"phrase_id": phrase_id, "text": text, **result}
            )

    return GlossRequestHandler


def create_server(
    config: ServerConfig,
    service: GlossService,
    limiter: RateLimiter | None = None,
) -> HTTPServer:
    """Builds the (not yet serving) HTTP server.

    Args:
        config: Settings; binds `config.host:config.port` (port 0 picks a free
            port, used by tests).
        service: The gloss service (loading is the caller's job).
        limiter: Defaults to one sized from `config`.

    Returns:
        A single-threaded `HTTPServer`.
    """
    limiter = limiter or RateLimiter(config.rate_limit_per_minute)
    return HTTPServer(
        (config.host, config.port), make_handler(service, config, limiter)
    )


def main() -> None:
    """CLI entry point: loads `.env`, starts the model load, serves forever."""
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    try:
        config = load_config()
    except ValueError as exc:
        print(f"Invalid configuration: {exc}")
        sys.exit(1)
    if not config.checkpoint_dir.is_dir():
        print(
            f"Model checkpoint not found at {config.checkpoint_dir}.\n"
            "It's gitignored, so it only exists where it was downloaded; see "
            "KAGGLE.md, or set GLOSS_CHECKPOINT_DIR."
        )
        sys.exit(1)

    from gloss_model.inference import translate

    service = GlossService(
        default_loader(config.checkpoint_dir),
        translate,
        checkpoint_label=config.checkpoint_dir.name,
    )
    server = create_server(config, service)

    def load_then_announce() -> None:
        service.load()
        if service.status == "ready":
            logger.info("Model ready (%s).", service.checkpoint_label)

    threading.Thread(target=load_then_announce, daemon=True).start()
    logger.info(
        "Gloss server on http://%s:%d (loading %s...)",
        config.host,
        config.port,
        service.checkpoint_label,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
