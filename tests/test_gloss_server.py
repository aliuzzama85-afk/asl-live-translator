"""Tests for pipeline.gloss_server.

A real `HTTPServer` is started on a free port and exercised over real HTTP;
only the model is faked (the project's usual pattern for model I/O). One
extra test loads the real `checkpoints_v2` model and is skipped when it
isn't on disk (it's gitignored, so a fresh clone won't have it).
"""

import http.client
import json
import threading
from pathlib import Path

import pytest

from gloss_model.inference import EmptyInputError, InputTooLongError
from pipeline.gloss_server import (
    DEFAULT_CHECKPOINT_DIR,
    MAX_BODY_BYTES,
    GlossService,
    RateLimiter,
    ServerConfig,
    create_server,
    load_config,
    sanitize_text,
)

ORIGIN = "http://localhost:5173"
CANNED = {
    "can you help me find my phone": "CAN X-YOU HELP X-I FIND X-MY PHONE",
    "where is the bathroom": "WHERE BE BATHROOM",
    "hello": "HALF",
}


class FakeModel:
    """Stands in for (model, tokenizer, translate) and records calls."""

    def __init__(self):
        self.load_calls = 0
        self.translate_calls = []
        self.raise_on = {}

    def loader(self):
        self.load_calls += 1
        return "model", "tokenizer"

    def translate(self, text, model, tokenizer):
        assert (model, tokenizer) == ("model", "tokenizer")
        self.translate_calls.append(text)
        if not text.strip():
            raise EmptyInputError("empty")
        if text in self.raise_on:
            raise self.raise_on[text]
        return CANNED.get(text, "TEST")


def _config(**overrides):
    base = {
        "checkpoint_dir": Path("unused"),
        "port": 0,
        "allowed_origins": frozenset({ORIGIN}),
        "rate_limit_per_minute": 60,
    }
    base.update(overrides)
    return ServerConfig(**base)


@pytest.fixture()
def running():
    """Yields (fake, service, request) for a live server; stops it after."""
    fake = FakeModel()
    service = GlossService(fake.loader, fake.translate, checkpoint_label="fake-ckpt")
    servers = []

    def start(load=True, limiter=None, **config_overrides):
        if load:
            service.load()
        server = create_server(_config(**config_overrides), service, limiter)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    def request(method, path, body=None, headers=None, port=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        payload = (
            raw if raw is not None else (None if body is None else json.dumps(body))
        )
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
        conn.close()
        return resp.status, data

    yield fake, service, start, request
    for server in servers:
        server.shutdown()
        server.server_close()


# --- pure helpers -----------------------------------------------------------


def test_sanitize_text_strips_control_characters_and_collapses_whitespace():
    assert sanitize_text("  where\tis\n\nthe\x00 bath\u200broom  ") == (
        "where is the bath room"
    )


def test_load_config_defaults_to_checkpoints_v2_on_localhost():
    config = load_config({})
    assert config.checkpoint_dir == DEFAULT_CHECKPOINT_DIR
    assert config.checkpoint_dir.name == "checkpoints_v2"
    assert config.host == "127.0.0.1"
    assert "http://localhost:5173" in config.allowed_origins


def test_load_config_reads_env_and_rejects_bad_numbers():
    config = load_config(
        {
            "GLOSS_SERVER_PORT": "9001",
            "ALLOWED_ORIGINS": "http://a.test, http://b.test",
            "RATE_LIMIT_PER_MINUTE": "5",
        }
    )
    assert config.port == 9001
    assert config.allowed_origins == {"http://a.test", "http://b.test"}
    assert config.rate_limit_per_minute == 5
    with pytest.raises(ValueError):
        load_config({"GLOSS_SERVER_PORT": "abc"})
    with pytest.raises(ValueError):
        load_config({"RATE_LIMIT_PER_MINUTE": "0"})


def test_rate_limiter_refills_over_time():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")  # per client
    now[0] = 30.0  # half a minute refills one token
    assert limiter.allow("a")
    assert not limiter.allow("a")


def test_service_loads_once_warms_up_and_reports_failure():
    fake = FakeModel()
    service = GlossService(fake.loader, fake.translate, "x")
    assert service.status == "loading"
    service.load()
    assert service.status == "ready"
    assert fake.load_calls == 1
    assert fake.translate_calls == ["hello"]  # the warm-up

    def broken():
        raise OSError("no such checkpoint")

    failing = GlossService(broken, fake.translate, "x")
    failing.load()
    assert failing.status == "failed"
    assert "no such checkpoint" in failing.failure


# --- over real HTTP ---------------------------------------------------------


def test_gloss_returns_gloss_words_and_dropped(running):
    _, _, start, request = running
    port = start()
    status, body = request(
        "POST",
        "/api/gloss",
        {"text": "can you help me find my phone", "phrase_id": 7},
        {"Origin": ORIGIN},
        port=port,
    )
    assert status == 200
    assert body["phrase_id"] == 7
    assert body["gloss"] == "CAN X-YOU HELP X-I FIND X-MY PHONE"
    assert body["words"] == ["can", "help", "find", "phone"]
    assert body["dropped"] == ["X-YOU", "X-I", "X-MY"]
    assert isinstance(body["inference_ms"], int)


def test_text_is_sanitized_before_the_model_sees_it(running):
    fake, _, start, request = running
    port = start()
    status, body = request(
        "POST", "/api/gloss", {"text": " where\n is the\x00 bathroom "}, port=port
    )
    assert status == 200
    assert fake.translate_calls[-1] == "where is the bathroom"
    assert body["text"] == "where is the bathroom"
    assert body["phrase_id"] is None


def test_model_is_loaded_once_across_many_requests(running):
    fake, _, start, request = running
    port = start()
    for _ in range(5):
        assert request("POST", "/api/gloss", {"text": "hello"}, port=port)[0] == 200
    assert fake.load_calls == 1


def test_health_reports_loading_then_ready(running):
    _, service, start, request = running
    port = start(load=False)
    assert request("GET", "/api/health", port=port) == (
        200,
        {"status": "loading", "checkpoint": "fake-ckpt"},
    )
    status, body = request("POST", "/api/gloss", {"text": "hello"}, port=port)
    assert (status, body["error"]) == (503, "not_ready")
    service.load()
    assert request("GET", "/api/health", port=port)[1]["status"] == "ready"


@pytest.mark.parametrize(
    "body, raw, code",
    [
        ({"text": "   "}, None, "empty_input"),
        ({"text": 42}, None, "bad_request"),
        ({"words": "hi"}, None, "bad_request"),
        ({"text": "hi", "phrase_id": "7"}, None, "bad_request"),
        ({"text": "hi", "phrase_id": True}, None, "bad_request"),
        (None, "not json", "bad_request"),
        (None, '["a list"]', "bad_request"),
    ],
)
def test_bad_requests_get_400_with_a_code(running, body, raw, code):
    _, _, start, request = running
    port = start()
    status, resp = request("POST", "/api/gloss", body, port=port, raw=raw)
    assert (status, resp["error"]) == (400, code)


def test_input_too_long_maps_to_its_code(running):
    fake, _, start, request = running
    port = start()
    fake.raise_on["way too long"] = InputTooLongError("over 500")
    status, resp = request("POST", "/api/gloss", {"text": "way too long"}, port=port)
    assert (status, resp["error"]) == (400, "input_too_long")


def test_oversized_body_is_rejected_without_reaching_the_model(running):
    fake, _, start, request = running
    port = start()
    calls_before = len(fake.translate_calls)
    status, resp = request(
        "POST", "/api/gloss", {"text": "a" * (MAX_BODY_BYTES + 10)}, port=port
    )
    assert (status, resp["error"]) == (400, "bad_request")
    assert len(fake.translate_calls) == calls_before


def test_unexpected_model_error_is_500_without_details(running):
    fake, _, start, request = running
    port = start()
    fake.raise_on["boom"] = RuntimeError("secret internal detail")
    status, resp = request("POST", "/api/gloss", {"text": "boom"}, port=port)
    assert (status, resp["error"]) == (500, "internal")
    assert "secret" not in resp["message"]


def test_disallowed_origin_is_rejected_but_no_origin_is_allowed(running):
    _, _, start, request = running
    port = start()
    status, resp = request(
        "POST",
        "/api/gloss",
        {"text": "hello"},
        {"Origin": "http://evil.test"},
        port=port,
    )
    assert (status, resp["error"]) == (403, "forbidden_origin")
    assert request("POST", "/api/gloss", {"text": "hello"}, port=port)[0] == 200


def test_rate_limit_returns_429(running):
    _, _, start, request = running
    port = start(limiter=RateLimiter(per_minute=2, clock=lambda: 0.0))
    assert request("POST", "/api/gloss", {"text": "hello"}, port=port)[0] == 200
    assert request("POST", "/api/gloss", {"text": "hello"}, port=port)[0] == 200
    status, resp = request("POST", "/api/gloss", {"text": "hello"}, port=port)
    assert (status, resp["error"]) == (429, "rate_limited")


def test_unknown_routes_are_404(running):
    _, _, start, request = running
    port = start()
    assert request("GET", "/api/nope", port=port)[0] == 404
    assert request("POST", "/api/health", {"text": "x"}, port=port)[0] == 404


# --- the real model -----------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_CHECKPOINT_DIR.is_dir(),
    reason="checkpoints_v2 is gitignored and not present on this machine",
)
def test_real_checkpoint_end_to_end():
    from gloss_model.inference import translate
    from pipeline.gloss_server import default_loader

    service = GlossService(default_loader(DEFAULT_CHECKPOINT_DIR), translate, "v2")
    service.load()
    assert service.status == "ready"
    result = service.gloss("can you help me find my phone")
    assert result["gloss"] == "CAN X-YOU HELP X-I FIND X-MY PHONE"
    assert result["words"] == ["can", "help", "find", "phone"]


# --- startup order (regression: the server used to be silent until torch loaded) ---


def test_importing_the_server_does_not_import_torch_or_transformers():
    # main() must be able to bind and log before the heavy stack loads, so the
    # module itself has to stay light. Checked in a fresh interpreter.
    import subprocess
    import sys

    code = (
        "import sys, pipeline.gloss_server; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["False", "False"]


def test_service_imports_the_real_translate_lazily_in_load(monkeypatch):
    import sys
    import types

    calls = []
    fake_inference = types.ModuleType("gloss_model.inference")
    fake_inference.translate = lambda text, model, tok: calls.append(text) or "HALF"
    monkeypatch.setitem(sys.modules, "gloss_model.inference", fake_inference)

    service = GlossService(lambda: ("m", "t"), translate_fn=None, checkpoint_label="x")
    service.load()
    assert service.status == "ready"
    assert calls == ["hello"]  # the warm-up went through the lazily imported translate


def test_cli_answers_health_before_the_model_finishes_loading(tmp_path):
    # The real `python -m pipeline.gloss_server`, pointed at an empty
    # checkpoint directory. It must be listening and reporting "loading" while
    # torch/transformers are still importing (4-55s measured on this machine),
    # not silent until they're done.
    import os
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "GLOSS_CHECKPOINT_DIR": str(tmp_path),
        "GLOSS_SERVER_PORT": str(port),
    }
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.gloss_server"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        status = None
        while time.monotonic() < deadline and status is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout=5
                ) as resp:
                    status = json.loads(resp.read())["status"]
            except OSError:
                time.sleep(0.2)
        assert status == "loading"
    finally:
        proc.kill()
        proc.wait()
