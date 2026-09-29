import pytest

from modules.llm_client import extract_json, LLMOutputError


def test_extracts_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extracts_plain_array():
    assert extract_json('[{"a": 1}]') == [{"a": 1}]


def test_extracts_fenced_object():
    raw = '```json\n{"a": 1, "b": 2}\n```'
    assert extract_json(raw) == {"a": 1, "b": 2}


def test_extracts_fenced_array():
    """This is the exact shape that previously broke: extract_json only
    ever looked for {...}, so array-returning calls (scenarios, step
    narration) fell back to fragile duplicated parsing logic."""
    raw = '```json\n[{"scenario": "x", "mapping": "y"}]\n```'
    assert extract_json(raw) == [{"scenario": "x", "mapping": "y"}]


def test_extracts_json_from_surrounding_prose():
    raw = 'Sure, here you go:\n[{"a": 1}]\nHope that helps!'
    assert extract_json(raw) == [{"a": 1}]


def test_raises_llm_output_error_on_pure_prose():
    """A model replying with a clarifying question instead of JSON should
    raise a catchable error carrying the raw text, not a bare crash."""
    raw = "I need more information to solve this -- what is d referring to?"
    with pytest.raises(LLMOutputError) as exc_info:
        extract_json(raw)
    assert exc_info.value.raw_output == raw


def test_raises_llm_output_error_on_truncated_json():
    raw = '```json\n[{"scenario": "unterminated...'
    with pytest.raises(LLMOutputError):
        extract_json(raw)


def test_raises_llm_output_error_on_invalid_json_syntax():
    raw = '{"a": 1, "b": }'  # syntactically broken
    with pytest.raises(LLMOutputError):
        extract_json(raw)


# ---------------------------------------------------------------- validate_model


def _client_with(is_available_result, models):
    from modules.llm_client import LMStudioClient
    client = LMStudioClient()
    client.is_available = lambda: is_available_result
    client.list_models = lambda: models
    return client


def test_validate_model_success_when_loaded():
    client = _client_with((True, "Connected."), ["model-a", "model-b"])
    ok, msg = client.validate_model("model-a")
    assert ok is True
    assert "model-a" in msg
    assert "loaded and ready" in msg


def test_validate_model_fails_when_server_unreachable():
    client = _client_with((False, "Could not reach LM Studio."), [])
    ok, msg = client.validate_model("model-a")
    assert ok is False
    assert msg == "Could not reach LM Studio."


def test_validate_model_fails_when_no_models_loaded():
    client = _client_with((True, "Connected."), [])
    ok, msg = client.validate_model("model-a")
    assert ok is False
    assert "no models are loaded" in msg.lower()


def test_validate_model_fails_when_model_not_in_loaded_list():
    client = _client_with((True, "Connected."), ["model-a", "model-b"])
    ok, msg = client.validate_model("model-typo")
    assert ok is False
    assert "model-typo" in msg
    assert "model-a" in msg and "model-b" in msg  # lists what IS loaded


def test_validate_model_distinguishes_unreachable_from_not_loaded():
    """The two failure modes must produce genuinely different messages --
    that's the whole point of this check, rather than both looking like
    a generic 'something went wrong'."""
    unreachable_client = _client_with((False, "Could not reach LM Studio."), [])
    _, unreachable_msg = unreachable_client.validate_model("model-a")

    not_loaded_client = _client_with((True, "Connected."), ["model-b"])
    _, not_loaded_msg = not_loaded_client.validate_model("model-a")

    assert unreachable_msg != not_loaded_msg
    assert "reach" in unreachable_msg.lower()
    assert "model-a" in not_loaded_msg


# ---------------------------------------------------------------- vision_extract_work

class _FakeChoice:
    def __init__(self, content):
        self.message = type("Msg", (), {"content": content})()


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletionsAPI:
    def __init__(self, reply):
        self.reply = reply
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeCompletionResponse(self.reply)


def _client_with_fake_vision(reply):
    from modules.llm_client import LMStudioClient
    client = LMStudioClient()
    fake_completions = _FakeCompletionsAPI(reply)
    client._client = type("FakeSDK", (), {
        "chat": type("Chat", (), {"completions": fake_completions})(),
    })()
    return client, fake_completions


def test_vision_extract_work_returns_transcribed_text():
    client, fake = _client_with_fake_vision("a = (v_f - v_i) / t\na = (20 - 8) / 6\na = 2.0")
    result = client.vision_extract_work(b"fake image bytes", mime_type="image/png")
    assert result == "a = (v_f - v_i) / t\na = (20 - 8) / 6\na = 2.0"


def test_vision_extract_work_sends_base64_image_and_grading_prompt():
    client, fake = _client_with_fake_vision("a = 2.0")
    client.vision_extract_work(b"fake image bytes", mime_type="image/jpeg")
    messages = fake.last_call_kwargs["messages"]
    system_msg = messages[0]["content"]
    assert "handwritten" in system_msg.lower()
    assert "do not solve" in system_msg.lower()
    user_content = messages[1]["content"]
    image_block = next(c for c in user_content if c["type"] == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vision_extract_work_and_vision_extract_use_different_prompts():
    """Regression guard: these two methods must NOT share a prompt --
    one transcribes a problem STATEMENT, the other a student's worked
    STEPS, and conflating them would produce garbled results for both."""
    client, fake = _client_with_fake_vision("some text")
    client.vision_extract_work(b"img", mime_type="image/png")
    work_system_prompt = fake.last_call_kwargs["messages"][0]["content"]

    client.vision_extract(b"img", mime_type="image/png")
    statement_system_prompt = fake.last_call_kwargs["messages"][0]["content"]

    assert work_system_prompt != statement_system_prompt
    assert "problem statement" in statement_system_prompt.lower()
    assert "handwritten" in work_system_prompt.lower()


# ---------------------------------------------------------------- connection check speed
#
# The sidebar's connection check runs on every fresh session and gates when the main panel can draw
# (see ui/sidebar.py), so is_available()/list_models() failing SLOWLY when nothing is listening was a
# real, user-visible bug (the OpenAI SDK's default retries/timeouts are tuned for a real completion
# call, not a "is anything there?" ping). These tests use real throwaway sockets rather than mocking
# httpx/OpenAI internals, so they exercise the actual TCP-probe-before-HTTP code path end to end.

import socket
import time

import config
from modules.llm_client import LMStudioClient, _server_reachable


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_server_reachable_false_for_a_closed_port():
    port = _free_port()  # nothing is listening here
    assert _server_reachable(f"http://127.0.0.1:{port}/v1") is False


def test_is_available_fails_fast_when_nothing_is_listening(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(config.settings, "lm_studio_base_url", f"http://127.0.0.1:{port}/v1")
    client = LMStudioClient()
    start = time.monotonic()
    ok, msg = client.is_available()
    elapsed = time.monotonic() - start
    assert ok is False
    assert "Could not reach LM Studio" in msg
    assert elapsed < 2.0  # previously this could take 10+ seconds (SDK retries/backoff)


def test_list_models_returns_empty_fast_when_nothing_is_listening(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(config.settings, "lm_studio_base_url", f"http://127.0.0.1:{port}/v1")
    client = LMStudioClient()
    start = time.monotonic()
    assert client.list_models() == []
    assert time.monotonic() - start < 2.0


def test_is_available_true_and_models_listed_for_a_real_server(monkeypatch):
    """A minimal HTTP server standing in for LM Studio -- confirms the fast
    TCP pre-check doesn't accidentally short-circuit the GENUINELY
    reachable case."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"object": "list",
                                 "data": [{"id": "model-b", "object": "model"},
                                          {"id": "model-a", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(config.settings, "lm_studio_base_url", f"http://127.0.0.1:{port}/v1")
        client = LMStudioClient()
        ok, msg = client.is_available()
        assert ok is True
        assert client.list_models() == ["model-a", "model-b"]  # sorted
    finally:
        server.shutdown()


def test_is_available_bounded_when_port_accepts_but_never_responds(monkeypatch):
    """A port that accepts the TCP connection (so the fast pre-check alone
    wouldn't catch this) but the process behind it never sends an HTTP
    response -- the probe client's own short timeout must still bound
    this, not the SDK's much longer default."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        monkeypatch.setattr(config.settings, "lm_studio_base_url", f"http://127.0.0.1:{port}/v1")
        client = LMStudioClient()
        start = time.monotonic()
        ok, msg = client.is_available()
        elapsed = time.monotonic() - start
        assert ok is False
        assert elapsed < 8.0
    finally:
        listener.close()
