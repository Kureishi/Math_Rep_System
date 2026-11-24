"""Thinking models served by LM Studio: an empty answer field must not be the end of the road (modules/llm_thinking.py, LMStudioClient.chat)."""
from types import SimpleNamespace as NS

import pytest

import config
from modules import llm_thinking as lt
from modules.llm_client import LLMOutputError, LMStudioClient, extract_json


def _chunk(content=None, reasoning=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, reasoning_content=reasoning), finish_reason=finish)])


class _Stream:
    def __init__(self, chunks, endless=False):
        self.chunks, self.endless, self.closed, self.read = chunks, endless, False, 0

    def __iter__(self):
        for c in self.chunks:
            self.read += 1
            yield c
        while self.endless and not self.closed:                      # a model that never stops thinking
            self.read += 1
            yield _chunk(reasoning="hmm ")

    def close(self):
        self.closed = True


def _response(content="", reasoning="", finish="stop"):
    return NS(choices=[NS(message=NS(content=content, reasoning_content=reasoning), finish_reason=finish)], usage=NS(prompt_tokens=10, completion_tokens=5), model="m")


class _API:
    """create() returns the queued replies in order and records every call's keyword arguments."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(*replies, **settings):
    client = LMStudioClient()
    api = _API(*replies)
    client._client = NS(chat=NS(completions=api))
    for k, v in settings.items():
        setattr(config.settings, k, v)
    return client, api


@pytest.fixture(autouse=True)
def _restore_settings():
    saved = (config.settings.thinking_mode, config.settings.reasoning_budget_tokens)
    yield
    config.settings.thinking_mode, config.settings.reasoning_budget_tokens = saved


GOOD = '{"problem_domain": "kinematics", "equations": []}'


# ------------------------------------------------------------------------------------------ helpers

def test_a_runaway_reasoning_stream_is_stopped_at_the_budget():
    stream = _Stream([], endless=True)
    out = lt.collect_stream(stream, reasoning_budget_tokens=50)
    assert out.aborted and stream.closed and stream.read <= 60 and out.content == ""


def test_a_stream_that_answers_within_budget_is_read_whole():
    stream = _Stream([_chunk(reasoning="think "), _chunk(content='{"a": '), _chunk(content='1}', finish="stop")])
    out = lt.collect_stream(stream, 5)
    assert not out.aborted and out.content == '{"a": 1}' and out.reasoning == "think " and out.finish_reason == "stop"


def test_an_inline_think_block_that_never_closes_counts_against_the_budget():
    stream = _Stream([_chunk(content="<think>")] + [_chunk(content="words ")] * 40)
    assert lt.collect_stream(stream, 10).aborted


def test_json_is_recovered_from_reasoning_text():
    text = 'Let me sketch {x} first. Draft {"a": 1, "b": 2}. Final: {"a": 3, "b": [1, 2]} done'
    assert lt.recover_json(text) == '{"a": 3, "b": [1, 2]}'
    assert lt.recover_json("no json, only {x} and {}") is None and lt.recover_json("") is None


# ------------------------------------------------------------------------------------------ the ladder

def test_a_normal_model_costs_one_call_and_nothing_changes():
    client, api = _client(_response(content=GOOD), thinking_mode="auto", reasoning_budget_tokens=0)
    out = client.chat("sys", "user", 0.1, json_mode=True)
    assert out == GOOD and len(api.calls) == 1 and out.meta["attempts"][0]["step"] == "plain"
    assert api.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} and "response_format" not in api.calls[0]


def test_an_empty_reply_is_retried_under_a_json_schema_and_the_answer_used():
    client, api = _client(_response(content="", reasoning="thinking...", finish="length"), _response(content=GOOD), thinking_mode="auto", reasoning_budget_tokens=0)
    out = client.chat("sys", "user", 0.1, json_mode=True)
    assert out == GOOD and len(api.calls) == 2
    assert api.calls[1]["response_format"]["type"] == "json_schema" and [a["step"] for a in out.meta["attempts"]] == ["plain", "json schema"]


def test_json_that_landed_in_the_reasoning_field_is_used():
    """LM Studio applies the schema to the reasoning stream of some models, leaving the answer field empty."""
    client, api = _client(_response(content="", reasoning="", finish="length"), _response(content="", reasoning=GOOD, finish="stop"), thinking_mode="auto", reasoning_budget_tokens=0)
    out = client.chat("sys", "user", 0.1, json_mode=True)
    assert extract_json(out)["problem_domain"] == "kinematics" and out.meta["recovered_from_reasoning"] is True


def test_a_model_that_never_stops_thinking_is_cut_off_and_then_reported_with_what_was_tried():
    client, api = _client(_Stream([], endless=True), _response(content="", reasoning="still thinking", finish="length"), thinking_mode="auto", reasoning_budget_tokens=40)
    with pytest.raises(LLMOutputError) as err:
        client.chat("sys", "user", 0.1, json_mode=True)
    attempts = err.value.diagnostics["attempts"]
    assert err.value.kind == "empty" and attempts[0]["stopped_early"] is True and attempts[1]["step"] == "json schema"
    from modules.llm_diagnosis import diagnose_llm_output
    d = diagnose_llm_output(err.value)
    assert "What the app tried" in dict(d.details) and "Do NOT raise" in d.top_fix and "Enable Thinking" in d.top_fix


def test_options_the_server_rejects_are_dropped_and_not_sent_again():
    from openai import BadRequestError
    bad = BadRequestError("nope", response=NS(status_code=400, request=NS(), headers={}), body=None)
    client, api = _client(bad, _response(content=GOOD), _response(content=GOOD), thinking_mode="auto", reasoning_budget_tokens=0)
    assert client.chat("s", "u", 0.1, json_mode=True) == GOOD
    assert "extra_body" in api.calls[0] and "extra_body" not in api.calls[1]
    client.chat("s", "u", 0.1, json_mode=True)
    assert "extra_body" not in api.calls[2]


def test_leave_mode_sends_nothing_extra_and_never_retries():
    client, api = _client(_response(content="", reasoning="x", finish="length"), thinking_mode="leave", reasoning_budget_tokens=0)
    with pytest.raises(LLMOutputError):
        client.chat("s", "u", 0.1, json_mode=True)
    assert len(api.calls) == 1 and "extra_body" not in api.calls[0] and "response_format" not in api.calls[0]


def test_text_calls_without_json_mode_never_use_a_schema():
    client, api = _client(_response(content="", reasoning="x", finish="length"), thinking_mode="auto", reasoning_budget_tokens=0)
    with pytest.raises(LLMOutputError):
        client.chat("s", "u", 0.4, json_mode=False)
    assert len(api.calls) == 1
