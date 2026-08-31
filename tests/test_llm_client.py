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
