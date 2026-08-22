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
