"""Failure explanations: what the model sent, why it was unusable, and what to try -- from facts, not guesses."""
from types import SimpleNamespace

import pytest

from config import settings
from modules import llm_diagnosis as ld
from modules.equation_engine import extract_model
from modules.llm_client import LLMOutputError, LLMText, LMStudioClient, extract_json
from modules.verifier import VerificationReport
from tests.conftest import FakeClient


def err(kind, raw="", **diag):
    return LLMOutputError("x", raw_output=raw, kind=kind, diagnostics=diag)


def texts(d):
    return " ".join(c.text + " " + c.fix for c in d.causes)


# ------------------------------------------------------------------------------------------ the client records how the call went

def _completion(content, finish="stop", completion_tokens=10, reasoning=None):
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(model="qwen", choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=completion_tokens))


def _client(response):
    c = LMStudioClient.__new__(LMStudioClient)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    return c


def test_a_normal_reply_is_still_a_string_and_carries_its_call_metadata():
    out = _client(_completion('{"a": 1}')).chat("s", "u", 0.1)
    assert isinstance(out, str) and out == '{"a": 1}' and isinstance(out, LLMText)
    assert out.meta["finish_reason"] == "stop" and out.meta["completion_tokens"] == 10 and out.meta["max_tokens"] == settings.max_tokens


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_an_empty_reply_is_an_error_that_says_how_the_call_ended(content):
    with pytest.raises(LLMOutputError) as info:
        _client(_completion(content, finish="length", completion_tokens=2048, reasoning="x" * 900)).chat("s", "u", 0.1)
    e = info.value
    assert e.kind == "empty" and e.diagnostics["finish_reason"] == "length" and e.diagnostics["completion_tokens"] == 2048 and e.diagnostics["reasoning_chars"] == 900


def test_a_servers_that_reports_nothing_still_works():
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    assert _client(resp).chat("s", "u", 0.1).meta["finish_reason"] is None


def test_extraction_failures_carry_the_call_facts():
    class C(FakeClient):
        def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
            return LLMText("Sorry, I can't help with that.", {"finish_reason": "stop", "completion_tokens": 9, "max_tokens": 2048, "model": "m"})
    with pytest.raises(LLMOutputError) as info:
        extract_model(C(), "problem")
    assert info.value.kind == "no_json" and info.value.diagnostics["completion_tokens"] == 9


def test_schema_failures_list_the_fields():
    class C(FakeClient):
        def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
            return '{"variables": [{"meaning": "no symbol"}], "equations": [], "solve_for": ["a"]}'
    with pytest.raises(LLMOutputError) as info:
        extract_model(C(), "problem")
    assert info.value.kind == "schema" and any("symbol" in p["field"] for p in info.value.diagnostics["problems"])


# ------------------------------------------------------------------------------------------ reasoning models and JSON extraction

def test_a_thinking_block_with_braces_is_not_mistaken_for_the_answer():
    raw = '<think>maybe {"a": 0} is the shape. Let me work it out {x}.</think>\n```json\n{"answer": 2}\n```'
    assert extract_json(raw) == {"answer": 2}


def test_an_unclosed_thinking_block_is_its_own_failure():
    with pytest.raises(LLMOutputError) as info:
        extract_json('<think>The car accelerates. Let me consider {the units}...')
    assert info.value.kind == "unclosed_thinking"


@pytest.mark.parametrize("raw, kind", [("Just prose.", "no_json"), ('{"a": [1, 2', "truncated"), ('{"a": 1,}', "bad_json")])
def test_each_json_failure_has_its_kind(raw, kind):
    with pytest.raises(LLMOutputError) as info:
        extract_json(raw)
    assert info.value.kind == kind


# ------------------------------------------------------------------------------------------ the explanations

def test_an_empty_reply_that_hit_the_limit_points_at_the_token_setting_first():
    d = ld.diagnose_llm_output(err("empty", finish_reason="length", completion_tokens=2048, max_tokens=2048, reasoning_chars=0, model="m"), "Deriving equations")
    assert d.kind == "empty" and d.causes[0].likelihood == "most likely"
    assert "Max response tokens" in d.causes[0].fix and "2048" in d.causes[0].text and "thinking" in d.causes[0].text
    assert ("Finish reason", "length (stopped because it ran out of tokens)") in d.details and d.stage == "Deriving equations"


def test_an_empty_reply_with_a_reasoning_field_says_so():
    d = ld.diagnose_llm_output(err("empty", finish_reason="stop", completion_tokens=500, max_tokens=2048, reasoning_chars=1800))
    assert "reasoning field" in d.causes[0].text and "thinking" in d.causes[0].fix.lower()
    assert ("Separate reasoning text", "1800 characters (not part of the reply)") in d.details


def test_an_empty_reply_after_zero_tokens_points_at_the_context_window():
    d = ld.diagnose_llm_output(err("empty", finish_reason="stop", completion_tokens=0, max_tokens=2048))
    assert "context" in d.causes[0].text.lower() and "4096" in d.causes[0].fix


def test_when_the_server_reports_nothing_the_diagnosis_says_it_does_not_know():
    d = ld.diagnose_llm_output(err("empty"))
    assert "did not report why" in texts(d) and "Developer" in texts(d)


@pytest.mark.parametrize("reply, expect", [
    ("Could you please specify which acceleration you mean?", "clarifying question"),
    ("I'm sorry, I can't help with that request.", "refused"),
    ("The acceleration is (20 - 8) / 6 = 2 m/s^2, and the distance is 84 m.", "prose"),
])
def test_prose_replies_are_told_apart(reply, expect):
    d = ld.diagnose_llm_output(err("no_json", reply))
    assert expect in texts(d) and d.reply_excerpt.startswith(reply[:20]) and d.causes[0].likelihood == "most likely"


def test_a_long_reply_is_excerpted_but_kept_whole_for_the_raw_view():
    d = ld.diagnose_llm_output(err("no_json", "word " * 400))
    assert len(d.reply_excerpt) < 450 and d.reply_excerpt.endswith("…") and len(d.raw_output) == 2000


def test_a_truncated_reply_blames_the_limit_when_the_numbers_say_so():
    d = ld.diagnose_llm_output(err("truncated", '{"a": [1,', finish_reason="length", completion_tokens=2048, max_tokens=2048))
    assert d.causes[0].likelihood == "most likely" and "limit" in d.causes[0].text and "Max response tokens" in d.causes[0].fix


def test_malformed_json_names_the_offending_line():
    raw = '{\n  "a": 1,\n  "b": 2,\n}'
    with pytest.raises(LLMOutputError) as info:
        extract_json(raw)
    d = ld.diagnose_llm_output(info.value)
    assert any(k == "Offending line" and "}" in v for k, v in d.details) and "trailing commas" in texts(d) and "Extraction temperature" in texts(d)


def test_schema_problems_are_listed():
    d = ld.diagnose_llm_output(err("schema", "{}", problems=[{"field": "equations.0.expression", "message": "Field required"}]))
    assert ("Problem", "equations.0.expression: Field required") in d.details


def test_unclosed_thinking_is_explained():
    d = ld.diagnose_llm_output(err("unclosed_thinking", "<think>..."))
    assert "still thinking" in d.title and "Max response tokens" in d.causes[0].fix


def test_the_facts_table_always_names_the_model_server_and_temperature():
    d = ld.diagnose_llm_output(err("no_json", "x"))
    labels = [k for k, _ in d.details]
    assert {"Model asked for", "LM Studio address", "Extraction temperature", "Reply length"} <= set(labels)


# ------------------------------------------------------------------------------------------ everything else the pipeline can throw

def _status_error(code, message):
    import httpx
    from openai import APIStatusError
    req = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    return APIStatusError(message, response=httpx.Response(code, request=req), body=None)


def test_a_dropped_connection_and_a_timeout_are_different_explanations():
    import httpx
    from openai import APIConnectionError, APITimeoutError
    req = httpx.Request("POST", "http://localhost:1234/v1")
    lost = ld.diagnose_exception(APIConnectionError(request=req))
    slow = ld.diagnose_exception(APITimeoutError(request=req))
    assert lost.kind == "connection" and "Start Server" in lost.causes[0].fix
    assert slow.kind == "timeout" and "smaller" in slow.causes[0].fix


def test_server_errors_are_classified_by_status_and_message():
    assert ld.diagnose_exception(_status_error(404, "model 'x' not found")).kind == "model_missing"
    ctx = ld.diagnose_exception(_status_error(400, "context length exceeded: n_ctx 2048"))
    assert ctx.kind == "context_length" and "context length" in ctx.causes[0].fix
    other = ld.diagnose_exception(_status_error(500, "boom"))
    assert other.kind == "server_error" and "500" in other.title


def test_a_symbolic_timeout_suggests_the_numerical_route():
    from modules.timeout_utils import ComputationTimeoutError
    d = ld.diagnose_exception(ComputationTimeoutError(10.0, "solve"))
    assert d.kind == "symbolic_timeout" and "numerical" in d.causes[0].fix and str(settings.computation_timeout_seconds).rstrip("0").rstrip(".") in d.summary


def test_a_missing_package_gets_the_pip_command():
    d = ld.diagnose_exception(ModuleNotFoundError("No module named 'docx'", name="docx"))
    assert d.kind == "missing_package" and "pip install python-docx" in d.causes[0].fix


def test_an_unreadable_equation_is_explained():
    from sympy.core.sympify import SympifyError
    d = ld.diagnose_exception(SympifyError("a**"))
    assert d.kind == "equation_syntax" and "^" in d.causes[0].text


def test_an_unknown_exception_is_called_what_it_is_and_still_not_a_traceback():
    d = ld.diagnose_exception(RuntimeError("weird"), stage="Verifying the derivation...")
    assert d.kind == "unexpected" and "RuntimeError" in d.summary and d.stage == "Verifying the derivation..." and "app.log" in d.causes[0].fix


def test_the_stage_recorded_on_the_exception_is_used():
    e = RuntimeError("x")
    e.pipeline_stage = "Computing step-by-step solution..."
    assert ld.diagnose_exception(e).stage == "Computing step-by-step solution..."


# ------------------------------------------------------------------------------------------ a derivation that never verified

def _failed(*labels):
    r = VerificationReport()
    for lab in labels:
        r.add(lab, False, "detail for " + lab)
    return r


def test_a_verified_report_needs_no_diagnosis():
    assert ld.diagnose_verification(VerificationReport()) is None


def test_failed_checks_become_specific_advice():
    d = ld.diagnose_verification(_failed("Dimensional consistency: accel", "Determinacy"))
    assert d.kind == "unverified" and d.severity == "warning"
    assert "units" in d.causes[0].fix or "unit" in d.causes[0].fix
    assert any("independent equations" in c.text for c in d.causes)
    assert ("Determinacy", "detail for Determinacy") in d.details and str(settings.max_verification_retries) in d.summary


def test_an_unknown_failed_check_still_gets_generic_advice():
    d = ld.diagnose_verification(_failed("Some brand new check"))
    assert d.causes and "failed checks below" in d.causes[0].fix
