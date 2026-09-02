import json

from modules.self_consistency import run_self_consistency_check


def _kinematics_payload():
    return {
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    }


class _FixedClient:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, **kwargs):
        return json.dumps(self.payload)


class _AlternatingClient:
    """Returns payload_a on the first call, payload_b on every call after."""
    def __init__(self, payload_a, payload_b):
        self.payload_a = payload_a
        self.payload_b = payload_b
        self.call_count = 0

    def chat(self, **kwargs):
        self.call_count += 1
        return json.dumps(self.payload_a if self.call_count == 1 else self.payload_b)


class _RaisingClient:
    def chat(self, **kwargs):
        raise RuntimeError("engine error")


def test_consistent_extraction_across_runs():
    client = _FixedClient(_kinematics_payload())
    result = run_self_consistency_check(client, "a car problem", runs=3)
    assert result.consistent is True
    assert result.runs == 3
    assert all(s == 1.0 for s in result.shapes_match)


def test_inconsistent_extraction_flagged():
    payload_a = _kinematics_payload()
    payload_b = dict(payload_a)
    payload_b["equations"] = [
        {"name": "accel", "kind": "equation", "expression": "Eq(a, v_f/t)", "derivation": ""},
    ]
    client = _AlternatingClient(payload_a, payload_b)
    result = run_self_consistency_check(client, "an ambiguous problem", runs=3)
    assert result.consistent is False
    assert any(s < 0.7 for s in result.shapes_match)


def test_all_runs_failing_returns_none_consistent():
    result = run_self_consistency_check(_RaisingClient(), "a problem", runs=3)
    assert result.consistent is None
    assert len(result.errors) == 3
    assert all(m is None for m in result.models)


def test_partial_failures_still_compares_successful_runs():
    class PartiallyFailingClient:
        def __init__(self):
            self.call_count = 0

        def chat(self, **kwargs):
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("one bad run")
            return json.dumps(_kinematics_payload())

    client = PartiallyFailingClient()
    result = run_self_consistency_check(client, "a problem", runs=3)
    assert result.consistent is True
    assert len(result.errors) == 1
    assert len(result.shapes_match) == 1


def test_runs_parameter_clamped_to_valid_range():
    client = _FixedClient(_kinematics_payload())
    result_low = run_self_consistency_check(client, "a problem", runs=0)
    assert result_low.runs == 2
    result_high = run_self_consistency_check(client, "a problem", runs=100)
    assert result_high.runs == 5


def test_min_similarity_threshold_respected():
    payload_a = _kinematics_payload()
    payload_b = dict(payload_a)
    payload_b["equations"] = [
        {"name": "accel", "kind": "equation", "expression": "Eq(a, v_f/t)", "derivation": ""},
    ]
    client = _AlternatingClient(payload_a, payload_b)
    result = run_self_consistency_check(client, "a problem", runs=2, min_similarity=0.0)
    assert result.consistent is True
