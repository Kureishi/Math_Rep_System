"""
Shared fixtures for the test suite.

FakeClient stands in for LMStudioClient so tests never need a live LM
Studio server -- it routes based on distinctive phrases in the system
prompt (the same routing technique used throughout manual testing while
building this app), and can be pointed at a custom extraction payload via
`payload_json`.
"""
import pytest


class FakeClient:
    """A stand-in LLM client. Routes by looking for phrases that only
    appear in one specific system prompt in the real app, so it can tell
    apart extraction / narration / scenario / independent-cross-check
    calls without any special-casing from the caller."""

    def __init__(self, payload_json: str = "{}", final_answers: dict[str, float] | None = None,
                 narration: str | None = None, scenarios: str | None = None):
        self.payload_json = payload_json
        self.final_answers = final_answers or {}
        self.narration = narration or '["step explanation"]'
        self.scenarios = scenarios or '[{"scenario": "test scenario", "mapping": "x -> y"}]'
        self.calls: list[tuple[str, str]] = []  # (system, user) log, for call-count assertions

    def chat(self, system: str, user: str, temperature: float = 0.0,
              json_mode: bool = False, model: str | None = None) -> str:
        self.calls.append((system, user))
        combined = system + user
        if "FINAL_NUMERIC_ANSWER" in combined:
            if not self.final_answers:
                return "FINAL_NUMERIC_ANSWER[x]: 0"
            return "\n".join(f"FINAL_NUMERIC_ANSWER[{k}]: {v}" for k, v in self.final_answers.items())
        if "explain math steps" in system.lower():
            return self.narration
        if "creative but mathematically" in system.lower():
            return self.scenarios
        return self.payload_json

    def list_models(self):
        return ["fake-model"]

    def is_available(self):
        return True, "Connected (fake)."


@pytest.fixture
def fake_client_factory():
    """Returns the FakeClient class itself so tests can construct
    instances with custom payload_json/final_answers per test."""
    return FakeClient


# ---------------------------------------------------------------- sample payloads

KINEMATICS_JSON = """{
  "problem_domain": "kinematics",
  "problem_type": "algebraic",
  "variables": [
    {"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
    {"symbol": "u", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
    {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    {"symbol": "a", "meaning": "acceleration", "known_value": null, "unit": "m/s^2"}
  ],
  "equations": [
    {"name": "kinematic velocity equation", "kind": "equation",
     "expression": "Eq(v, u + a*t)", "derivation": "final velocity equals initial velocity plus acceleration times time"}
  ],
  "solve_for": ["a"],
  "assumptions": ["acceleration is uniform (constant)"]
}"""

KINEMATICS_TWO_TARGET_JSON = """{
  "problem_domain": "kinematics",
  "problem_type": "algebraic",
  "variables": [
    {"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
    {"symbol": "u", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
    {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    {"symbol": "a", "meaning": "acceleration", "known_value": null, "unit": "m/s^2"},
    {"symbol": "d", "meaning": "distance traveled", "known_value": null, "unit": "m"}
  ],
  "equations": [
    {"name": "final velocity equation", "kind": "equation", "expression": "Eq(v, a*t + u)", "derivation": "x"},
    {"name": "displacement equation", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2 + t*u)", "derivation": "x"}
  ],
  "solve_for": ["a", "d"],
  "assumptions": ["acceleration is uniform"]
}"""

KINEMATICS_BUGGY_JSON = """{
  "problem_domain": "kinematics",
  "problem_type": "algebraic",
  "variables": [
    {"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
    {"symbol": "u", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
    {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    {"symbol": "a", "meaning": "acceleration", "known_value": null, "unit": "m/s^2"}
  ],
  "equations": [
    {"name": "kinematic velocity equation (missing *t)", "kind": "equation",
     "expression": "Eq(v, u + a)", "derivation": "WRONG on purpose: missing the *t term"}
  ],
  "solve_for": ["a"],
  "assumptions": []
}"""

INEQUALITY_JSON = """{
  "problem_domain": "speed limit constraint",
  "problem_type": "algebraic",
  "variables": [
    {"symbol": "v", "meaning": "speed", "known_value": null, "unit": "mph"},
    {"symbol": "limit", "meaning": "posted speed limit", "known_value": "65", "unit": "mph"}
  ],
  "equations": [
    {"name": "speed constraint", "kind": "inequality", "expression": "v <= limit", "derivation": "x"}
  ],
  "solve_for": ["v"],
  "assumptions": []
}"""

ODE_JSON = """{
  "problem_domain": "radioactive decay",
  "problem_type": "ode",
  "independent_variable": "t",
  "variables": [
    {"symbol": "N", "meaning": "remaining mass", "known_value": null, "unit": "g", "is_function": true},
    {"symbol": "t", "meaning": "time", "known_value": null, "unit": "years", "is_function": false},
    {"symbol": "k", "meaning": "decay constant", "known_value": "0.1", "unit": "1/year", "is_function": false}
  ],
  "equations": [
    {"name": "decay ODE", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))",
     "derivation": "rate of decay proportional to remaining mass"}
  ],
  "initial_conditions": [{"expression": "N(0)", "value": "500"}],
  "solve_for": ["N"],
  "assumptions": ["continuous decay"]
}"""


COUPLED_ODE_JSON = """{
  "problem_domain": "decay chain",
  "problem_type": "ode",
  "independent_variable": "t",
  "variables": [
    {"symbol": "A", "meaning": "parent isotope", "known_value": null, "unit": "g", "is_function": true},
    {"symbol": "B", "meaning": "daughter isotope", "known_value": null, "unit": "g", "is_function": true},
    {"symbol": "t", "meaning": "time", "known_value": null, "unit": "years", "is_function": false},
    {"symbol": "k1", "meaning": "A decay rate", "known_value": "0.5", "unit": "1/year", "is_function": false},
    {"symbol": "k2", "meaning": "B decay rate", "known_value": "0.2", "unit": "1/year", "is_function": false}
  ],
  "equations": [
    {"name": "A decay", "kind": "ode", "expression": "Eq(Derivative(A(t), t), -k1*A(t))", "derivation": "x"},
    {"name": "B production/decay", "kind": "ode",
     "expression": "Eq(Derivative(B(t), t), k1*A(t) - k2*B(t))", "derivation": "x"}
  ],
  "initial_conditions": [
    {"expression": "A(0)", "value": "1000"},
    {"expression": "B(0)", "value": "0"}
  ],
  "solve_for": ["A", "B"],
  "assumptions": []
}"""

MULTI_CONSTRAINT_JSON = """{
  "problem_domain": "resource allocation",
  "problem_type": "algebraic",
  "variables": [
    {"symbol": "x", "meaning": "units of product A", "known_value": null, "unit": null},
    {"symbol": "y", "meaning": "units of product B", "known_value": null, "unit": null}
  ],
  "equations": [
    {"name": "budget constraint", "kind": "inequality", "expression": "x + y <= 10", "derivation": "x"},
    {"name": "time constraint", "kind": "inequality", "expression": "2*x + y <= 15", "derivation": "x"},
    {"name": "x non-negative", "kind": "inequality", "expression": "x >= 0", "derivation": "x"},
    {"name": "y non-negative", "kind": "inequality", "expression": "y >= 0", "derivation": "x"}
  ],
  "solve_for": [],
  "assumptions": []
}"""


@pytest.fixture
def kinematics_json():
    return KINEMATICS_JSON


@pytest.fixture
def kinematics_two_target_json():
    return KINEMATICS_TWO_TARGET_JSON


@pytest.fixture
def kinematics_buggy_json():
    return KINEMATICS_BUGGY_JSON


@pytest.fixture
def inequality_json():
    return INEQUALITY_JSON


@pytest.fixture
def ode_json():
    return ODE_JSON


@pytest.fixture
def coupled_ode_json():
    return COUPLED_ODE_JSON


@pytest.fixture
def multi_constraint_json():
    return MULTI_CONSTRAINT_JSON
