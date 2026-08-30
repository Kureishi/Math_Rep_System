import time
import pytest

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError


# ---------------------------------------------------------------- core mechanism


def test_fast_function_returns_normally():
    assert run_with_timeout(lambda: 2 + 2, timeout=1.0) == 4


def test_slow_function_raises_computation_timeout_error():
    def slow():
        time.sleep(2)
        return "done"

    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow, timeout=0.2)


def test_timeout_returns_control_promptly_not_after_full_duration():
    def slow():
        time.sleep(2)

    t0 = time.time()
    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow, timeout=0.2)
    elapsed = time.time() - t0
    assert elapsed < 1.0  # should return around 0.2s, not wait for the full 2s sleep


def test_non_timeout_exception_propagates_unchanged():
    def raises():
        raise ValueError("a real error")

    with pytest.raises(ValueError, match="a real error"):
        run_with_timeout(raises, timeout=1.0)


def test_args_and_kwargs_passed_through():
    def add(a, b, c=0):
        return a + b + c

    assert run_with_timeout(add, 1, 2, timeout=1.0, c=3) == 6


def test_uses_default_timeout_from_settings(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.2)

    def slow():
        time.sleep(2)

    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow)  # no explicit timeout -- should use the patched setting


def test_error_message_includes_seconds_and_label():
    def slow():
        time.sleep(2)

    try:
        run_with_timeout(slow, timeout=0.2, label="my computation")
        assert False, "expected ComputationTimeoutError"
    except ComputationTimeoutError as e:
        assert "0.2" in str(e)
        assert "my computation" in str(e)


def test_error_message_omits_label_when_not_given():
    def slow():
        time.sleep(2)

    try:
        run_with_timeout(slow, timeout=0.2)
        assert False, "expected ComputationTimeoutError"
    except ComputationTimeoutError as e:
        assert "(" not in str(e)


# ---------------------------------------------------------------- wired-in call sites


def test_verifier_solve_reports_timeout_as_explicit_check(monkeypatch):
    import modules.verifier as verifier_module
    from modules.equation_engine import build_model

    def _raise_timeout(model):
        raise ComputationTimeoutError(10.0, label="algebraic solve")
    monkeypatch.setattr(verifier_module, "_solve_sympy", _raise_timeout)

    class FakeClient:
        def chat(self, **kwargs):
            return "{}"

    model = build_model({
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
    })
    report = verifier_module.verify(model, FakeClient(), "x")
    timeout_checks = [c for c in report.checks if c.label == "Symbolic solve"]
    assert len(timeout_checks) == 1
    assert timeout_checks[0].passed is False
    assert "timed out" in timeout_checks[0].detail.lower()
    assert report.passed is False
    assert report.sympy_numeric_answers == {}


def test_matrix_analyze_partial_degradation_on_timeout(monkeypatch):
    """Regression test: a timeout on eigenvalues/linsolve must not lose
    the still-fast rank-based classification -- genuine partial
    degradation, not all-or-nothing failure."""
    import sympy as sp
    from config import settings
    from modules.matrix_utils import analyze_linear_system

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    A = sp.Matrix([[2, 3], [1, -1]])
    b = sp.Matrix([8, 1])
    result = analyze_linear_system(A, b, ["x", "y"])
    assert result.rank_A == 2
    assert result.consistent is True
    assert result.unique is True
    assert "Unique solution" in result.classification


def test_uncertainty_solve_symbolic_never_raises_on_timeout(monkeypatch):
    from config import settings
    from modules.equation_engine import build_model
    from modules.uncertainty import solve_symbolic_for_target

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    model = build_model({
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
    })
    # with an essentially-zero timeout, this should never raise, whether
    # or not the solve happened to squeak in under the wire
    solve_symbolic_for_target(model, "a")


def test_equivalence_check_degrades_to_undetermined_on_timeout(monkeypatch):
    from config import settings
    from modules.equivalence import check_equivalence

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert result.equivalent is None
    assert result.method == "undetermined"
    assert "timed out" in result.detail.lower()
    assert result.error is None  # this is a timeout, not a parse error


def test_proof_build_returns_none_when_equivalence_check_timed_out(monkeypatch):
    from config import settings
    from modules.equivalence import check_equivalence
    from modules.proof import build_proof

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert build_proof(result) is None
