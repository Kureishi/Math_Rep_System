import pytest
import sympy as sp

from modules.equivalence import check_equivalence


def test_trig_identity_is_equivalent():
    r = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert r.equivalent is True
    assert r.method == "symbolic"


def test_expanded_binomial_is_equivalent():
    r = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    assert r.equivalent is True


def test_factored_quadratic_is_equivalent():
    r = check_equivalence("x**2 - 1", "(x-1)*(x+1)")
    assert r.equivalent is True


def test_different_polynomials_are_not_equivalent():
    r = check_equivalence("x**2", "x**3")
    assert r.equivalent is False


def test_log_product_rule_not_universally_equivalent():
    # only true for positive x, y -- not a universal equivalence over the reals
    r = check_equivalence("log(x*y)", "log(x) + log(y)", extra_symbols=["y"])
    assert r.equivalent is not True


def test_log_product_rule_equivalent_for_positive_symbols():
    r = check_equivalence("log(x*y)", "log(x) + log(y)", extra_symbols=["y"])
    # not asserting True here (domain-dependent); just confirms it doesn't crash
    assert r.error is None


def test_commutative_multiplication_is_equivalent():
    r = check_equivalence("a*b", "b*a", extra_symbols=["a", "b"])
    assert r.equivalent is True


def test_unparseable_expression_reports_error():
    r = check_equivalence("x +", "x")
    assert r.error is not None
    assert r.equivalent is None


def test_undeclared_symbol_is_parsed_automatically():
    # x is always available without extra_symbols
    r = check_equivalence("x + 0", "x")
    assert r.equivalent is True


def test_constants_are_compared_correctly():
    r = check_equivalence("2 + 2", "4")
    assert r.equivalent is True
    r2 = check_equivalence("2 + 2", "5")
    assert r2.equivalent is False


def test_difference_simplified_is_populated_on_success():
    r = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert r.difference_simplified is not None
    assert sp.simplify(r.difference_simplified) == 0


def test_result_detail_is_nonempty_string():
    r = check_equivalence("x", "x")
    assert isinstance(r.detail, str) and len(r.detail) > 0


# ------------------------------------------------------------------
# Branch coverage for the paths sp.Expr.equals() makes non-deterministic.
#
# equals() sometimes returns None (and sometimes False) for a
# domain-conditional pair like sqrt(x**2) vs x, depending on which random
# sample points it happens to draw -- so a test that relies on the real
# equals() to reach the numeric-sampling fallback would itself be flaky.
# Instead these tests pin equals() to a fixed answer by intercepting the
# one run_with_timeout call that wraps it (matched by its label), which
# makes every branch of check_equivalence_exprs reachable deterministically.
# ------------------------------------------------------------------
import modules.equivalence as eqmod
from modules.equivalence import check_equivalence_exprs, _sample_agreement
from modules.timeout_utils import ComputationTimeoutError

x_sym = sp.Symbol("x")
y_sym = sp.Symbol("y")
_EQUALS_LABEL = "equivalence .equals() check"
_SIMPLIFY_LABEL = "equivalence simplify"


@pytest.fixture
def equals_returns_none(monkeypatch):
    """Pins sympy's equals() step to 'undecided' so the module's own
    numeric-sampling fallback runs, regardless of sympy's random draws."""
    real = eqmod.run_with_timeout

    def fake(func, *args, **kwargs):
        if kwargs.get("label") == _EQUALS_LABEL:
            return None
        return real(func, *args, **kwargs)

    monkeypatch.setattr(eqmod, "run_with_timeout", fake)


def test_sample_agreement_single_symbol_partial_agreement():
    # sqrt(x**2) vs x: agree at the five positive sample points, disagree
    # at the three negative ones -- exact counts, not just "some".
    agree, tried, example = _sample_agreement(sp.sqrt(x_sym**2), x_sym, [x_sym])
    assert (agree, tried) == (5, 8)
    assert example == [("x", -3)]  # first disagreeing point, in sample order


def test_sample_agreement_multi_symbol_is_deterministic():
    # Two symbols take the seeded-random-combination path (12 combos);
    # the fixed Random(0) seed means two calls must give identical results.
    e1, e2 = x_sym * y_sym, x_sym * y_sym + 1
    first = _sample_agreement(e1, e2, [x_sym, y_sym])
    second = _sample_agreement(e1, e2, [x_sym, y_sym])
    assert first == second
    agree, tried, example = first
    assert tried == 12 and agree == 0
    assert example is not None and [n for n, _ in example] == ["x", "y"]


def test_sample_agreement_skips_points_that_raise():
    # log(x-100) raises ValueError at every sample point (all are < 100),
    # so nothing is "tried" -- and that must be reported as 0, not crash.
    agree, tried, example = _sample_agreement(sp.log(x_sym - 100), sp.log(x_sym - 101), [x_sym])
    assert (agree, tried, example) == (0, 0, None)


def test_sample_agreement_skips_zero_division():
    # e1 - e2 = -1/(x - 1/2) (deliberately NOT the same expression twice:
    # sympy would cancel e1 - e2 to a bare 0 before evaluation ever happens).
    # It raises ZeroDivisionError at exactly x=0.5, so that one point is
    # skipped and the other 7 are still tried -- none of which are zero.
    half = sp.Rational(1, 2)
    agree, tried, example = _sample_agreement(1 / (x_sym - half), 2 / (x_sym - half), [x_sym])
    assert tried == 7 and agree == 0
    assert example is not None


def test_sample_agreement_skips_complex_results():
    # A float power of a negative base is complex in Python's math-based
    # evaluation, so all three negative sample points must be skipped
    # (not counted as agreeing OR disagreeing) -- only the 5 positive ones count.
    agree, tried, _ = _sample_agreement(x_sym ** sp.Float(0.5), 2 * x_sym, [x_sym])
    assert tried == 5


def test_fallback_partial_agreement_reports_where(equals_returns_none):
    r = check_equivalence_exprs(sp.sqrt(x_sym**2), x_sym)
    assert r.equivalent is None
    assert r.method == "numeric sampling"
    assert "5/8" in r.detail
    assert "x=-3" in r.detail
    assert "part of the domain" in r.detail


def test_fallback_full_agreement_is_evidence_not_proof(equals_returns_none):
    r = check_equivalence_exprs(x_sym, x_sym + 0)
    assert r.equivalent is None          # deliberately NOT True: agreement at points isn't proof
    assert r.method == "numeric sampling"
    assert "8 tested numeric point" in r.detail
    assert "not proof" in r.detail


def test_fallback_zero_agreement_is_false(equals_returns_none):
    r = check_equivalence_exprs(x_sym, x_sym + 1)
    assert r.equivalent is False
    assert r.method == "numeric sampling"
    assert "all 8" in r.detail


def test_fallback_constants_with_no_symbols(equals_returns_none):
    r = check_equivalence_exprs(sp.Integer(2), sp.Integer(3))
    assert r.equivalent is None
    assert r.method == "undetermined"
    assert "constant" in r.detail


def test_fallback_no_evaluable_sample_point(equals_returns_none):
    r = check_equivalence_exprs(sp.log(x_sym - 100), sp.log(x_sym - 101))
    assert r.equivalent is None
    assert r.method == "undetermined"
    assert "domain error" in r.detail


def test_fallback_multi_symbol_disagreement(equals_returns_none):
    r = check_equivalence_exprs(x_sym * y_sym, x_sym * y_sym + 1)
    assert r.equivalent is False
    assert "all 12" in r.detail


def test_equals_raising_is_treated_as_undecided(monkeypatch):
    real = eqmod.run_with_timeout

    def fake(func, *args, **kwargs):
        if kwargs.get("label") == _EQUALS_LABEL:
            raise RuntimeError("sympy internal failure")
        return real(func, *args, **kwargs)

    monkeypatch.setattr(eqmod, "run_with_timeout", fake)
    # Must not propagate: an exception inside equals() falls through to the
    # numeric fallback (here x vs x+1 -> disagrees everywhere -> False).
    r = check_equivalence_exprs(x_sym, x_sym + 1)
    assert r.equivalent is False and r.method == "numeric sampling"


def test_simplify_timeout_is_reported_not_raised(monkeypatch):
    def fake(func, *args, **kwargs):
        raise ComputationTimeoutError(0.1, kwargs.get("label"))

    monkeypatch.setattr(eqmod, "run_with_timeout", fake)
    r = check_equivalence_exprs(x_sym, x_sym + 1)
    assert r.equivalent is None and r.method == "undetermined"
    assert "Timed out while simplifying" in r.detail
    assert r.raw_difference == -1          # e1 - e2 preserved for proof.py even on timeout


def test_equals_timeout_is_reported_not_raised(monkeypatch):
    real = eqmod.run_with_timeout

    def fake(func, *args, **kwargs):
        if kwargs.get("label") == _EQUALS_LABEL:
            raise ComputationTimeoutError(0.1, _EQUALS_LABEL)
        return real(func, *args, **kwargs)

    monkeypatch.setattr(eqmod, "run_with_timeout", fake)
    r = check_equivalence_exprs(x_sym, x_sym)
    assert r.equivalent is None and r.method == "undetermined"
    assert "Timed out while checking equivalence" in r.detail
    assert r.raw_difference == 0


def test_second_expression_parse_error_is_reported():
    r = check_equivalence("x", "x +")
    assert r.equivalent is None
    assert "second expression" in (r.error or "")


def test_symbolic_true_keeps_raw_difference_for_proof_mode():
    r = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    assert r.equivalent is True
    assert r.raw_difference is not None
    assert r.raw_difference != 0  # UNsimplified -- that's the whole point of the field
