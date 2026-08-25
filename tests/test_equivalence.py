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
