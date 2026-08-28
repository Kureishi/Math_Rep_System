import sympy as sp

from modules.equation_engine import build_model
from modules.similarity import canonicalize_equation, problem_shape, jaccard_similarity, find_similar_shapes


def _mk(vars_, eqs, sf, problem_type="algebraic", domain="x", **extra):
    payload = {"problem_domain": domain, "problem_type": problem_type,
               "variables": vars_, "equations": eqs, "solve_for": sf, "assumptions": []}
    payload.update(extra)
    return build_model(payload)


# ---------------------------------------------------------------- canonicalize_equation


def test_same_structure_different_names_canonicalizes_identically():
    a, vf, vi, t = sp.symbols("a v_f v_i t")
    x, y, z, w = sp.symbols("x y z w")
    eq1 = sp.Eq(a, (vf - vi) / t)
    eq2 = sp.Eq(x, (y - z) / w)
    assert canonicalize_equation(eq1) == canonicalize_equation(eq2)


def test_different_structure_canonicalizes_differently():
    a, vf, t = sp.symbols("a v_f t")
    eq1 = sp.Eq(a, (vf - 1) / t)
    eq2 = sp.Eq(a, vf / t)
    assert canonicalize_equation(eq1) != canonicalize_equation(eq2)


def test_numeric_coefficients_preserved_not_anonymized():
    a, t = sp.symbols("a t")
    eq1 = sp.Eq(a, sp.Rational(1, 2) * t ** 2)
    eq2 = sp.Eq(a, t ** 2)  # missing the 1/2 coefficient -- genuinely different formula
    assert canonicalize_equation(eq1) != canonicalize_equation(eq2)


def test_symbol_order_in_declaration_does_not_matter():
    a, b = sp.symbols("a b")
    x, y = sp.symbols("x y")
    eq1 = sp.Eq(a + b, 1)
    eq2 = sp.Eq(y + x, 1)  # same shape, symbols just declared/used in different objects
    assert canonicalize_equation(eq1) == canonicalize_equation(eq2)


# ---------------------------------------------------------------- problem_shape


def test_problem_shape_matches_across_domains_and_variable_names():
    m1 = _mk(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None}],
        [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""}],
        ["a"], domain="car kinematics",
    )
    m2 = _mk(
        [{"symbol": "r", "meaning": "r", "known_value": None, "unit": None},
         {"symbol": "p", "meaning": "p", "known_value": "100", "unit": None},
         {"symbol": "q", "meaning": "q", "known_value": "40", "unit": None},
         {"symbol": "s", "meaning": "s", "known_value": "5", "unit": None}],
        [{"name": "rate", "kind": "equation", "expression": "Eq(r, (p-q)/s)", "derivation": ""}],
        ["r"], domain="chemistry rate",
    )
    assert jaccard_similarity(problem_shape(m1), problem_shape(m2)) == 1.0


def test_problem_shape_differs_for_different_formula():
    m1 = _mk(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None}],
        [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""}],
        ["a"],
    )
    m2 = _mk(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None}],
        [{"name": "accel", "kind": "equation", "expression": "Eq(a, v_f/t)", "derivation": ""}],
        ["a"],
    )
    assert jaccard_similarity(problem_shape(m1), problem_shape(m2)) == 0.0


def test_problem_shape_empty_for_optimization_only_problem():
    m = _mk(
        [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        [],
        ["x"],
        objective={"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
    )
    assert problem_shape(m) == frozenset()


def test_problem_shape_partial_overlap_multi_equation():
    m1 = _mk(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
         {"symbol": "d", "meaning": "d", "known_value": None, "unit": None}],
        [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""},
            {"name": "disp", "kind": "equation", "expression": "Eq(d, v_f*t)", "derivation": ""},
        ],
        ["a", "d"],
    )
    m2 = _mk(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None}],
        [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""}],
        ["a"],
    )
    # m1 has 2 equation-shapes, m2 has 1, intersection is 1 -> jaccard = 1/2
    sim = jaccard_similarity(problem_shape(m1), problem_shape(m2))
    assert sim == 0.5


# ---------------------------------------------------------------- jaccard_similarity


def test_jaccard_identity():
    s = frozenset({"a", "b", "c"})
    assert jaccard_similarity(s, s) == 1.0


def test_jaccard_both_empty_returns_zero():
    assert jaccard_similarity(frozenset(), frozenset()) == 0.0


def test_jaccard_one_empty_returns_zero():
    assert jaccard_similarity(frozenset({"a"}), frozenset()) == 0.0


def test_jaccard_disjoint_sets():
    assert jaccard_similarity(frozenset({"a"}), frozenset({"b"})) == 0.0


# ---------------------------------------------------------------- find_similar_shapes


def test_find_similar_shapes_ranks_by_score():
    target = frozenset({"eq:x", "eq:y"})
    candidates = [
        (1, frozenset({"eq:x", "eq:y"})),        # perfect match
        (2, frozenset({"eq:x"})),                 # partial match (0.5)
        (3, frozenset({"eq:z"})),                  # no match
    ]
    ranked = find_similar_shapes(target, candidates, limit=5, min_similarity=0.0)
    assert [cid for cid, _ in ranked] == [1, 2, 3]
    assert ranked[0][1] == 1.0
    assert ranked[1][1] == 0.5
    assert ranked[2][1] == 0.0


def test_find_similar_shapes_respects_min_similarity():
    target = frozenset({"eq:x", "eq:y"})
    candidates = [(1, frozenset({"eq:x", "eq:y"})), (2, frozenset({"eq:z"}))]
    ranked = find_similar_shapes(target, candidates, min_similarity=0.5)
    assert [cid for cid, _ in ranked] == [1]


def test_find_similar_shapes_respects_limit():
    target = frozenset({"eq:x"})
    candidates = [(i, frozenset({"eq:x"})) for i in range(10)]
    ranked = find_similar_shapes(target, candidates, limit=3, min_similarity=0.0)
    assert len(ranked) == 3
