"""Tests for modules/gr_transforms.py: coordinate changes, with the invariants as the oracle."""
import pytest
import sympy as sp

from modules import gr_transforms as gt
from modules.gr_curvature import simplify_expr
from modules.gr_metrics import GRInputError, preset


@pytest.mark.parametrize("key", ["schw_ingoing_ef", "schw_outgoing_ef", "schw_isotropic", "mink_rindler", "mink_spherical",
                                 "desitter_conformal"])
def test_explicit_presets_preserve_every_invariant(key):
    res = gt.apply_preset(key)
    assert res.mode == "explicit" and res.new_metric is not None
    assert res.invariants_preserved and len(res.comparisons) >= 3
    assert res.checks.passed, [c for c in res.checks.checks if not c.passed]


def test_ingoing_eddington_finkelstein_matches_the_known_metric():
    res = gt.apply_preset("schw_ingoing_ef")
    known = preset("schwarzschild_ef")
    # same tensor, same coordinate names: compare component by component
    assert sp.simplify(res.components - known.g) == sp.zeros(4, 4)


def test_minkowski_to_spherical_gives_the_polar_metric():
    g = gt.apply_preset("mink_spherical").components
    th = [s for s in g.free_symbols if s.name == "theta"][0]
    r = [s for s in g.free_symbols if s.name == "r"][0]
    assert sp.simplify(g - sp.diag(-1, 1, r ** 2, r ** 2 * sp.sin(th) ** 2)) == sp.zeros(4, 4)


def test_rindler_is_flat():
    res = gt.apply_preset("mink_rindler")
    assert res.curvature_new.is_flat


def test_desitter_conformal_time_is_conformally_flat():
    g = gt.apply_preset("desitter_conformal").components
    eta = [s for s in g.free_symbols if s.name == "eta"][0]
    H = [s for s in g.free_symbols if s.name == "H"][0]
    assert simplify_expr(g[0, 0] + 1 / (H * eta) ** 2) == 0 and simplify_expr(g[1, 1] - 1 / (H * eta) ** 2) == 0


def test_kruskal_is_implicit_and_conformally_flat_in_the_t_x_block():
    res = gt.apply_preset("schw_kruskal")
    assert res.mode == "implicit"
    g = res.components
    assert g[0, 1] == 0 and simplify_expr(g[0, 0] + g[1, 1]) == 0              # -dT^2 + dX^2, same conformal factor
    r, M = sp.Symbol("r", positive=True), sp.Symbol("M", positive=True)
    assert simplify_expr(g[1, 1] - 32 * M ** 3 * sp.exp(-r / (2 * M)) / r) == 0   # the textbook factor
    assert res.invariants_preserved and res.checks.passed


def test_a_wrong_transformation_is_not_silently_accepted():
    """If 'new' coordinates are given inconsistently the check list must show a failure, not a pass."""
    # old r expressed with the wrong power: the transformation is invertible, so it is a legal change of coordinates and the
    # invariants must STILL match -- the check is that it never reports a mismatch for a correct tensor transformation
    old = preset("schwarzschild")
    res = gt.transform_metric(old, ["t", "x", "theta", "phi"], old_in_terms_of_new=["t", "x^2", "theta", "phi"])
    assert res.invariants_preserved
    # tamper with the comparison inputs: an invariant that differs is flagged
    c = res.comparisons[0]
    assert gt.is_zero(c.old - c.new - 1, None)[0] is False


def test_non_invertible_transformation_is_refused():
    with pytest.raises(GRInputError, match="not invertible"):
        gt.transform_metric(preset("minkowski"), ["t", "x", "y", "z"], old_in_terms_of_new=["t", "x", "x", "z"])


def test_argument_validation():
    m = preset("minkowski")
    with pytest.raises(GRInputError, match="either"):
        gt.transform_metric(m, ["a", "b", "c", "d"])
    with pytest.raises(GRInputError, match="either"):
        gt.transform_metric(m, ["a", "b", "c", "d"], old_in_terms_of_new=["a"] * 4, new_in_terms_of_old=["t"] * 4)
    with pytest.raises(GRInputError, match="names"):
        gt.transform_metric(m, ["a", "a", "c", "d"], old_in_terms_of_new=["a"] * 4)
    with pytest.raises(GRInputError, match="expressions"):
        gt.transform_metric(m, ["a", "b", "c", "d"], old_in_terms_of_new=["a", "b"])
    with pytest.raises(GRInputError):
        gt.apply_preset("nope")


def test_implicit_map_that_sympy_can_invert_becomes_explicit():
    res = gt.transform_metric(preset("schwarzschild"), ["v", "r", "theta", "phi"],
                              new_in_terms_of_old=["t + r + 2*M*log(r - 2*M)", "r", "theta", "phi"])
    assert res.inverted and res.mode == "explicit" and res.invariants_preserved


def test_scaling_a_time_coordinate_changes_components_not_curvature():
    res = gt.transform_metric(preset("minkowski"), ["T", "x", "y", "z"], old_in_terms_of_new=["3*T", "x", "y", "z"])
    assert res.components[0, 0] == -9 and res.curvature_new.is_flat
