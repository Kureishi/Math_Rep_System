"""Tests for modules/gr_curvature.py: invariants against closed forms, identity checks, field equations, singularities."""
import pytest
import sympy as sp

from modules import gr_curvature as gc
from modules.gr_metrics import GRInputError, parse_user_metric, preset
from modules.tensor_calculus import analyze_metric

M, Q, a = (sp.Symbol(n, positive=True) for n in ("M", "Q", "a"))
r, th = sp.Symbol("r", positive=True), sp.Symbol("theta", real=True)


@pytest.fixture(scope="module")
def schw():
    return gc.compute_curvature(preset("schwarzschild"))


@pytest.fixture(scope="module")
def rn():
    return gc.compute_curvature(preset("reissner_nordstrom"))


@pytest.fixture(scope="module")
def kerr():
    return gc.compute_curvature(preset("kerr"))


# ---------------------------------------------------------------- simplifier
def test_simplify_expr_reduces_trig_identities_exactly():
    x = sp.Symbol("x", real=True)
    assert gc.simplify_expr(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1) == 0
    assert gc.simplify_expr((1 - sp.cos(x) ** 2) / sp.sin(x) ** 2) == 1
    assert gc.simplify_expr(sp.cosh(x) ** 2 - sp.sinh(x) ** 2 - 1) == 0
    assert gc.simplify_expr(sp.tan(x) * sp.cos(x) - sp.sin(x)) == 0


def test_simplify_expr_does_not_hide_nonzero_expressions():
    x = sp.Symbol("x", real=True)
    assert gc.simplify_expr(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 2) == -1
    assert gc.simplify_expr(sp.sin(x) - sp.cos(x)) != 0


def test_is_zero_distinguishes_exact_numeric_and_nonzero():
    x = sp.Symbol("x", real=True)
    assert gc.is_zero(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1) == (True, False)
    assert gc.is_zero(x - x + 1)[0] is False


# ---------------------------------------------------------------- Schwarzschild / RN / Kerr
def test_schwarzschild_is_ricci_flat_with_known_kretschmann(schw):
    assert schw.is_ricci_flat and not schw.is_flat
    assert sp.simplify(schw.invariants["Kretschmann R_abcd R^abcd"] - 48 * M ** 2 / r ** 6) == 0
    # in vacuum Weyl squared equals Kretschmann
    assert sp.simplify(schw.invariants["Weyl squared C_abcd C^abcd"] - 48 * M ** 2 / r ** 6) == 0


def test_all_identity_checks_pass_on_schwarzschild(schw):
    assert schw.checks.passed, [c for c in schw.checks.checks if not c.passed]
    assert len(schw.checks.checks) >= 7


def test_agrees_with_the_independent_tensor_calculus_module(schw):
    """Two different engines (this one and modules/tensor_calculus.py) must give the same Ricci tensor and scalar."""
    old = analyze_metric([[str(schw.g[i, j]) for j in range(4)] for i in range(4)], ["t", "r", "theta", "phi"])
    assert old.error is None
    assert sp.simplify(old.ricci_scalar - schw.ricci_scalar) == 0
    assert all(sp.simplify(old.ricci_tensor[i, j] - schw.ricci[i, j]) == 0 for i in range(4) for j in range(4))


def test_sign_convention_sphere_has_positive_curvature():
    m = parse_user_metric([["R^2", "0"], ["0", "R^2*sin(theta)^2"]], ["theta", "phi"])
    # not Lorentzian, but the engine still computes; only the signature check should fail
    c = gc.compute_curvature(m)
    assert sp.simplify(c.ricci_scalar - 2 / sp.Symbol("R", positive=True) ** 2) == 0
    failed = [ck.label for ck in c.checks.checks if not ck.passed]
    assert failed == ["Lorentzian signature at the sample point"]


def test_reissner_nordstrom_invariants_match_closed_forms(rn):
    inv = rn.invariants
    assert sp.simplify(inv["Ricci scalar R"]) == 0                          # traceless electromagnetic source
    assert sp.simplify(inv["Ricci squared R_ab R^ab"] - 4 * Q ** 4 / r ** 8) == 0
    kret = 8 * (6 * M ** 2 * r ** 2 - 12 * M * Q ** 2 * r + 7 * Q ** 4) / r ** 8
    assert sp.simplify(inv["Kretschmann R_abcd R^abcd"] - kret) == 0
    assert rn.checks.passed


def test_kerr_is_ricci_flat_with_known_kretschmann_and_passes_all_checks(kerr):
    assert kerr.is_ricci_flat
    c = sp.cos(th)
    known = 48 * M ** 2 * (r ** 2 - a ** 2 * c ** 2) * (r ** 4 - 14 * a ** 2 * r ** 2 * c ** 2 + a ** 4 * c ** 4) \
        / (r ** 2 + a ** 2 * c ** 2) ** 6
    assert gc.is_zero(kerr.invariants["Kretschmann R_abcd R^abcd"] - known, [kerr.sample_subs])[0]
    assert kerr.checks.passed, [c_ for c_ in kerr.checks.checks if not c_.passed]


def test_degenerate_metric_is_a_clear_error():
    m = parse_user_metric([["-1", "0"], ["0", "0"]], ["t", "x"])
    with pytest.raises(GRInputError, match="inverted|degenerate"):
        gc.compute_curvature(m)


def test_flat_metric_in_curvilinear_coordinates_is_flat():
    c = gc.compute_curvature(preset("minkowski_spherical"))
    assert c.is_flat and c.checks.passed


# ---------------------------------------------------------------- FRW and the fluid reading
def test_frw_effective_fluid_reproduces_the_friedmann_equations():
    m = preset("frw_power")
    c = gc.compute_curvature(m)
    fl = gc.perfect_fluid_decomposition(c)
    t, p = sp.Symbol("t", real=True), sp.Symbol("p", positive=True)
    H = p / t
    assert fl.isotropic
    assert sp.simplify(fl.density - 3 * H ** 2 / (8 * sp.pi)) == 0
    assert sp.simplify(fl.pressures[0] + (2 * sp.diff(H, t) + 3 * H ** 2) / (8 * sp.pi)) == 0


def test_de_sitter_is_vacuum_energy_with_lambda_3H2():
    c = gc.compute_curvature(preset("de_sitter"))
    H = sp.Symbol("H", positive=True)
    res = gc.verify_field_equations(c, None, 3 * H ** 2)
    assert res.satisfied
    assert not gc.verify_field_equations(c, None, 0).satisfied
    fl = gc.perfect_fluid_decomposition(c, 0)
    assert sp.simplify(fl.density - 3 * H ** 2 / (8 * sp.pi)) == 0 and sp.simplify(fl.density + fl.pressures[0]) == 0


def test_fluid_decomposition_refuses_a_non_diagonal_metric():
    c = gc.compute_curvature(preset("schwarzschild_ef"))
    with pytest.raises(GRInputError, match="diagonal"):
        gc.perfect_fluid_decomposition(c)


# ---------------------------------------------------------------- field equations
def test_vacuum_equations_hold_for_schwarzschild_and_fail_for_rn(schw, rn):
    assert gc.verify_field_equations(schw).satisfied
    bad = gc.verify_field_equations(rn)
    assert not bad.satisfied and bad.nonzero


def test_reissner_nordstrom_solves_einstein_maxwell(rn):
    T = gc.electromagnetic_stress_energy(rn, [["0", "Q/r^2", "0", "0"], ["-Q/r^2", "0", "0", "0"], ["0"] * 4, ["0"] * 4])
    res = gc.verify_field_equations(rn, T)
    assert res.satisfied, res.nonzero
    # the wrong charge is detected
    T2 = gc.electromagnetic_stress_energy(rn, [["0", "2*Q/r^2", "0", "0"], ["-2*Q/r^2", "0", "0", "0"], ["0"] * 4, ["0"] * 4])
    assert not gc.verify_field_equations(rn, T2).satisfied


def test_em_field_must_be_antisymmetric(rn):
    with pytest.raises(GRInputError, match="antisymmetric"):
        gc.electromagnetic_stress_energy(rn, [["0", "1", "0", "0"], ["1", "0", "0", "0"], ["0"] * 4, ["0"] * 4])


def test_tensor_grid_shape_is_validated():
    with pytest.raises(GRInputError, match="grid"):
        gc.parse_tensor_rows(preset("schwarzschild"), [["0", "0"]])


def test_effective_stress_energy_of_vacuum_is_zero(schw):
    assert gc.effective_stress_energy(schw) == sp.zeros(4, 4)


# ---------------------------------------------------------------- singularities
def _loci(curv):
    return {s.description: s.kind for s in gc.analyze_singularities(curv)}


def test_schwarzschild_r0_is_singular_r2M_is_not(schw):
    loci = _loci(schw)
    assert loci["r = 0"] == "curvature singularity"
    assert loci["r = 2"] == "coordinate artefact"
    assert loci["theta = 0"] == "coordinate artefact" and loci["theta = pi"] == "coordinate artefact"


def test_kerr_ring_singularity_and_horizons(kerr):
    loci = _loci(kerr)
    assert loci["r = 0 and theta = pi/2"] == "curvature singularity"
    horizons = [k for k in loci if k.startswith("r = ") and "sqrt" in k]
    assert len(horizons) == 2 and all(loci[h] == "coordinate artefact" for h in horizons)


def test_frw_big_bang_is_a_curvature_singularity():
    c = gc.compute_curvature(preset("frw_power"))
    assert _loci(c)["t = 0"] == "curvature singularity"


def test_singularity_analysis_respects_parameter_values(rn):
    # |Q| > M: no horizons, only the r = 0 singularity (naked)
    loci = {s.description: s.kind for s in gc.analyze_singularities(rn, {"M": 1.0, "Q": 2.0})}
    assert loci["r = 0"] == "curvature singularity"
    assert not [k for k in loci if "sqrt" in k]


def test_regular_metric_has_no_curvature_singularity():
    c = gc.compute_curvature(preset("minkowski"))
    assert gc.analyze_singularities(c) == []
