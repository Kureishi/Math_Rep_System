"""Tests for modules/sm_randommatrix.py: ensembles, semicircle, level-spacing statistics, Wishart."""
import math

import numpy as np
import pytest

from modules import sm_randommatrix as rm


@pytest.mark.parametrize("kind", rm.KINDS)
def test_every_check_passes_for_each_ensemble(kind):
    rep = rm.analyse(kind, 200, 40, seed=1)
    assert rep.checks.passed, [(c.label, c.detail) for c in rep.checks.checks if not c.passed]
    assert rep.nearest == kind and rep.seed == 1


def test_matrices_have_the_right_symmetry_and_variances():
    rng = np.random.default_rng(0)
    goe, gue = rm.sample_matrix("GOE", 300, rng), rm.sample_matrix("GUE", 300, rng)
    assert np.allclose(goe, goe.T) and np.allclose(gue, gue.conj().T) and np.iscomplexobj(gue) and not np.iscomplexobj(goe)
    off = ~np.eye(300, dtype=bool)
    assert goe[off].var() == pytest.approx(1 / 300, rel=0.05) and np.diag(goe).var() == pytest.approx(2 / 300, rel=0.35)
    assert (np.abs(gue[off]) ** 2).mean() == pytest.approx(1 / 300, rel=0.05)
    with pytest.raises(rm.StatMechError):
        rm.sample_matrix("GSE", 10, rng)


def test_second_moment_exact_finite_N_values():
    N = 60
    goe, _ = rm.spectra("GOE", N, 400, seed=2)
    gue, _ = rm.spectra("GUE", N, 400, seed=2)
    assert rm.empirical_moments(goe, (2,))[2][0] == pytest.approx(1 + 1 / N, abs=4 * rm.empirical_moments(goe, (2,))[2][1])
    assert rm.empirical_moments(gue, (2,))[2][0] == pytest.approx(1.0, abs=4 * rm.empirical_moments(gue, (2,))[2][1])


def test_semicircle_density_cdf_and_catalan_moments():
    x = np.linspace(-2, 2, 400001)
    pdf = rm.semicircle_pdf(x)
    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-8)
    for k, cat in rm.CATALAN.items():
        assert np.trapezoid(x ** k * pdf, x) == pytest.approx(cat, rel=1e-7)
    cdf = rm.semicircle_cdf(x)
    assert cdf[0] == pytest.approx(0, abs=1e-12) and cdf[-1] == pytest.approx(1) and cdf[len(x) // 2] == pytest.approx(0.5)
    assert np.all(np.diff(cdf) >= 0) and float(rm.semicircle_pdf(2.5)) == 0.0


def test_surmises_are_normalised_with_unit_mean_spacing():
    s = np.linspace(0, 40, 400001)
    for kind in rm.KINDS:
        p = rm.surmise_pdf(kind, s)
        assert np.trapezoid(p, s) == pytest.approx(1.0, abs=1e-8)
        assert np.trapezoid(s * p, s) == pytest.approx(1.0, abs=1e-8)             # the surmises are normalised to mean spacing 1
        c = rm.surmise_cdf(kind, s)
        assert c[0] == pytest.approx(0, abs=1e-9) and c[-1] == pytest.approx(1, abs=1e-6) and np.all(np.diff(c) >= -1e-12)
    # level repulsion: p(s) ~ s^beta near zero
    assert rm.surmise_pdf("GOE", np.array([0.01]))[0] < rm.surmise_pdf("Poisson", np.array([0.01]))[0]
    assert rm.surmise_pdf("GUE", np.array([0.01]))[0] < rm.surmise_pdf("GOE", np.array([0.01]))[0]
    with pytest.raises(rm.StatMechError):
        rm.surmise_pdf("GSE", np.array([1.0]))


def test_2x2_spacings_are_exactly_the_wigner_surmise():
    ck, seed = rm.verify_surmise_exact(150000, seed=5)
    assert ck.passed, [(c.label, c.detail) for c in ck.checks if not c.passed] and seed == 5
    with pytest.raises(rm.StatMechError):
        rm.two_by_two_spacings("Poisson", 10)


def test_classifier_separates_the_three_universality_classes_and_rejects_a_mislabel():
    for kind in rm.KINDS:
        eigs, _ = rm.spectra(kind, 200, 40, seed=7)
        best, dist = rm.classify(rm.unfolded_spacings(eigs))
        assert best == kind
        others = [d for k, d in dist.items() if k != kind]
        assert dist[kind] < min(others) / 2
    # GOE spectra are NOT classified as GUE: the distances differ decisively
    goe, _ = rm.spectra("GOE", 200, 40, seed=8)
    d = rm.spacing_distances(rm.unfolded_spacings(goe))
    assert d["GUE"] > 3 * d["GOE"]


def test_level_repulsion_shows_in_small_spacings():
    frac = {}
    for kind in rm.KINDS:
        s = rm.unfolded_spacings(rm.spectra(kind, 200, 40, seed=3)[0])
        frac[kind] = float(np.mean(s < 0.2))
    assert frac["Poisson"] > 0.15 > frac["GOE"] > frac["GUE"]


def test_ratio_statistic_values():
    assert rm.RATIO_LITERATURE["Poisson"] == pytest.approx(2 * math.log(2) - 1)
    vals = {k: rm.ratio_statistic(rm.spectra(k, 300, 30, seed=4)[0]) for k in rm.KINDS}
    assert vals["Poisson"] < vals["GOE"] < vals["GUE"]
    for k in rm.KINDS:
        assert vals[k] == pytest.approx(rm.RATIO_LITERATURE[k], abs=0.012)


def test_unfolding_gives_unit_mean_spacing_even_though_the_raw_density_varies():
    eigs, _ = rm.spectra("GUE", 300, 20, seed=1)
    s = rm.unfolded_spacings(eigs)
    assert s.mean() == pytest.approx(1.0, abs=1e-12)
    raw = np.diff(eigs, axis=1)
    assert raw[:, 140:160].mean() < raw[:, 15:30].mean()                          # the raw spacings are far from uniform: dense in the middle


def test_wishart_matches_marchenko_pastur():
    ck, eigs, seed = rm.verify_wishart(200, 400, 15, seed=6)
    assert ck.passed and seed == 6
    lo, hi = (1 - math.sqrt(0.5)) ** 2, (1 + math.sqrt(0.5)) ** 2
    assert eigs.min() > lo - 0.1 and eigs.max() < hi + 0.1
    x = np.linspace(lo, hi, 100001)
    assert np.trapezoid(rm.marchenko_pastur_pdf(x, 0.5), x) == pytest.approx(1.0, abs=1e-4)
    assert float(rm.marchenko_pastur_cdf(np.array([lo - 1]), 0.5)[0]) == 0.0 and float(rm.marchenko_pastur_cdf(np.array([hi + 1]), 0.5)[0]) == 1.0
    with pytest.raises(rm.StatMechError):
        rm.wishart_spectra(50, 20, 2)


def test_reproducibility_and_input_limits():
    a, sa = rm.spectra("GOE", 50, 5, seed=11)
    b, _ = rm.spectra("GOE", 50, 5, seed=11)
    assert np.array_equal(a, b) and sa == 11 and isinstance(rm.spectra("GOE", 20, 2)[1], int)
    for bad in (dict(kind="GSE", N=50, n_samples=2), dict(kind="GOE", N=2, n_samples=2), dict(kind="GOE", N=50, n_samples=0),
                dict(kind="GOE", N=5000, n_samples=100)):
        with pytest.raises(rm.StatMechError):
            rm.spectra(**bad)
