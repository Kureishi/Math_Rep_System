"""The chemistry figures build, mark what they should, and describe themselves."""
import numpy as np
import pytest

from modules import chemistry as ch
from modules import chemistry_plots as cp
from modules import palette


def test_the_titration_figure_marks_equivalence_and_half_equivalence_against_the_pka():
    fig = cp.titration_figure(ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1))
    assert [t.name for t in fig.data] == ["pH", "equivalence point", "half-equivalence"]
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == pytest.approx(4.76)          # the pKa guide
    assert list(fig.layout.yaxis.range) == [0, 14]


def test_a_strong_acid_titration_has_no_half_equivalence_or_pka_guide():
    fig = cp.titration_figure(ch.titration_curve("acid", [], 0.1, 25.0, 0.1, strong=True))
    assert [t.name for t in fig.data] == ["pH", "equivalence point"] and len(fig.layout.shapes) == 0


def test_the_distribution_figure_has_one_curve_per_form_that_sum_to_one_and_a_guide_per_pka():
    fig = cp.distribution_figure([2.15, 7.2, 12.35], marker_ph=7.0)
    assert len(fig.data) == 4 and len(fig.layout.shapes) == 4
    assert np.allclose(sum(np.asarray(t.y) for t in fig.data), 1.0)
    assert len(cp.distribution_figure([4.76]).data) == 2


def test_the_kinetics_figures_show_every_order_the_data_and_the_residuals():
    t, c = ch.demo_kinetics(1, 0.12, 1.0, seed=2)
    rep = ch.fit_rate_law(t, c)
    fig = cp.kinetics_figure(rep)
    names = [x.name for x in fig.data]
    assert "data" in names and "residual (best fit)" in names and any(n.endswith("(best)") for n in names) and len(fig.data) == 5
    lin = cp.linearised_figure(rep)
    assert len(lin.data) == 6 and all("R² =" in a.text for a in lin.layout.annotations)


def test_the_other_figures_build():
    T = np.array([290.0, 300.0, 310.0, 320.0])
    fit = ch.arrhenius_fit(T, 1e9 * np.exp(-60_000 / (ch.R_GAS * T)))
    assert len(cp.arrhenius_figure(fit).data) == 2
    res = ch.stoichiometry(ch.balance_text("H2 + O2 -> H2O"), {0: (4.0, "g"), 1: (32.0, "g")})
    assert len(cp.stoichiometry_figure(res).data) == 2


def test_the_figures_describe_themselves_for_screen_readers():
    d = palette.describe_figure(cp.titration_figure(ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1)))
    assert "Titration curve" in d and "equivalence point" in d and "x axis: titrant added (mL)" in d


def test_no_figure_uses_pure_black_so_all_survive_the_dark_theme():
    import json
    figs = [cp.titration_figure(ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1)), cp.distribution_figure([4.76]),
            cp.kinetics_figure(ch.fit_rate_law(*ch.demo_kinetics(1, 0.12, 1.0, seed=2)))]
    for f in figs:
        assert '"black"' not in json.dumps(f.to_plotly_json()) and '#000000' not in json.dumps(f.to_plotly_json())
