"""One lean end-to-end check of the Discover page, and that its figures build. The algorithm is tested in test_discover.py."""
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules import discover as dc
from modules import discover_plots as dp
from tests.test_ui_smoke import _app

MODE = "🧭 Discover equations"


def test_the_page_runs_a_demo_and_shows_the_equations_with_a_verdict():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    next(b for b in at.button if b.key == "discover_run").click()
    at.run(timeout=120)
    assert not at.exception
    latex = " ".join(m.value for m in at.latex)
    assert r"\dot{x}" in latex and r"\dot{v}" in latex
    assert any("Verified" in m.value for m in at.markdown)


def test_bad_pasted_data_is_explained_not_crashed():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    next(r for r in at.radio if r.key == "discover_source").set_value("Paste or upload data")
    at.run(timeout=120)
    next(t for t in at.text_area if t.key == "discover_paste").set_value("t,x\n0,1\n1,oops\n2,3\n")
    at.run(timeout=120)
    assert not at.exception and any("not a number" in e.value for e in at.error)


@pytest.fixture(scope="module")
def result():
    t, X, names, _ = dc.demo_data("Damped oscillator", noise=0.005, seed=3)
    return dc.fit(t, X, names, degree=2, bootstrap=10)


def test_the_figures_build_with_the_expected_traces(result):
    sim = dc.simulate(result, result.X[0], result.t[:400])
    fit = dp.fit_figure(result, sim)
    assert len(fit.data) == 2 * len(result.names)
    assert len(dp.path_figure(result).data) == 2
    inc = dp.inclusion_figure(result)
    assert len(inc.data[0].x) == result.n_terms and all(0 <= v <= 1 for v in inc.data[0].x)


def test_a_diverged_run_still_plots(result):
    sim = np.full((50, len(result.names)), np.nan)
    assert dp.fit_figure(result, sim) is not None
