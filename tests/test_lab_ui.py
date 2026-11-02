"""The Numerics & verification lab, driven through the real app; and the automatic conservation monitor on the ODE results page."""
from streamlit.testing.v1 import AppTest

from modules.command_palette import MODE_LABELS, search
from tests.test_ui_smoke import TIMEOUT, _app, _exceptions, isolated_data  # noqa: F401

MODE = "🧰 Numerics & verification lab"


def page():
    at = _app()
    at.session_state["app_mode"] = MODE
    return at.run()


def press(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


def widget(at, kind, key):
    return next(w for w in getattr(at, kind) if w.key == key)


def successes(at):
    return [e.value for e in at.success]


def csv(at):
    return " ".join(df.value.to_csv() for df in at.dataframe)


def test_registered_and_landing_renders_all_seven_tabs():
    assert MODE in MODE_LABELS
    for q in ("symplectic", "noether", "nondimensionalize", "reproducible", "cross-validation", "convergence"):
        assert search(q)[0] == MODE, q
    at = page()
    assert not _exceptions(at), _exceptions(at)
    assert len(at.tabs) >= 7
    assert "Pi1" in " ".join(l.value for l in at.latex) or "Pi_1" in " ".join(l.value for l in at.latex)        # nondimensionalise tab is live
    assert any("checks passed" in s for s in successes(at))                                                      # limits tab is live
    widget(at, "text_input", "lab_nd_eq_damped, driven oscillator").set_value("m*x_tt + k*x_t = F").run()
    assert not _exceptions(at) and any("not dimensionally homogeneous" in e.value for e in at.error)


def test_invariants_symplectic_comparison_and_monitors_for_existing_simulations():
    at = press(page(), "lab_inv_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at)) and "bounded oscillation" in csv(at)
    widget(at, "text_area", "lab_inv_rhs_harmonic oscillator").set_value("a*y\n-x").run()
    at = press(at, "lab_inv_run")
    assert not _exceptions(at) and any("numbers" in e.value for e in at.error)                                   # unresolved parameter: refused politely
    widget(at, "selectbox", "lab_inv_preset").set_value("damped oscillator (no invariant)").run()
    at = press(at, "lab_inv_run")
    assert not _exceptions(at) and any("No conserved quantity" in w.value for w in at.warning)
    at = press(at, "lab_sym_run")
    assert not _exceptions(at), _exceptions(at)
    table = csv(at)
    assert "bounded oscillation" in table and "secular drift" in table and "leapfrog" in table
    at = press(at, "lab_ex_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))


def test_convergence_hamiltonian_and_cross_validation_tabs():
    at = press(page(), "lab_cv_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at)) and "pairwise order" in csv(at)
    widget(at, "selectbox", "lab_h_sys").set_value("central").run()
    assert not _exceptions(at), _exceptions(at)
    assert "rotation" in csv(at) and "energy" in csv(at)
    widget(at, "selectbox", "lab_h_sys").set_value("henon_heiles").run()
    at = press(at, "lab_h_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "selectbox", "lab_h_sys").set_value("custom").run()
    widget(at, "text_input", "lab_h_text").set_value("x*xdot").run()
    assert not _exceptions(at) and any("singular" in e.value for e in at.error)
    widget(at, "selectbox", "lab_cx_task").set_value("demonstration set (all of them)").run()
    at = press(at, "lab_cx_suite")
    assert not _exceptions(at), _exceptions(at)
    assert "numeric only" in csv(at) and "agree" in csv(at) and "DISAGREE" not in csv(at)


def test_notebook_export_self_verification_and_bad_parameters():
    at = press(page(), "lab_nb_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("Run recorded" in s for s in successes(at))
    at = press(at, "lab_nb_verify")
    assert not _exceptions(at) and any("reproduced" in s for s in successes(at))
    widget(at, "text_area", "lab_nb_params_partition").set_value("{\"gap\": ").run()
    assert not _exceptions(at) and any("JSON" in e.value for e in at.error)
    widget(at, "text_area", "lab_nb_params_partition").set_value('{"nope": 1}').run()
    at = press(at, "lab_nb_run")
    assert not _exceptions(at) and any("Unknown parameter" in e.value for e in at.error)


# ---------------------------------------------------------------- the automatic monitor on the existing ODE page
def _ode_monitor_script(which):
    """The ODE results section for an undamped (conservative) or damped coupled system."""
    import streamlit as st
    from modules.workspace import Workspace
    from tests.test_ode_trajectories import _model, spiral_model
    from ui.results.solutions import render_ode_solution
    for key, default in [("plot_snapshots", {}), ("pdf_bytes", None), ("problem_text", "")]:
        st.session_state.setdefault(key, default)
    if which == "undamped":
        model = _model([("xdot", "Eq(Derivative(x(t), t), y(t))"), ("ydot", "Eq(Derivative(y(t), t), -x(t))")], ["x", "y"], [("x(0)", 1.0), ("y(0)", 0.0)])
    else:
        model = spiral_model()
    render_ode_solution(Workspace(st.session_state), model)


def test_ode_page_finds_and_verifies_the_conserved_quantity_of_an_undamped_system():
    at = AppTest.from_function(_ode_monitor_script, args=("undamped",), default_timeout=TIMEOUT).run()
    assert not _exceptions(at), _exceptions(at)
    labels = [e.label for e in at.expander]
    assert any("Conserved quantities found automatically" in l for l in labels), labels
    assert any("Q_{1}" in l.value or "x^{2}" in l.value for l in at.latex)
    assert any("bounded oscillation" in w.value or "constant" in w.value for w in list(at.success) + list(at.warning))


def test_ode_page_says_plainly_when_a_dissipative_system_has_nothing_conserved():
    at = AppTest.from_function(_ode_monitor_script, args=("damped",), default_timeout=TIMEOUT).run()
    assert not _exceptions(at), _exceptions(at)
    assert any("Conserved quantities found automatically" in e.label for e in at.expander)
    assert any("no conserved" in c.value.lower() for c in at.caption)
