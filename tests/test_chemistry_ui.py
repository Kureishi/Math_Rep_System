"""The Chemistry page end to end: balance, stoichiometry, exact pH and titration, kinetics, and the handoff of a rate-law fit to the posterior sampler."""
import pytest

from modules import next_steps as ns
from tests.test_ui_smoke import _app

MODE = "🧪 Chemistry"


@pytest.fixture
def at():
    app = _app()
    app.run(timeout=120)
    app.session_state["app_mode"] = MODE
    app.run(timeout=120)
    assert not app.exception
    return app


def latex(app):
    return " ".join(m.value for m in app.latex)


def text(app):
    return " ".join(m.value for m in app.markdown)


def test_the_default_reaction_balances_with_its_recount(at):
    assert r"\longrightarrow" in latex(at) and r"5\,\mathrm{O_{2}}" in latex(at) and r"3\,\mathrm{CO_{2}}" in latex(at)
    assert "✅ Balanced" in text(at) and "every atom recounted from the formulas" in text(at)
    assert any(df.value.shape[0] == 3 for df in at.dataframe)                                   # C, H, O in the ledger


def test_choosing_an_example_fills_the_reaction_and_balances_it(at):
    at.selectbox(key="chem_example").set_value("Rusting of iron")
    at.run(timeout=120)
    assert at.text_input(key="chem_reaction").value == "Fe + O2 -> Fe2O3"
    assert r"4\,\mathrm{Fe}" in latex(at) and r"2\,\mathrm{Fe_{2}O_{3}}" in latex(at)


def test_a_redox_reaction_reports_that_charge_is_conserved(at):
    at.selectbox(key="chem_example").set_value("Permanganate and iron(II) (redox, with charge)")
    at.run(timeout=120)
    assert "charge conserved" in text(at) and not at.exception


@pytest.mark.parametrize("reaction, message", [("H2 + O2 H2O", "No arrow"), ("H2 + Xx -> H2Xx", "Unknown element"), ("H2 -> O2", "conserves")])
def test_a_bad_reaction_is_explained(at, reaction, message):
    at.text_input(key="chem_reaction").set_value(reaction)
    at.run(timeout=120)
    shown = " ".join(e.value for e in at.error) + " " + " ".join(w.value for w in at.warning)
    assert not at.exception and message in shown


def test_stoichiometry_finds_the_limiting_reagent_and_checks_mass(at):
    at.text_input(key="chem_stoich_reaction").set_value("H2 + O2 -> H2O")
    at.run(timeout=120)
    assert "limiting: O2" in text(at)                                  # the default 10 g of each: 10 g of O2 is the scarcer reactant
    at.number_input(key="chem_amt_0").set_value(4.0)
    at.number_input(key="chem_amt_1").set_value(32.0)
    at.run(timeout=120)
    assert not at.exception
    body = text(at)
    assert "limiting: H2" in body and "✅ mass conserved" in body


def test_the_exact_ph_of_a_strong_acid(at):
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["pH (exact)"] == "2.000" and "Textbook shortcut" in metrics


def test_weak_acid_ph_is_exact_and_the_shortcut_is_shown_beside_it(at):
    at.selectbox(key="chem_ph_kind").set_value("Weak acid")
    at.run(timeout=120)
    at.number_input(key="chem_ph_c").set_value(0.1)                    # the concentration box keeps what was typed for the previous kind
    at.run(timeout=120)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["pH (exact)"] == "2.883" and metrics["Textbook shortcut"] == "2.880"


def test_a_titration_curve_shows_its_checks(at):
    at.radio(key="chem_acid_panel").set_value("Titration curve")
    at.run(timeout=120)
    assert not at.exception
    body = text(at)
    assert "Charge balance" in body and "Steepest point at equivalence" in body and "✅ all checks pass" in body
    assert any("pKa" in df.value.columns for df in at.dataframe)                              # the half-equivalence table


def test_kinetics_names_the_order_and_offers_the_posterior(at):
    body = text(at)
    assert "first order" in body and "1/s" in body
    assert any(b.key == "ns_kin_posterior" for b in at.button)


def test_the_handoff_opens_the_posterior_page_with_the_data_and_model_already_filled_in(at):
    next(b for b in at.button if b.key == "ns_kin_posterior").click()
    at.run(timeout=180)
    assert not at.exception
    assert at.session_state["app_mode"] == ns.mode_for(ns.PAGE_POSTERIOR)
    assert at.session_state["post_expr"] == "C0*exp(-k*x)" and at.session_state["post_names"] == "C0, k"
    assert at.text_area(key="post_paste").value.startswith("x,y\n0,")
    assert at.text_input(key="post_expr").value == "C0*exp(-k*x)"                              # the page took them


def test_pasted_kinetics_data_is_analysed_and_bad_data_is_explained(at):
    at.radio(key="chem_kin_source").set_value("Paste data")
    at.run(timeout=120)
    at.text_area(key="chem_kin_paste").set_value("t,c\n0,1.0\n1,0.9\n2,0.8\n")
    at.run(timeout=120)
    assert not at.exception and any("at least 5" in e.value for e in at.error)
    at.text_area(key="chem_kin_paste").set_value("t,c\n0,1.0\n5,0.55\n10,0.30\n15,0.17\n20,0.09\n25,0.05\n30,0.03\n")
    at.run(timeout=120)
    assert not at.exception and "first order" in text(at)


def test_the_arrhenius_fit_reports_the_activation_energy(at):
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Activation energy"].endswith("kJ/mol") and float(metrics["R² of ln k against 1/T"]) > 0.99
