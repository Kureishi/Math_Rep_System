"""Chemistry: right against textbook values, audited independently of the method, honest about what it cannot decide."""
import math

import numpy as np
import pytest

from modules import chemistry as ch
from modules import mcmc
from modules.chemistry import ChemistryError


def species(text):
    return ch.parse_species(text)


# ------------------------------------------------------------------------------------------ formulas and masses

@pytest.mark.parametrize("formula, mass", [
    ("H2O", 18.015), ("CO2", 44.009), ("NaCl", 58.44), ("C6H12O6", 180.156), ("Ca(OH)2", 74.092), ("(NH4)2SO4", 132.14),
    ("CuSO4.5H2O", 249.685), ("Fe2(SO4)3", 399.86), ("K4[Fe(CN)6]", 368.35),
])
def test_molar_masses_match_the_reference_values(formula, mass):
    assert ch.molar_mass(species(formula)) == pytest.approx(mass, abs=0.02)


def test_nested_groups_hydrates_and_multipliers_are_counted():
    assert ch.parse_formula("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}
    assert ch.parse_formula("Al2(SO4)3") == {"Al": 2, "S": 3, "O": 12}
    assert ch.parse_formula("CuSO4·5H2O") == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert ch.parse_formula("K4[Fe(CN)6]") == {"K": 4, "Fe": 1, "C": 6, "N": 6}


@pytest.mark.parametrize("text, charge, atoms", [
    ("Fe^3+", 3, {"Fe": 1}), ("SO4^2-", -2, {"S": 1, "O": 4}), ("Na+", 1, {"Na": 1}), ("OH-", -1, {"O": 1, "H": 1}),
    ("H2PO4-", -1, {"H": 2, "P": 1, "O": 4}), ("Cu^{2+}", 2, {"Cu": 1}), ("NH4+", 1, {"N": 1, "H": 4}),
])
def test_charges_are_read_in_each_notation(text, charge, atoms):
    s = species(text)
    assert s.charge == charge and s.counts == atoms


def test_the_electron_state_symbols_and_a_leading_coefficient():
    assert species("e-").is_electron and species("e-").charge == -1
    assert species("H2O (l)").state == "l" and species("H2O (l)").counts == {"H": 2, "O": 1}
    assert species("2H2O").counts == {"H": 2, "O": 1}                                  # a coefficient already written is ignored
    assert species("NaCl(aq)").state == "aq"


@pytest.mark.parametrize("text, message", [
    ("Xx2", "Unknown element"), ("H2O)", "Unbalanced"), ("(H2O", "Unbalanced"), ("h2o", "capital"), ("", "empty"),
    ("Ca(OH", "Unbalanced"),
])
def test_bad_formulas_are_refused_with_a_reason(text, message):
    with pytest.raises(ChemistryError, match=message):
        species(text)


@pytest.mark.parametrize("text, message", [("H2 + O2 H2O", "No arrow"), ("A -> B -> C", "More than one"), ("-> H2O", "empty|at least one"),
                                            ("H2 + O2 ->", "empty|at least one")])
def test_bad_reactions_are_refused(text, message):
    with pytest.raises(ChemistryError, match=message):
        ch.parse_reaction(text)


def test_reactions_accept_every_arrow_and_plusses_without_spaces():
    for arrow in ("->", "→", "=>", "=", "<->", "<=>", "⇌", "⟶"):
        r, p = ch.parse_reaction(f"H2 + O2 {arrow} H2O")
        assert [s.text for s in r] == ["H2", "O2"] and [s.text for s in p] == ["H2O"]
    r, p = ch.parse_reaction("H2+O2->H2O")
    assert [s.text for s in r] == ["H2", "O2"]


def test_an_electron_in_a_half_reaction_is_a_species_in_its_own_right():
    r, p = ch.parse_reaction("MnO4^- + 8 H^+ + 5 e- -> Mn^2+ + 4 H2O")
    assert [s.text for s in r] == ["MnO4", "H", "e-"] and r[2].is_electron


def test_percent_composition_sums_to_one_hundred():
    comp = ch.percent_composition(species("C6H12O6"))
    assert sum(comp.values()) == pytest.approx(100.0) and comp["C"] == pytest.approx(40.0, abs=0.05)
    with pytest.raises(ChemistryError):
        ch.percent_composition(species("e-"))


# ------------------------------------------------------------------------------------------ balancing

@pytest.mark.parametrize("reaction, expected", [
    ("C3H8 + O2 -> CO2 + H2O", "C3H8 + 5 O2 → 3 CO2 + 4 H2O"),
    ("Fe + O2 -> Fe2O3", "4 Fe + 3 O2 → 2 Fe2O3"),
    ("Ca(OH)2 + H3PO4 -> Ca3(PO4)2 + H2O", "3 Ca(OH)2 + 2 H3PO4 → Ca3(PO4)2 + 6 H2O"),
    ("KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2", "2 KMnO4 + 16 HCl → 2 KCl + 2 MnCl2 + 8 H2O + 5 Cl2"),
    ("CuSO4.5H2O -> CuSO4 + H2O", "CuSO4.5H2O → CuSO4 + 5 H2O"),
    ("H2 + O2 -> H2O", "2 H2 + O2 → 2 H2O"),
    ("Al + HCl -> AlCl3 + H2", "2 Al + 6 HCl → 2 AlCl3 + 3 H2"),
])
def test_reactions_balance_to_the_smallest_whole_numbers(reaction, expected):
    r = ch.balance_text(reaction)
    assert r.status == "balanced" and r.equation() == expected and r.verified


def test_charge_is_conserved_in_redox_reactions():
    r = ch.balance_text("MnO4^- + Fe^2+ + H^+ -> Mn^2+ + Fe^3+ + H2O")
    assert r.equation() == "MnO4^- + 5 Fe^2+ + 8 H^+ → Mn^2+ + 5 Fe^3+ + 4 H2O"
    charge = next(row for row in r.ledger if row.item == "charge")
    assert charge.left == charge.right == 17


def test_the_ledger_is_recounted_from_the_formulas_not_taken_from_the_algebra():
    r = ch.balance_text("C3H8 + O2 -> CO2 + H2O")
    assert all(row.ok for row in r.ledger) and {row.item for row in r.ledger} == {"C", "H", "O"}
    left = ch._ledger(r.reactants, r.products, [1, 4, 3, 4])                 # a wrong answer must show up as unbalanced
    assert not all(row.ok for row in left)


def test_every_element_must_appear_on_both_sides():
    r = ch.balance_text("H2 + O2 -> H2O + NaCl")
    assert r.status == "unbalanceable" and r.coefficients is None and "NaCl" in r.message


def test_a_reaction_that_cannot_balance_as_written_says_which_species_may_be_misplaced():
    r = ch.balance_text("H2 + H2O -> O2")
    assert r.status == "unbalanceable" and "wrong side" in r.message


def test_a_reaction_with_several_independent_balances_is_not_pretended_to_be_unique():
    r = ch.balance_text("H2 + O2 + H2O2 -> H2O")
    assert r.status == "underdetermined" and "not unique" in r.message and len(r.basis) == 2
    assert r.coefficients is not None and r.verified and all(c > 0 for c in r.coefficients)


def test_nothing_balances_when_the_elements_do_not_match():
    assert ch.balance_text("H2 -> O2").status == "unbalanceable"


def test_the_latex_has_subscripts_charges_and_an_arrow():
    latex = ch.balance_text("Ca(OH)2 + H3PO4 -> Ca3(PO4)2 + H2O").latex()
    assert r"\longrightarrow" in latex and r"\mathrm{Ca(OH)_{2}}" in latex and r"\mathrm{H_{2}O}" in latex
    assert "^{3+}" in ch.species_latex(species("Fe^3+")) and "^{-}" in ch.species_latex(species("OH-"))


def test_coefficients_are_coprime_integers():
    from math import gcd
    from functools import reduce
    for rxn in ("C2H6 + O2 -> CO2 + H2O", "NH3 + O2 -> NO + H2O", "C6H12O6 + O2 -> CO2 + H2O"):
        c = ch.balance_text(rxn).coefficients
        assert all(isinstance(x, int) and x > 0 for x in c) and reduce(gcd, c) == 1


# ------------------------------------------------------------------------------------------ stoichiometry

def test_the_limiting_reagent_and_yield_follow_the_balanced_equation():
    r = ch.balance_text("H2 + O2 -> H2O")
    res = ch.stoichiometry(r, {0: (4.0, "g"), 1: (32.0, "g")})
    assert res.limiting == ["H2"]
    water = next(row for row in res.rows if row.label == "H2O")
    assert water.final_mol == pytest.approx(4.0 / 2.016, rel=1e-6) and water.final_g == pytest.approx(35.74, abs=0.02)
    o2 = next(row for row in res.rows if row.label == "O2")
    assert o2.final_mol > 0 and res.mass_conserved


def test_mass_is_conserved_exactly_for_any_balanced_reaction():
    for rxn, amounts in (("C3H8 + O2 -> CO2 + H2O", {0: (10.0, "g"), 1: (50.0, "g")}), ("Fe + O2 -> Fe2O3", {0: (5.0, "mol"), 1: (2.0, "mol")}),
                         ("KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2", {0: (3.0, "g"), 1: (20.0, "g")})):
        res = ch.stoichiometry(ch.balance_text(rxn), amounts)
        assert res.mass_conserved and res.mass_consumed_g == pytest.approx(res.mass_produced_g, rel=1e-12)


def test_stoichiometric_proportions_make_every_reactant_limiting_and_say_so():
    res = ch.stoichiometry(ch.balance_text("H2 + O2 -> H2O"), {0: (2.0, "mol"), 1: (1.0, "mol")})
    assert sorted(res.limiting) == ["H2", "O2"] and any("run out together" in n for n in res.notes)
    assert all(row.final_mol == pytest.approx(0.0, abs=1e-12) for row in res.rows if row.role == "reactant")


def test_a_reactant_without_an_amount_is_assumed_in_excess():
    res = ch.stoichiometry(ch.balance_text("H2 + O2 -> H2O"), {0: (1.0, "mol")})
    assert res.limiting == ["H2"] and any("in excess" in n for n in res.notes)
    o2 = next(row for row in res.rows if row.label == "O2")
    assert o2.initial_mol is None and o2.change_mol == pytest.approx(-0.5)


def test_percent_yield():
    assert ch.percent_yield(8.0, 10.0) == pytest.approx(80.0)
    with pytest.raises(ChemistryError):
        ch.percent_yield(1.0, 0.0)


@pytest.mark.parametrize("amounts, message", [({}, "at least one"), ({5: (1.0, "mol")}, "only be given for reactants"),
                                               ({0: (-1.0, "mol")}, "non-negative"), ({0: (1.0, "kg")}, "'mol' or 'g'"),
                                               ({0: (float("nan"), "mol")}, "non-negative")])
def test_bad_amounts_are_refused(amounts, message):
    with pytest.raises(ChemistryError, match=message):
        ch.stoichiometry(ch.balance_text("H2 + O2 -> H2O"), amounts)


def test_an_unbalanced_reaction_cannot_be_used_for_stoichiometry():
    with pytest.raises(ChemistryError, match="Balance the reaction first"):
        ch.stoichiometry(ch.balance_text("H2 -> O2"), {0: (1.0, "mol")})


# ------------------------------------------------------------------------------------------ pH

def test_strong_acids_and_bases_are_exact_even_when_water_matters():
    assert ch.ph_strong(0.01).ph == pytest.approx(2.0, abs=1e-6)
    assert ch.ph_strong(0.01, "base").ph == pytest.approx(12.0, abs=1e-6)
    dilute = ch.ph_strong(1e-8)
    assert dilute.ph == pytest.approx(6.978, abs=0.001) and dilute.approximation == pytest.approx(8.0)      # the shortcut says a base!


@pytest.mark.parametrize("c, pka", [(0.1, 4.76), (0.01, 4.76), (0.1, 2.0), (0.5, 3.1), (0.001, 3.75)])
def test_weak_acid_ph_matches_the_independent_quadratic_solution(c, pka):
    """For a monoprotic acid that is not extremely dilute, water's own ions are negligible and h solves h^2 + Ka h - Ka c = 0."""
    ka = 10.0 ** (-pka)
    h = (-ka + math.sqrt(ka * ka + 4 * ka * c)) / 2
    r = ch.ph_weak_acid(c, [pka])
    assert r.ph == pytest.approx(-math.log10(h), abs=2e-4) and r.residual < 1e-10


def test_known_textbook_values():
    assert ch.ph_weak_acid(0.1, [4.76]).ph == pytest.approx(2.883, abs=0.005)          # 0.1 M acetic acid
    assert ch.ph_weak_acid(0.01, [4.76]).ph == pytest.approx(3.389, abs=0.003)


def test_the_textbook_shortcut_fails_exactly_where_it_should():
    assert abs(ch.ph_weak_acid(0.1, [4.76]).approximation_error) < 0.005          # fine for a concentrated weak acid
    assert abs(ch.ph_weak_acid(1e-4, [4.76]).approximation_error) > 0.05           # off for a dilute one
    assert abs(ch.ph_weak_acid(0.1, [2.0]).approximation_error) > 0.05             # and for a fairly strong one (exact 1.568, shortcut 1.5)


def test_ph_falls_as_concentration_rises_and_rises_with_pka():
    phs = [ch.ph_weak_acid(c, [4.76]).ph for c in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
    assert all(a > b for a, b in zip(phs, phs[1:]))
    assert ch.ph_weak_acid(0.1, [3.0]).ph < ch.ph_weak_acid(0.1, [5.0]).ph


def test_a_very_dilute_acid_cannot_be_more_basic_than_water():
    assert 6.0 < ch.ph_weak_acid(1e-7, [4.76]).ph < 7.0


def test_polyprotic_acids_use_every_step():
    r = ch.ph_weak_acid(0.1, [2.15, 7.20, 12.35])
    assert r.ph == pytest.approx(1.63, abs=0.02) and r.residual < 1e-10


def test_weak_bases():
    r = ch.ph_weak_base(0.1, 4.75)
    assert r.ph == pytest.approx(11.12, abs=0.01) and r.residual < 1e-10
    assert ch.ph_weak_base(1e-6, 4.75).ph < 8.0 < ch.ph_weak_base(1e-6, 4.75).approximation        # the shortcut overshoots for a dilute base


def test_buffers_and_when_henderson_hasselbalch_holds():
    even = ch.buffer_ph(4.76, 0.1, 0.1)
    assert even.exact_ph == pytest.approx(4.76, abs=1e-3) and abs(even.difference) < 1e-3       # the exact solve includes the small [H+] term
    assert ch.buffer_ph(4.76, 0.01, 0.1).exact_ph == pytest.approx(5.76, abs=0.01)
    assert abs(ch.buffer_ph(3.0, 0.001, 0.001).difference) > 0.03                                   # a dilute, strong-ish acid: HH is off


@pytest.mark.parametrize("call, message", [(lambda: ch.ph_weak_acid(-1, [4.7]), "positive"), (lambda: ch.ph_weak_acid(0.1, []), "between 1 and 4"),
                                            (lambda: ch.ph_weak_acid(0.1, [7.2, 2.1]), "increase"), (lambda: ch.ph_strong(0.1, "salt"), "kind"),
                                            (lambda: ch.ph_weak_base(0, 4.7), "positive"), (lambda: ch.buffer_ph(4.7, 0, 0.1), "positive")])
def test_bad_ph_inputs_are_refused(call, message):
    with pytest.raises(ChemistryError, match=message):
        call()


def test_alpha_fractions_sum_to_one_and_cross_at_the_pka():
    h = np.array([1e-1, 1e-3, 10 ** -4.76, 1e-7, 1e-12])
    alphas = ch.alpha_fractions(h, [4.76])
    assert np.allclose(sum(alphas), 1.0)
    a = ch.alpha_fractions(10 ** -4.76, [4.76])
    assert float(a[0]) == pytest.approx(0.5) and float(a[1]) == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------ titration

def test_acetic_acid_titrated_with_sodium_hydroxide():
    c = ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1)
    (v_eq, ph_eq), = c.equivalence
    (v_half, ph_half, expected), = c.half_equivalence
    assert v_eq == pytest.approx(25.0) and ph_eq == pytest.approx(8.73, abs=0.03)
    assert v_half == pytest.approx(12.5) and ph_half == pytest.approx(4.76, abs=0.01) and expected == 4.76
    assert c.passed and {x.label for x in c.checks} >= {"Charge balance", "Physical trend", "Steepest point at equivalence"}


def test_a_strong_acid_reaches_exactly_seven_at_equivalence():
    c = ch.titration_curve("acid", [], 0.1, 25.0, 0.1, strong=True)
    assert c.equivalence[0][1] == pytest.approx(7.0, abs=0.005) and c.passed
    assert any(x.label == "Strong acid / strong base" and x.passed for x in c.checks)


def test_the_start_of_a_titration_agrees_with_the_independent_weak_acid_solver():
    """Two different code paths (the titration's diluted charge balance at V = 0, and ph_weak_acid) must give the same pH."""
    for c, pka in ((0.1, 4.76), (0.01, 3.2), (0.5, 6.4)):
        curve = ch.titration_curve("acid", [pka], c, 20.0, 0.1)
        assert curve.ph[0] == pytest.approx(ch.ph_weak_acid(c, [pka]).ph, abs=1e-9)


def test_a_weak_base_titrated_with_a_strong_acid():
    c = ch.titration_curve("base", [9.25], 0.1, 25.0, 0.1)
    assert c.equivalence[0][0] == pytest.approx(25.0) and c.equivalence[0][1] == pytest.approx(5.28, abs=0.03)
    assert c.half_equivalence[0][1] == pytest.approx(9.25, abs=0.01) and c.passed
    assert c.ph[0] == pytest.approx(ch.ph_weak_base(0.1, 14 - 9.25).ph, abs=1e-6) and c.ph[0] > c.ph[-1]


def test_a_diprotic_acid_has_two_equivalence_points_and_half_points_near_each_pka():
    c = ch.titration_curve("acid", [2.0, 7.0], 0.1, 25.0, 0.1)
    assert [round(v, 6) for v, _ in c.equivalence] == [25.0, 50.0]
    assert c.half_equivalence[1][1] == pytest.approx(7.0, abs=0.05) and c.passed


def test_the_half_equivalence_shortcut_is_reported_where_it_fails():
    c = ch.titration_curve("acid", [2.15, 7.20, 12.35], 0.1, 25.0, 0.1)
    deviations = [abs(p - e) for _, p, e in c.half_equivalence]
    assert deviations[1] < 0.05 and deviations[2] > 0.3


def test_equivalence_and_half_volumes_are_on_the_grid_and_the_range_is_respected():
    c = ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1, v_max_ml=40.0)
    assert 25.0 in c.volume_ml and 12.5 in c.volume_ml and c.volume_ml.max() <= 40.0 + 1e-9 and c.volume_ml.min() == 0.0


def test_after_a_large_excess_of_titrant_the_ph_approaches_that_of_the_strong_base():
    c = ch.titration_curve("acid", [4.76], 0.1, 25.0, 0.1, v_max_ml=200.0)
    excess = 0.1 * (200.0 - 25.0) / (225.0)
    assert c.ph[-1] == pytest.approx(14 + math.log10(excess), abs=0.01)


@pytest.mark.parametrize("kwargs, message", [
    (dict(kind="salt", pkas=[4.7], c_analyte=0.1, v_analyte_ml=25, c_titrant=0.1), "kind"),
    (dict(kind="acid", pkas=[4.7], c_analyte=0, v_analyte_ml=25, c_titrant=0.1), "positive"),
    (dict(kind="base", pkas=[4.7, 9.0], c_analyte=0.1, v_analyte_ml=25, c_titrant=0.1), "exactly one"),
    (dict(kind="acid", pkas=[7.0, 2.0], c_analyte=0.1, v_analyte_ml=25, c_titrant=0.1), "increase"),
    (dict(kind="acid", pkas=[4.7], c_analyte=0.1, v_analyte_ml=25, c_titrant=0.1, v_max_ml=-1), "positive"),
])
def test_bad_titrations_are_refused(kwargs, message):
    with pytest.raises(ChemistryError, match=message):
        ch.titration_curve(**kwargs)


# ------------------------------------------------------------------------------------------ kinetics

@pytest.mark.parametrize("order, k, c0", [(0, 0.02, 1.0), (1, 0.12, 1.0), (2, 0.35, 1.0)])
def test_the_reaction_order_and_rate_constant_are_recovered(order, k, c0):
    t, c = ch.demo_kinetics(order, k, c0, noise=0.004, n=16, seed=3)
    rep = ch.fit_rate_law(t, c)
    assert rep.best.order == order and rep.best.k == pytest.approx(k, rel=0.08) and rep.best.c0 == pytest.approx(c0, rel=0.05)
    assert not rep.ambiguous


def test_the_fit_reports_uncertainty_and_the_half_life():
    t, c = ch.demo_kinetics(1, 0.12, 1.0, noise=0.01, seed=1)
    best = ch.fit_rate_law(t, c).best
    assert best.k_se > 0 and abs(best.k - 0.12) < 3 * best.k_se
    assert best.half_life == pytest.approx(math.log(2) / best.k)
    assert ch.half_life(0, 1.0, 0.02) == pytest.approx(25.0) and ch.half_life(2, 2.0, 0.5) == pytest.approx(1.0)


def test_when_the_data_cannot_tell_two_orders_apart_it_says_so():
    t, c = ch.demo_kinetics(1, 0.12, 1.0, noise=0.02, n=10, t_max=2.0, seed=2)       # only ~25% conversion: first and second order look alike
    rep = ch.fit_rate_law(t, c)
    assert rep.ambiguous and any("cannot tell" in w for w in rep.warnings)


def test_a_mechanism_that_is_not_a_single_simple_order_leaves_a_residual_pattern_and_a_warning():
    t = np.linspace(0, 40, 40)
    c = 0.6 * np.exp(-1.0 * t) + 0.4 * np.exp(-0.05 * t)
    rep = ch.fit_rate_law(t, c + np.random.default_rng(0).normal(0, 0.002, t.shape))
    assert abs(rep.best.lag1_z) > 3 and any("pattern in the residuals" in w for w in rep.warnings)


def test_the_linearised_r_squared_is_reported_but_the_nonlinear_fit_decides():
    t, c = ch.demo_kinetics(1, 0.12, 1.0, noise=0.01, seed=4)
    rep = ch.fit_rate_law(t, c)
    assert all(f.r2_linearised is not None for f in rep.fits) and rep.fits == sorted(rep.fits, key=lambda f: f.bic)


def test_data_in_the_wrong_order_is_sorted_and_duplicate_times_are_refused():
    t, c = ch.demo_kinetics(1, 0.12, 1.0, noise=0.005, seed=5)
    shuffled = np.random.default_rng(0).permutation(len(t))
    assert ch.fit_rate_law(t[shuffled], c[shuffled]).best.order == 1
    with pytest.raises(ChemistryError, match="distinct"):
        ch.fit_rate_law([0, 1, 1, 2, 3, 4], [1, .9, .9, .8, .7, .6])


@pytest.mark.parametrize("t, c, message", [([0, 1, 2], [1, .9, .8], "at least 5"), ([0, 1, 2, 3, 4], [1, .9, .8], "same length"),
                                            ([0, 1, 2, 3, 4], [1, .9, -.1, .7, .6], "negative"), ([0, 1, 2, 3, float("nan")], [1, .9, .8, .7, .6], "non-finite")])
def test_bad_kinetics_data_is_refused(t, c, message):
    with pytest.raises(ChemistryError, match=message):
        ch.fit_rate_law(t, c)


def test_units_and_curves():
    assert ch.k_units(0) == "M/s" and ch.k_units(1, "mol/L", "min") == "1/min" and ch.k_units(2) == "1/(M·s)"
    assert ch.rate_curve(1, 2.0, 0.5, 0.0) == pytest.approx(2.0) and ch.rate_curve(0, 1.0, 0.5, 4.0) == 0.0       # zero order cannot go negative
    with pytest.raises(ChemistryError):
        ch.rate_curve(3, 1.0, 1.0, 1.0)


def test_each_rate_law_can_be_handed_to_the_posterior_sampler():
    """Cross-module: the expressions the kinetics page sends to MCMC must be accepted by it, and mean what they say."""
    for order in (0, 1, 2):
        expr, params, p0 = ch.posterior_model(order)
        model = mcmc.compile_model(expr, params)
        t = np.array([0.0, 1.0, 5.0])
        assert np.allclose(model(t, np.array([1.0, 0.1])), ch.rate_curve(order, 1.0, 0.1, t))
        assert len(p0) == len(params)


def test_the_arrhenius_parameters_are_recovered():
    T = np.array([280.0, 295.0, 310.0, 325.0, 340.0])
    k = 1e10 * np.exp(-50_000.0 / (ch.R_GAS * T))
    fit = ch.arrhenius_fit(T, k * (1 + np.random.default_rng(0).normal(0, 0.01, T.shape)))
    assert fit.ea_kj == pytest.approx(50.0, rel=0.03) and fit.r2 > 0.999 and fit.ea_se_kj > 0
    assert fit.predict_k(300.0) == pytest.approx(1e10 * math.exp(-50_000.0 / (ch.R_GAS * 300.0)), rel=0.1)


@pytest.mark.parametrize("T, k, message", [([300, 310], [1, 2], "at least 3"), ([300, 310, -5], [1, 2, 3], "positive"), ([300, 310, 320], [1, 0, 3], "positive")])
def test_bad_arrhenius_data_is_refused(T, k, message):
    with pytest.raises(ChemistryError, match=message):
        ch.arrhenius_fit(T, k)
