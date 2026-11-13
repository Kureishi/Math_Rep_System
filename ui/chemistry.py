"""The Chemistry mode: balancing, stoichiometry, acids and titration, kinetics (modules/chemistry.py). No language model.

Every result is shown with the check that backs it (an atom-and-charge recount, mass conservation, the charge-balance residual, ...)."""
import numpy as np
import streamlit as st

from modules import chemistry as ch
from modules import chemistry_plots as cp
from modules import discover
from modules import next_steps as ns
from ui.a11y import show_figure
from ui.next_steps import render_next_steps
from ui.theme import badge_row

EXAMPLES = {
    "(type your own)": "",
    "Propane burning": "C3H8 + O2 -> CO2 + H2O",
    "Rusting of iron": "Fe + O2 -> Fe2O3",
    "Calcium phosphate": "Ca(OH)2 + H3PO4 -> Ca3(PO4)2 + H2O",
    "Permanganate and iron(II) (redox, with charge)": "MnO4^- + Fe^2+ + H^+ -> Mn^2+ + Fe^3+ + H2O",
    "Permanganate and hydrochloric acid": "KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2",
    "Copper sulfate hydrate": "CuSO4.5H2O -> CuSO4 + H2O",
}
_NOTATION = "Separate species with ` + ` (spaces around the plus). Charges: `Fe^3+`, `SO4^2-`, or `Na+` / `OH-`. Electron: `e-`. Hydrates: `CuSO4.5H2O`."


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]


def _pick_example() -> None:
    chosen = EXAMPLES.get(st.session_state.get("chem_example", ""), "")
    if chosen:
        st.session_state["chem_reaction"] = chosen


# ------------------------------------------------------------------------------------------ balance

def _balance_tab() -> None:
    st.selectbox("Examples", list(EXAMPLES), key="chem_example", on_change=_pick_example)
    text = st.text_input("Reaction", "C3H8 + O2 -> CO2 + H2O", key="chem_reaction", help=_NOTATION)
    st.caption(_NOTATION)
    try:
        result = ch.balance_text(text)
    except ch.ChemistryError as exc:
        st.error(str(exc))
        return
    st.session_state["_chem_balanced_text"] = text
    if result.status == "unbalanceable":
        st.warning(result.message)
        return
    if result.status == "underdetermined":
        st.info(result.message)
        if result.coefficients is None:
            return
    st.latex(result.latex())
    has_charge = any(row.item == "charge" for row in result.ledger)
    badge_row([("✅ Balanced" if result.status == "balanced" else "🟡 One of several balances", "pass" if result.status == "balanced" else "warn"),
               ("every atom recounted from the formulas", "pass"), ("charge conserved", "pass") if has_charge else ("neutral reaction", "neutral")])
    left, right = st.columns(2)
    with left:
        st.markdown("**The recount** (independent of how the coefficients were found)")
        st.dataframe([{"": row.item, "left side": row.left, "right side": row.right, "balanced": "✓" if row.ok else "✗"} for row in result.ledger],
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Molar masses**")
        n = len(result.reactants)
        coeffs = result.coefficients or []
        st.dataframe([{"species": ch.format_species(s), "coefficient": coeffs[i], "molar mass (g/mol)": round(ch.molar_mass(s), 3),
                       "side": "reactant" if i < n else "product"}
                      for i, s in enumerate(result.reactants + result.products)], hide_index=True, width="stretch")
    if result.basis:
        with st.expander("The independent balances"):
            st.write(result.basis)


# ------------------------------------------------------------------------------------------ stoichiometry

def _stoich_tab() -> None:
    text = st.text_input("Reaction", st.session_state.get("_chem_balanced_text", "C3H8 + O2 -> CO2 + H2O"), key="chem_stoich_reaction")
    try:
        result = ch.balance_text(text)
    except ch.ChemistryError as exc:
        st.error(str(exc))
        return
    if result.status != "balanced":
        st.info("This needs a uniquely balanced reaction: " + (result.message or "see the Balance tab."))
        return
    st.latex(result.latex())
    st.markdown("**Amounts of the reactants** (leave one unticked to treat it as in excess)")
    amounts: dict[int, tuple[float, str]] = {}
    for i, s in enumerate(result.reactants):
        c1, c2, c3 = st.columns([1, 2, 1])
        given = c1.checkbox(ch.format_species(s), value=i < 2, key=f"chem_given_{i}")
        value = c2.number_input("amount", 0.0, 1e9, 10.0, format="%g", key=f"chem_amt_{i}", disabled=not given, label_visibility="collapsed")
        unit = c3.selectbox("unit", ["g", "mol"], key=f"chem_unit_{i}", disabled=not given, label_visibility="collapsed")
        if given:
            amounts[i] = (float(value), unit)
    if not amounts:
        st.info("Tick at least one reactant and give its amount.")
        return
    try:
        res = ch.stoichiometry(result, amounts)
    except ch.ChemistryError as exc:
        st.error(str(exc))
        return
    badge_row([(f"limiting: {', '.join(res.limiting)}", "pass"), ("✅ mass conserved" if res.mass_conserved else "mass NOT conserved", "pass" if res.mass_conserved else "fail"),
               (f"{res.extent:.4g} mol of reaction", "neutral")])
    for note in res.notes:
        st.caption(f"ℹ️ {note}")
    st.dataframe([{"species": r.label, "role": r.role, "ν": r.coefficient, "M (g/mol)": round(r.molar_mass, 3),
                   "before (mol)": None if r.initial_mol is None else round(r.initial_mol, 5), "change (mol)": round(r.change_mol, 5),
                   "after (mol)": None if r.final_mol is None else round(r.final_mol, 5), "after (g)": None if r.final_g is None else round(r.final_g, 4)}
                  for r in res.rows], hide_index=True, width="stretch")
    st.caption(f"Check: {res.mass_consumed_g:.6g} g consumed = {res.mass_produced_g:.6g} g produced.")
    show_figure(cp.stoichiometry_figure(res), width="stretch")
    products = [r for r in res.rows if r.role == "product"]
    with st.expander("Percent yield"):
        which = st.selectbox("Product", [p.label for p in products], key="chem_yield_product")
        actual = st.number_input("Actual yield obtained (g)", 0.0, 1e12, 0.0, format="%g", key="chem_yield_actual")
        theo = next(p for p in products if p.label == which).final_g or 0.0
        if actual > 0 and theo > 0:
            st.metric("Percent yield", f"{ch.percent_yield(actual, theo):.1f}%", help=f"Theoretical yield: {theo:.4g} g")


# ------------------------------------------------------------------------------------------ acids and titration

def _ph_panel() -> None:
    kind = st.selectbox("Solution", ["Strong acid", "Strong base", "Weak acid", "Weak base", "Buffer (weak acid + its salt)"], key="chem_ph_kind")
    try:
        if kind in ("Strong acid", "Strong base"):
            c = st.number_input("Concentration (mol/L)", 1e-12, 100.0, 0.01, format="%g", key="chem_ph_c")
            r = ch.ph_strong(c, "acid" if kind == "Strong acid" else "base")
        elif kind == "Weak acid":
            c = st.number_input("Concentration (mol/L)", 1e-12, 100.0, 0.1, format="%g", key="chem_ph_c")
            pkas = _floats(st.text_input("pKa value(s), one per proton (up to 4)", "4.76", key="chem_ph_pka"))
            r = ch.ph_weak_acid(c, pkas)
        elif kind == "Weak base":
            c = st.number_input("Concentration (mol/L)", 1e-12, 100.0, 0.1, format="%g", key="chem_ph_c")
            r = ch.ph_weak_base(c, st.number_input("pKb", 0.0, 14.0, 4.75, format="%g", key="chem_ph_pkb"))
        else:
            pka = st.number_input("pKa", 0.0, 14.0, 4.76, format="%g", key="chem_ph_bpka")
            c1, c2 = st.columns(2)
            ca = c1.number_input("Weak acid (mol/L)", 1e-9, 100.0, 0.1, format="%g", key="chem_ph_ca")
            cb = c2.number_input("Its salt / conjugate base (mol/L)", 1e-9, 100.0, 0.1, format="%g", key="chem_ph_cb")
            b = ch.buffer_ph(pka, ca, cb)
            m1, m2, m3 = st.columns(3)
            m1.metric("pH (exact)", f"{b.exact_ph:.3f}")
            m2.metric("Henderson–Hasselbalch", f"{b.henderson_hasselbalch:.3f}")
            m3.metric("Difference", f"{b.difference:+.3f}")
            badge_row([("charge balance solved", "pass"), (f"residual {b.residual:.0e}", "pass")])
            return
    except (ch.ChemistryError, ValueError) as exc:
        st.error(str(exc))
        return
    m1, m2 = st.columns(2)
    m1.metric("pH (exact)", f"{r.ph:.3f}", help="Solved from the full charge balance, including water's own ions.")
    if r.approximation is not None:
        m2.metric("Textbook shortcut", f"{r.approximation:.3f}", delta=f"{r.approximation_error:+.3f}", delta_color="off")
    badge_row([("charge balance solved", "pass"), (f"residual {r.residual:.0e}", "pass")])
    if r.note:
        st.caption(r.note)


def _titration_panel() -> None:
    kind = st.radio("Titrating", ["Weak acid with NaOH", "Strong acid with NaOH", "Weak base with HCl", "Strong base with HCl"], horizontal=True, key="chem_tit_kind")
    c1, c2, c3 = st.columns(3)
    ca = c1.number_input("Analyte (mol/L)", 1e-6, 10.0, 0.1, format="%g", key="chem_tit_ca")
    va = c2.number_input("Analyte volume (mL)", 0.1, 10_000.0, 25.0, format="%g", key="chem_tit_va")
    cb = c3.number_input("Titrant (mol/L)", 1e-6, 10.0, 0.1, format="%g", key="chem_tit_cb")
    try:
        if kind.startswith("Weak acid"):
            pkas = _floats(st.text_input("pKa value(s), one per proton", "4.76", key="chem_tit_pka"))
            curve = ch.titration_curve("acid", pkas, ca, va, cb)
        elif kind.startswith("Strong acid"):
            curve = ch.titration_curve("acid", [], ca, va, cb, strong=True)
        elif kind.startswith("Weak base"):
            pkb = st.number_input("pKb of the base", 0.0, 14.0, 4.75, format="%g", key="chem_tit_pkb")
            curve = ch.titration_curve("base", [14.0 - pkb], ca, va, cb)
        else:
            curve = ch.titration_curve("base", [], ca, va, cb, strong=True)
    except (ch.ChemistryError, ValueError) as exc:
        st.error(str(exc))
        return
    badge_row([("✅ all checks pass" if curve.passed else "a check failed", "pass" if curve.passed else "fail")])
    show_figure(cp.titration_figure(curve), width="stretch")
    st.markdown("**The checks**")
    for chk in curve.checks:
        st.markdown(f"- {'✅' if chk.passed else '❌'} **{chk.label}**: {chk.detail}")
    if curve.half_equivalence:
        st.markdown("**Half-equivalence against pKa** (the textbook shortcut says they are equal)")
        st.dataframe([{"at (mL)": round(v, 3), "pH": round(p, 3), "pKa": e, "pH − pKa": round(p - e, 3)} for v, p, e in curve.half_equivalence],
                     hide_index=True, width="stretch")
        st.caption("A large difference means the shortcut does not apply there (overlapping steps, or a very dilute or strong acid).")


def _distribution_panel() -> None:
    pkas = _floats(st.text_input("pKa value(s)", "2.15, 7.20, 12.35", key="chem_dist_pka"))
    try:
        ph = st.slider("Mark a pH", 0.0, 14.0, 7.0, 0.1, key="chem_dist_ph")
        show_figure(cp.distribution_figure(pkas, ph), width="stretch")
        fractions = ch.alpha_fractions(10.0 ** (-ph), pkas)
        st.dataframe([{"form": f"{i} protons lost", "fraction at this pH": round(float(a), 5)} for i, a in enumerate(fractions)], hide_index=True, width="stretch")
    except (ch.ChemistryError, ValueError, IndexError) as exc:
        st.error(str(exc))


def _acids_tab() -> None:
    panel = st.radio("Show", ["pH of a solution", "Titration curve", "Species distribution"], horizontal=True, key="chem_acid_panel")
    {"pH of a solution": _ph_panel, "Titration curve": _titration_panel, "Species distribution": _distribution_panel}[panel]()


# ------------------------------------------------------------------------------------------ kinetics

def _kinetics_data() -> tuple[np.ndarray, np.ndarray] | None:
    source = st.radio("Data", ["Built-in example", "Paste data"], horizontal=True, key="chem_kin_source")
    if source == "Built-in example":
        c1, c2, c3, c4 = st.columns(4)
        order = int(c1.selectbox("True order", [0, 1, 2], index=1, key="chem_kin_order"))
        k = float(c2.number_input("k", 1e-4, 100.0, {0: 0.02, 1: 0.12, 2: 0.35}[order], format="%g", key=f"chem_kin_k_{order}"))
        noise = float(c3.slider("Noise (fraction of C0)", 0.0, 0.05, 0.01, 0.005, key="chem_kin_noise"))
        seed = int(c4.number_input("Seed", 0, 10 ** 6, 1, key="chem_kin_seed"))
        return ch.demo_kinetics(order, k, 1.0, noise, seed=seed)
    st.caption("Two columns, time then concentration (a header row is optional).")
    text = st.text_area("Data", height=120, key="chem_kin_paste", placeholder="t,c\n0,1.00\n5,0.55\n10,0.31\n…")
    if not text.strip():
        st.info("Paste concentration–time data to analyse.")
        return None
    try:
        t, X, _ = discover.parse_table(text)
    except discover.DiscoverError as exc:
        st.error(str(exc))
        return None
    if X.shape[1] != 1:
        st.error("Give exactly two columns: time and concentration.")
        return None
    return t, X[:, 0]


def _kinetics_tab() -> None:
    data = _kinetics_data()
    if data is None:
        return
    t, c = data
    try:
        rep = ch.fit_rate_law(t, c)
    except ch.ChemistryError as exc:
        st.error(str(exc))
        return
    best = rep.best
    if rep.ambiguous:
        st.warning(f"The data cannot tell {ch.ORDER_NAMES[best.order]} from {ch.ORDER_NAMES[rep.fits[1].order]}. Treat the order below as provisional.")
    badge_row([(f"{ch.ORDER_NAMES[best.order]}", "warn" if rep.ambiguous else "pass"), (f"k = {best.k:.4g} ± {best.k_se:.2g} {ch.k_units(best.order)}", "neutral"),
               (f"half-life ≈ {best.half_life:.4g}", "neutral")])
    for w in rep.warnings:
        if not w.startswith("The data cannot tell"):
            st.warning(w)
    st.dataframe([{"order": ch.ORDER_NAMES[f.order], "k": f"{f.k:.4g} ± {f.k_se:.2g}", "C₀": f"{f.c0:.4g}", "R² (nonlinear fit)": round(f.r2, 5),
                   "R² (linearised plot)": None if f.r2_linearised is None else round(f.r2_linearised, 5), "BIC": round(f.bic, 1)} for f in rep.fits],
                 hide_index=True, width="stretch")
    st.caption("The order is chosen by BIC on the nonlinear fits (lower is better; a difference below 2 is a tie), not by the R² of the straightened plots, "
               "which distort the noise.")
    show_figure(cp.kinetics_figure(rep), width="stretch")
    with st.expander("Linearised plots (the textbook diagnostic)"):
        show_figure(cp.linearised_figure(rep), width="stretch")
    expr, names, _ = ch.posterior_model(best.order)
    render_next_steps(ns.for_fit(t, c, expr, names, [best.c0, best.k], "🎲 Sample the posterior of k"), "kin")
    with st.expander("⚡ Activation energy from rate constants at several temperatures (Arrhenius)"):
        text = st.text_area("Temperature (K) and k, one pair per line", "290, 0.0021\n300, 0.0062\n310, 0.0168\n320, 0.0420", height=110, key="chem_arr_paste")
        try:
            rows = [[float(x) for x in line.replace(";", ",").split(",")] for line in text.strip().splitlines() if line.strip()]
            fit = ch.arrhenius_fit([r[0] for r in rows], [r[1] for r in rows])
        except (ValueError, IndexError, ch.ChemistryError) as exc:
            st.error(f"Give pairs of temperature (K) and k: {exc}")
            return
        m1, m2 = st.columns(2)
        m1.metric("Activation energy", f"{fit.ea_kj:.1f} kJ/mol", help=f"± {fit.ea_se_kj:.1f}")
        m2.metric("R² of ln k against 1/T", f"{fit.r2:.4f}")
        show_figure(cp.arrhenius_figure(fit), width="stretch")
        temp = st.number_input("Predict k at T (K)", 1.0, 5000.0, 305.0, key="chem_arr_T")
        st.write(f"k({temp:g} K) ≈ {fit.predict_k(temp):.4g}")


def render_chemistry_tab() -> None:
    st.subheader("🧪 Chemistry")
    st.caption("Balance reactions, find the limiting reagent, compute exact pH and titration curves, and fit reaction rates. Each result comes with the "
               "check behind it: an atom-and-charge recount, mass conservation, the charge-balance residual, or an honest note when the data cannot "
               "decide. No language model involved; 25 °C throughout.")
    tabs = st.tabs(["⚖️ Balance", "🧮 Stoichiometry", "🧪 Acids & titration", "⏱️ Kinetics"])
    with tabs[0]:
        _balance_tab()
    with tabs[1]:
        _stoich_tab()
    with tabs[2]:
        _acids_tab()
    with tabs[3]:
        _kinetics_tab()
