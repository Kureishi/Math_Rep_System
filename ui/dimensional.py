"""
Dimensional-analysis-only mode: unit consistency of an expression with no problem to solve.
"""
import streamlit as st
from modules.dimensional_analysis import analyze_dimensions
from ui.common import tooltip_header


def render_dimensional_analysis_tab():
    """Dimensional-analysis-only mode: no numbers, no explicit formula --
    just the DIMENSIONS of a set of candidate input quantities and a
    desired output dimension, exploring which exponent combinations
    could possibly reach it. A genuinely different kind of problem from
    the rest of this app: nothing here solves a stated equation, it
    DISCOVERS which combinations of units could even plausibly combine
    into a target unit at all. See dimensional_analysis.py."""
    st.subheader("📐 Dimensional analysis")
    st.caption("Given just the UNITS of some candidate inputs and a desired output unit -- no "
                "numbers, no equation -- find which combinations of exponents could possibly "
                "reach it. The Buckingham-Pi-style exploration physics problems sometimes ask for "
                "directly, before any formula is even proposed.")

    st.write("**Candidate inputs** (one per line, `name: unit`):")
    inputs_text = st.text_area(
        "Inputs", height=120, key="dim_inputs_text", label_visibility="collapsed",
        placeholder="m: kg\na: m/s^2",
    )
    target_unit = st.text_input("Target unit", key="dim_target_unit", placeholder="N")

    if not st.button("Analyze", type="primary", key="dim_analyze_button"):
        return
    if not inputs_text.strip() or not target_unit.strip():
        st.warning("Enter at least one input and a target unit.")
        return

    input_units = {}
    for line in inputs_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            st.error(f"Couldn't parse line '{line}' -- expected `name: unit`.")
            return
        name, unit = line.split(":", 1)
        name, unit = name.strip(), unit.strip()
        if not name or not unit:
            st.error(f"Couldn't parse line '{line}' -- expected `name: unit`.")
            return
        input_units[name] = unit

    result = analyze_dimensions(input_units, target_unit.strip())

    if not result.feasible:
        st.error(result.message)
        return

    st.success(result.message)
    if result.particular_solution:
        exps = ", ".join(f"{name}^{exp}" for name, exp in result.particular_solution.items())
        tooltip_header("A particular solution:", "One valid exponent combination out of "
                          "potentially many (see 'degrees of freedom' in the message above) -- "
                          "'particular' just means this is ONE answer, not necessarily the only one, "
                          "the way a differential equation's 'particular solution' is one specific "
                          "solution rather than the whole family.", level="**")
        st.write(exps)

    if result.candidates:
        st.write("### Candidate formulas")
        st.caption("Every combination found by a bounded search over small rational exponents "
                    "(from \u22123 to 3 in steps of \u00bd) that matches the target dimension "
                    "exactly -- up to an unknown dimensionless constant C.")
        for c in result.candidates:
            st.write(f"- {c.description}")
    else:
        st.caption("No small-exponent candidates found in the search range, but the system is "
                    "still feasible (see the particular solution above, which may involve larger "
                    "or non-half-integer exponents).")
