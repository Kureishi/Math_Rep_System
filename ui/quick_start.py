"""
Quick Start mode: built-in example problems that jump into the right mode pre-filled.
"""
import streamlit as st


def render_quick_start_tab():
    """A gallery of one-click example problems, one per major mode --
    discoverability for a project whose sidebar has grown to cover many modes
    (see modules.command_palette.MODE_LABELS for the current count and list,
    several with their own multiple sub-tabs), where it's not always obvious
    from a mode's label alone what it's scoped to or what a well-formed
    input looks like. Clicking a card pre-fills the target mode's
    inputs via session_state (the same mechanism restore_from_query_param
    uses for shared links) and switches straight to it."""
    st.subheader("🚀 Quick start")
    st.caption("Pick an example to see a mode in action -- each button jumps straight to that "
                "mode with the inputs already filled in; press its Solve/Analyze/etc. button "
                "once there.")

    examples = [
        ("📝 Word problem solver", "Kinematics word problem",
         "A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its "
         "acceleration, and how far does it travel in that time?",
         {"problem_text": "A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. "
                            "What is its acceleration, and how far does it travel in that time?"}),
        ("📈 Curve fitting", "Fit a dataset (best fit, try every family)",
         "A small x/y dataset that fits an exponential curve well.",
         {"fit_csv_paste": "x,y\n1,5.4\n2,7.3\n3,9.9\n4,13.4\n5,18.2\n6,24.6\n7,33.4\n8,45.3",
          "fit_family": "best fit (try all)"}),
        ("🔁 Check equivalence", "A classic trig identity",
         "Confirm sin²(x) + cos²(x) really does equal 1, with the proof steps shown.",
         {"equiv_expr1": "sin(x)**2 + cos(x)**2", "equiv_expr2": "1"}),
        ("📐 Dimensional analysis", "Find a force-like combination",
         "Given mass and acceleration, discover which exponent combination reaches Newtons.",
         {"dim_inputs_text": "m: kg\na: m/s^2", "dim_target_unit": "N"}),
        ("🔄 Transforms & series", "Laplace transform of a damped oscillation",
         "Transform exp(-2t)sin(3t) and see it round-tripped back through the inverse transform.",
         {"laplace_expr": "exp(-2*t)*sin(3*t)"}),
        ("🌡️ PDE solver", "Heat equation, fixed-zero ends",
         "A rod pinned at 0° at both ends, initially warmer in the middle -- fills the "
         "\"Heat (fixed ends)\" sub-tab specifically.",
         {"heat_ic": "x*(1-x)"}),
        ("🧮 Tensor calculus", "Curvature of a sphere",
         "The default metric already IS this example -- just press Analyze to see the Ricci "
         "scalar come out to 2/R², with a slider to explore how it depends on R.",
         {}),
    ]

    for target_mode, title, description, prefill in examples:
        with st.container(border=True):
            st.write(f"**{title}**")
            st.caption(f"{target_mode} · {description}")
            if st.button("Try this →", key=f"quickstart_{title}"):
                # can't set st.session_state["app_mode"] directly here --
                # the app_mode radio widget was already instantiated
                # earlier in THIS script run (Streamlit forbids writing to
                # a widget's key after that widget exists in the current
                # run), so the mode switch + prefill is deferred to a
                # pending-state flag, applied at the very top of the next
                # run BEFORE the radio widget is (re)created -- see the
                # "_pending_mode" handling near the top of this file.
                st.session_state["_pending_mode"] = target_mode
                st.session_state["_pending_prefill"] = prefill
                st.rerun()
