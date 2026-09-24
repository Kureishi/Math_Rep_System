"""
Problem-chains mode: feed one solved problem's result into the next as an input.
"""
import streamlit as st
import numpy as np
from modules.llm_client import LMStudioClient
from modules.equation_engine import extract_model, target_kind
from modules.verifier import verify
from modules.plotter import build_chain_sweep_plot
from modules.plot_snapshot import snapshot_chain_sweep_plot
from modules import chains
from modules.chains import InputBinding
from ui.common import format_download_button


def render_chains_tab(client: LMStudioClient):
    """Multi-problem dependency chains -- a named, persistent sequence
    of separately-extracted problems where a downstream step's input
    can be wired directly to an upstream step's solved output. Distinct
    from the single-problem "extract to workspace" flow (workspace.py,
    one-shot copy) and from dependency_graph.py (visualizes structure
    WITHIN one already-extracted problem): this module spans MULTIPLE
    problems and actually re-solves on an edit, cascading downstream
    the way a spreadsheet cell ripples through formulas that reference
    it. See modules/chains.py for the underlying persistence/resolve
    logic."""
    st.subheader("🔗 Problem chains")
    st.caption("A named, persistent sequence of solved problems where one step's output can feed "
                "the next step's input -- change an upstream input and everything downstream "
                "re-solves automatically, like a lightweight spreadsheet.")

    chain_list = chains.list_chains()
    chain_names = {c["id"]: c["name"] for c in chain_list}

    with st.expander("➕ New chain", expanded=not chain_list):
        new_name = st.text_input("Chain name", key="new_chain_name",
                                   placeholder="e.g. Projectile motion homework")
        if st.button("Create chain", key="create_chain_button") and new_name.strip():
            new_id = chains.create_chain(new_name.strip())
            st.session_state["active_chain_id"] = new_id
            st.rerun()

    if not chain_list:
        st.info("No chains yet -- create one above to get started.")
        return

    ids = [c["id"] for c in chain_list]
    default_index = ids.index(st.session_state["active_chain_id"]) \
        if st.session_state.get("active_chain_id") in ids else 0
    chosen_id = st.selectbox("Chain", ids, index=default_index,
                               format_func=lambda i: chain_names[i], key="active_chain_selector")
    st.session_state["active_chain_id"] = chosen_id
    chain = chains.load_chain(chosen_id)

    if st.button("🗑️ Delete this chain", key="delete_chain_button"):
        chains.delete_chain(chosen_id)
        st.session_state["active_chain_id"] = None
        for k in ["chain_pending_model", "chain_pending_text"]:
            st.session_state.pop(k, None)
        st.rerun()

    if chain.steps:
        st.write("### Steps")
        for step in chain.steps:
            icon = {"ok": "✅", "error": "❌", "stale": "➖"}.get(step.status, "➖")
            with st.expander(f"{icon} Step {step.position + 1}: {step.problem_text[:60]}"):
                st.caption(step.problem_text)
                st.write(f"Solves for **{step.output_symbol}**")
                if step.status == "ok":
                    st.success(f"{step.output_symbol} = {step.output_value:.6g}")
                elif step.status == "error":
                    st.error(step.error_detail)

                if step.bindings:
                    st.write("Inputs:")
                    for b in step.bindings:
                        if b.source == "literal":
                            st.write(f"- `{b.symbol}` = {b.literal_value} (fixed value)")
                        else:
                            st.write(f"- `{b.symbol}` ← step {b.upstream_position + 1}'s "
                                      f"`{b.upstream_symbol}`")

                    literal_bindings = [b for b in step.bindings if b.source == "literal"]
                    if literal_bindings:
                        edit_symbol = st.selectbox(
                            "Try a different value for...", [b.symbol for b in literal_bindings],
                            key=f"edit_binding_symbol_{step.position}")
                        current = next(b.literal_value for b in literal_bindings
                                        if b.symbol == edit_symbol)
                        new_val = st.number_input("New value", value=float(current),
                                                    key=f"edit_binding_value_{step.position}")
                        if st.button("Apply & re-solve chain", key=f"apply_binding_{step.position}"):
                            updated = [
                                InputBinding(symbol=b.symbol, source=b.source, literal_value=new_val,
                                              upstream_position=b.upstream_position,
                                              upstream_symbol=b.upstream_symbol)
                                if b.symbol == edit_symbol else b
                                for b in step.bindings
                            ]
                            chains.set_step_bindings(chosen_id, step.position, updated)
                            st.rerun()

                        # ---- research: sweep this input across a range and see
                        # every downstream step's output ripple in response, all
                        # at once -- the "what if I varied this input" view,
                        # rather than testing one value at a time by hand
                        with st.expander(f"📊 Sweep `{edit_symbol}` across a range"):
                            sweep_lo, sweep_hi = st.slider(
                                "Range", float(current) - 20, float(current) + 20,
                                (float(current) - 5, float(current) + 5),
                                key=f"sweep_range_{step.position}_{edit_symbol}",
                            )
                            sweep_points = st.slider("Number of points", 5, 50, 15,
                                                       key=f"sweep_points_{step.position}_{edit_symbol}")
                            if st.button("Run sweep", key=f"run_sweep_{step.position}_{edit_symbol}"):
                                sweep_values = np.linspace(sweep_lo, sweep_hi, sweep_points).tolist()
                                try:
                                    rows = chains.sweep_step_binding(
                                        chosen_id, step.position, edit_symbol, sweep_values)
                                except ValueError as e:
                                    st.error(str(e))
                                else:
                                    st.session_state[f"sweep_result_{step.position}_{edit_symbol}"] = rows
                            sweep_rows = st.session_state.get(f"sweep_result_{step.position}_{edit_symbol}")
                            if sweep_rows:
                                step_labels = {s.position: f"step {s.position + 1}: {s.output_symbol}"
                                                for s in chain.steps}
                                sweep_fig = build_chain_sweep_plot(sweep_rows, edit_symbol, step_labels)
                                st.plotly_chart(sweep_fig, width='stretch')
                                format_download_button(
                                    key=f"chain_sweep_{step.position}_{edit_symbol}",
                                    file_stem=f"chain_sweep_{edit_symbol}",
                                    render_fn=lambda fmt, r=sweep_rows, sl=step_labels, sym=edit_symbol:
                                        snapshot_chain_sweep_plot(r, sym, sl, fmt=fmt),
                                )

                if st.button("Remove this step", key=f"remove_step_{step.position}"):
                    chains.remove_step(chosen_id, step.position)
                    st.rerun()

    st.write("### Add a step")
    step_text = st.text_area("New problem text", key="chain_new_step_text", height=100,
                               placeholder="e.g. If the car keeps that same acceleration for another "
                                           "10 seconds, how much extra distance does it cover?")
    if st.button("Extract & verify", key="chain_extract_button") and step_text.strip():
        with st.spinner("Deriving and verifying..."):
            try:
                new_model = extract_model(client, step_text)
                new_report = verify(new_model, client, step_text)
            except Exception as e:  # noqa: BLE001
                st.error(f"Extraction failed: {e}")
                new_model, new_report = None, None
        if new_model is not None:
            st.session_state["chain_pending_model"] = new_model
            st.session_state["chain_pending_text"] = step_text
            if new_report is not None and not new_report.passed:
                st.warning(f"Verification didn't fully pass: {new_report.failure_reason}. "
                            "You can still add it, but double-check the derivation.")

    pending_model = st.session_state.get("chain_pending_model")
    if pending_model is not None:
        algebraic_targets = [t for t in pending_model.solve_for
                               if target_kind(pending_model, t) == "equation"]
        if not algebraic_targets:
            st.warning("This problem has no algebraic solve_for target -- it can't be added to a chain.")
        else:
            output_symbol = st.selectbox("Which target should this step expose downstream?",
                                           algebraic_targets, key="chain_output_symbol")
            unknown_vars = [v for v in pending_model.variables
                              if v.known_value is None and v.symbol != output_symbol]
            bindings: list[InputBinding] = []
            if unknown_vars:
                st.caption("This problem has unknown inputs -- give each one a value before adding it:")
                suggested = {b.symbol: b for b in chains.suggest_bindings(chain, pending_model)}
                for var in unknown_vars:
                    step_options = [f"step {s.position + 1}: {s.output_symbol}" for s in chain.steps]
                    options = ["(leave unbound)"] + step_options + ["fixed value"]
                    default_idx = 0
                    match = suggested.get(var.symbol)
                    if match is not None:
                        default_idx = options.index(
                            f"step {match.upstream_position + 1}: {match.upstream_symbol}")
                    choice = st.selectbox(f"Input for `{var.symbol}` ({var.meaning})", options,
                                            index=default_idx, key=f"chain_bind_choice_{var.symbol}")
                    if choice == "fixed value":
                        lit = st.number_input(f"Value for `{var.symbol}`",
                                                key=f"chain_bind_literal_{var.symbol}")
                        bindings.append(InputBinding(symbol=var.symbol, source="literal",
                                                       literal_value=lit))
                    elif choice != "(leave unbound)":
                        pos = int(choice.split(":")[0].replace("step ", "")) - 1
                        upstream_step = next(s for s in chain.steps if s.position == pos)
                        bindings.append(InputBinding(symbol=var.symbol, source="upstream",
                                                       upstream_position=pos,
                                                       upstream_symbol=upstream_step.output_symbol))
            if st.button("➕ Add to chain", key="chain_add_step_button"):
                try:
                    chains.add_step(chosen_id, st.session_state["chain_pending_text"], pending_model,
                                      output_symbol, bindings)
                except ValueError as e:
                    st.error(str(e))
                else:
                    for k in ["chain_pending_model", "chain_pending_text"]:
                        st.session_state.pop(k, None)
                    st.rerun()
