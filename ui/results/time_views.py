"""
Time-resolved views of a solved problem that need the symbolic solution as a
FORMULA in its parameters (and initial values), not just one evaluated curve:
the uncertainty fan and the parameter morph for an ODE, and the bifurcation
diagram (with a cobweb) for a first-order map.
"""
import numpy as np
import streamlit as st
import sympy as sp

from modules.bifurcation import bifurcation_diagram, lambdify_map
from modules.equation_engine import ProblemModel
from modules.ode_utils import _funcs_used, _ics_for_group, group_coupled_odes, solve_ode
from modules.parameter_morph import first_regime_change, morph_family
from modules.plot_snapshot import (
    snapshot_bifurcation_plot, snapshot_cobweb_gif, snapshot_parameter_morph_gif, snapshot_uncertainty_fan,
)
from modules.plotter import build_bifurcation_plot, build_cobweb_plot, build_parameter_morph, build_uncertainty_fan
from modules.recurrence_utils import extract_step_map, solve_recurrence
from modules.time_uncertainty import candidate_parameters, ode_uncertainty_fan
from ui.cache import cached, cached_by_model
from ui.fragments import isolated
from ui.common import gif_download_button, snapshot_button


def _symbolic_solutions(model: ProblemModel) -> dict[str, sp.Eq]:
    """The ODE solutions with every initial value left symbolic (see
    solve_ode(symbolic_initial_conditions=True)), cached for the session:
    a Streamlit rerun happens on every widget touch, and re-solving the
    differential equation each time would make the sliders crawl."""
    return cached_by_model("solve_ode_symbolic", model, lambda: solve_ode(model, symbolic_initial_conditions=True))


def _initial_time(model: ProblemModel, group) -> float:
    func_names: set[str] = set()
    for eq in group:
        if eq.sympy_eq is not None:
            func_names |= _funcs_used(eq.sympy_eq)
    for lhs in _ics_for_group(model, func_names):
        point = lhs.args[0]
        if point.is_number:
            return float(point)
    return 0.0


def render_ode_time_views(model: ProblemModel) -> None:
    """Uncertainty fan + parameter morph under every ODE group whose
    solution is fully determined (every initial value given, every
    parameter known). Silently skipped otherwise, same convention as the
    phase portrait and cobweb: a fan around an unresolved constant means
    nothing."""
    ode_eqs = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    if not ode_eqs:
        return
    symbolic = _symbolic_solutions(model)
    if not symbolic:
        return
    for group in group_coupled_odes(ode_eqs):
        params, unresolved = candidate_parameters(model, group, symbolic)
        if unresolved or not params:
            continue
        names = sorted(n for n in {f for e in group if e.sympy_eq is not None for f in _funcs_used(e.sympy_eq)}
                       if n in symbolic)
        if not names:
            continue
        label = ", ".join(names)
        t0 = _initial_time(model, group)
        t_sym = next(iter(symbolic.values())).lhs.args[0]
        _render_fan(model, group, symbolic, params, label, t0)
        _render_morph(symbolic, params, names, label, t0, t_sym)


@isolated
def _render_fan(model, group, symbolic, params, label, t0) -> None:
    with st.expander(f"🌫️ Uncertainty over time: {label}"):
        st.caption("Give any parameter or initial value a standard deviation. Draws from those distributions "
                    "are pushed through the closed-form solution at every time, and the bands show how the "
                    "uncertainty grows or shrinks as the solution evolves.")
        cols = st.columns(min(3, len(params)))
        sigmas: dict[str, float] = {}
        for i, p in enumerate(params):
            with cols[i % len(cols)]:
                default = abs(p.value) * 0.05
                sigmas[p.name] = st.number_input(
                    f"{p.name} \u00b1 (std)", min_value=0.0, value=float(default), format="%.4g",
                    key=f"fan_sd_{label}_{p.name}_{p.value:g}",
                    help=f"Nominal value {p.value:g} ({p.kind}). 0 means certain.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            t_end = st.number_input("Out to t =", value=t0 + 10.0, key=f"fan_tend_{label}")
        with c2:
            n_samples = st.slider("Samples", 200, 5000, 800, step=100, key=f"fan_n_{label}")
        with c3:
            seed = st.number_input("Random seed", 0, 2 ** 31 - 1, 12345, key=f"fan_seed_{label}",
                                    help="Same seed, same fan -- recorded in the exported figure.")
        with c4:
            want_envelope = st.checkbox("Guaranteed envelope", value=True, key=f"fan_env_{label}",
                                         help="A bound from interval arithmetic that the solution cannot "
                                               "leave, as opposed to the sampled bands.")
            k_sigmas = st.number_input("\u00b1 \u03c3 for envelope", 0.5, 6.0, 2.0, step=0.5,
                                        key=f"fan_k_{label}", disabled=not want_envelope)
        if not any(s > 0 for s in sigmas.values()):
            st.info("Give at least one input a standard deviation above zero to see a fan.")
            return
        fan = ode_uncertainty_fan(model, group, symbolic, sigmas, (t0, float(t_end)), n_samples=int(n_samples),
                                   seed=int(seed), envelope_sigmas=float(k_sigmas) if want_envelope else None)
        if not fan.applicable:
            st.warning(fan.reason)
            return
        st.plotly_chart(cached(f"fan:{label}", (fan,), lambda: build_uncertainty_fan(fan)),
                         width="stretch", key=f"fan_{label}")
        for i, name in enumerate(fan.names):
            st.write(f"**{name}** at t = {fan.t[-1]:g}: nominal {fan.nominal[i][-1]:.4g}, 90% of samples in "
                      f"[{fan.p5[i][-1]:.4g}, {fan.p95[i][-1]:.4g}]")
        if want_envelope:
            st.caption(fan.envelope_note or "No guaranteed envelope for this solution.")
        if fan.finite_fraction < 1.0:
            st.warning(f"{(1 - fan.finite_fraction) * 100:.1f}% of the sampled values had no real result "
                        "and were left out of the bands.")
        snapshot_button(
            key=f"timefan_{label}", title=f"Uncertainty over time: {label}",
            caption=f"{fan.n_samples} samples, seed {fan.seed}; uncertain: "
                    + ", ".join(f"{k} \u00b1{v:g}" for k, v in fan.uncertain.items()),
            render_fn=lambda f=fan: snapshot_uncertainty_fan(f))


@isolated
def _render_morph(symbolic, params, names, label, t0, t_sym) -> None:
    with st.expander(f"🎞️ Parameter morph: {label}"):
        st.caption("Vary one parameter (or initial value) across a range and watch the whole solution change "
                    "shape. The grey curves are every value tried; the red one is the current frame.")
        c1, c2 = st.columns(2)
        with c1:
            func = st.selectbox("Function", names, key=f"morph_func_{label}") if len(names) > 1 else names[0]
        with c2:
            chosen = st.selectbox("Vary", [p.name for p in params], key=f"morph_param_{label}")
        p = next(q for q in params if q.name == chosen)
        v = p.value
        default_lo, default_hi = ((0.1 * v, 3.0 * v) if v > 0 else (3.0 * v, 0.1 * v) if v < 0 else (-1.0, 1.0))
        c3, c4, c5, c6 = st.columns(4)
        with c3:
            lo = st.number_input("From", value=float(default_lo), format="%.4g", key=f"morph_lo_{label}_{chosen}_{v:g}")
        with c4:
            hi = st.number_input("To", value=float(default_hi), format="%.4g", key=f"morph_hi_{label}_{chosen}_{v:g}")
        with c5:
            t_end = st.number_input("Out to t =", value=t0 + 10.0, key=f"morph_tend_{label}")
        with c6:
            n_values = st.slider("Values", 10, 80, 40, key=f"morph_n_{label}")
        morph = morph_family(symbolic[func].rhs, t_sym, p.symbol, float(lo), float(hi), (t0, float(t_end)),
                              {q.symbol: q.value for q in params if q is not p}, function=func,
                              n_values=int(n_values), nominal_value=v if lo <= v <= hi else None, label=chosen)
        if not morph.applicable:
            st.warning(morph.reason)
            return
        st.plotly_chart(cached(f"morph:{label}", (morph,), lambda: build_parameter_morph(morph)),
                         width="stretch", key=f"morph_{label}_{chosen}")
        change = first_regime_change(morph)
        if change is None:
            st.caption(f"The shape doesn't change character across this range "
                        f"({morph.turning_points[0]} visible turning point(s) throughout).")
        else:
            before, after, tp_before, tp_after = change
            st.caption(f"The shape first changes between {chosen} = {before:.4g} and {after:.4g}: "
                        f"{tp_before} \u2192 {tp_after} visible turning points.")
        gif_download_button(key=f"morph_{label}_{chosen}", file_stem=f"morph_{func}_{chosen}",
                             render_fn=lambda m=morph: snapshot_parameter_morph_gif(m))


def render_map_views(model: ProblemModel) -> None:
    """Bifurcation diagram (and, where the recurrence has no closed form
    and so no cobweb elsewhere, a cobweb) for every first-order map
    a(n+1) = g(a(n)) in the model."""
    funcs = sorted({f for e in model.equations if e.kind == "recurrence" and e.sympy_eq is not None
                    for f in _funcs_used(e.sympy_eq)})
    has_closed_form = set(cached_by_model("solve_recurrence", model, lambda: solve_recurrence(model)))
    for func_name in funcs:
        step_map = extract_step_map(model, func_name)
        if step_map is None:
            continue
        g_expr, g_var = step_map
        known = {sp.Symbol(v.symbol): float(v.known_value) for v in model.variables if v.known_value is not None}
        free = sorted(g_expr.free_symbols - {g_var}, key=str)
        if not free:
            continue
        _render_bifurcation(model, func_name, g_expr, g_var, free, known)
        if func_name not in has_closed_form:
            _render_cobweb_without_closed_form(model, func_name, g_expr, g_var, known)


def _initial_value(model: ProblemModel, func_name: str) -> float | None:
    for ic in model.initial_conditions:
        if ic.sympy_eq is None:
            continue
        for f in ic.sympy_eq.lhs.atoms(sp.core.function.AppliedUndef):
            if str(f.func) == func_name:
                return float(ic.sympy_eq.rhs)
    return None


@isolated
def _render_bifurcation(model, func_name, g_expr, g_var, free, known) -> None:
    with st.expander(f"🌿 Bifurcation diagram for {func_name}"):
        st.caption("Iterate the map for each value of one parameter, throw away the first iterates, and plot "
                    "where the rest settle: a fixed point, then a cycle that doubles (2, 4, 8, ...), then "
                    "chaos. The green line marks the parameter's current value.")
        chosen = st.selectbox("Vary", [s.name for s in free], key=f"bif_param_{func_name}") if len(free) > 1 \
            else free[0].name
        param = next(s for s in free if s.name == chosen)
        others = {s: known[s] for s in free if s != param and s in known}
        missing = [s.name for s in free if s != param and s not in known]
        if missing:
            st.caption(f"Give {', '.join(missing)} a value to draw this diagram.")
            return
        current = known.get(param)
        if current is not None and current > 0:
            default_lo, default_hi = 0.5 * current, 1.25 * current
        elif current is not None and current < 0:
            default_lo, default_hi = 1.25 * current, 0.5 * current
        else:
            default_lo, default_hi = (0.0, 4.0) if current is None else (-1.0, 1.0)
        start = _initial_value(model, func_name)
        c1, c2, c3 = st.columns(3)
        with c1:
            lo = st.number_input("From", value=float(default_lo), format="%.4g", key=f"bif_lo_{func_name}_{chosen}")
        with c2:
            hi = st.number_input("To", value=float(default_hi), format="%.4g", key=f"bif_hi_{func_name}_{chosen}")
        with c3:
            x0 = st.number_input("Start value", value=float(start if start is not None else 0.5),
                                  format="%.4g", key=f"bif_x0_{func_name}")
        c4, c5 = st.columns(2)
        with c4:
            n_params = st.slider("Parameter values", 100, 1500, 500, step=100, key=f"bif_n_{func_name}")
        with c5:
            n_transient = st.slider("Iterates discarded", 100, 3000, 600, step=100, key=f"bif_t_{func_name}")
        g = lambdify_map(g_expr, g_var, param, others)
        if isinstance(g, str):
            st.warning(g)
            return
        result = bifurcation_diagram(g, chosen, (float(lo), float(hi)), float(x0), n_params=int(n_params),
                                      n_transient=int(n_transient))
        if not result.applicable:
            st.warning(result.reason)
            return
        marker = current if current is not None else None
        st.plotly_chart(cached(f"bifurcation:{func_name}", (result, marker),
                                lambda: build_bifurcation_plot(result, marker=marker)),
                         width="stretch", key=f"bif_{func_name}")
        if result.transitions:
            shown = [f"{chosen} \u2248 {p:.4g}: " + (f"period {k}" if k else "no short cycle")
                     for p, k in result.transitions[:8]]
            st.caption("Regime changes: " + "; ".join(shown) + (" ..." if len(result.transitions) > 8 else ""))
        else:
            st.caption("One long-run behaviour throughout this range.")
        snapshot_button(
            key=f"bifurcation_{func_name}", title=f"Bifurcation diagram: {func_name}",
            caption=f"{chosen} from {lo:g} to {hi:g}, start value {x0:g}",
            render_fn=lambda r=result, m=marker: snapshot_bifurcation_plot(r, marker=m))


@isolated
def _render_cobweb_without_closed_form(model, func_name, g_expr, g_var, known) -> None:
    """The cobweb diagram, for a map whose recurrence has no closed form
    (a nonlinear map like the logistic): the cobweb in the main recurrence
    section is only drawn next to a closed-form solution, so it would
    otherwise never appear for exactly the maps where it is most useful."""
    leftover = sorted(s.name for s in g_expr.free_symbols - {g_var} if s not in known)
    if leftover:
        return
    with st.expander(f"🕸️ Cobweb diagram for {func_name}"):
        st.caption("The curve is the one-step map, the diagonal is where a value would repeat itself, and the "
                    "staircase traces the sequence bouncing between the two.")
        try:
            g_numeric = sp.lambdify(g_var, g_expr.subs(known), "numpy")
            start = _initial_value(model, func_name)
            x0 = float(start if start is not None else 0.5)
            span = max(abs(x0), 1.0) * 1.5
            x_range = st.slider("Plot range", -10.0 * span, 10.0 * span, (min(-0.1, x0 - span), max(1.1, x0 + span)),
                                 key=f"cobweb2_range_{func_name}")
            steps = st.slider("Steps to show", 2, 60, 25, key=f"cobweb2_steps_{func_name}")
            st.plotly_chart(build_cobweb_plot(g_numeric, x0, x_range, n_steps=steps, x_label=func_name),
                             width="stretch", key=f"cobweb2_{func_name}")
            gif_download_button(
                key=f"cobweb2_{func_name}", file_stem=f"cobweb_{func_name}",
                render_fn=lambda gn=g_numeric, x0_=x0, xr=x_range, n=steps, fn=func_name:
                    snapshot_cobweb_gif(gn, x0_, xr, n_steps=n, x_label=fn))
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Couldn't build a cobweb diagram: {exc}")
