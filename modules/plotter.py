"""
Plots any derived equation/expression that has exactly one free
"plotting" variable once knowns + workspace values + slider-controlled
parameters are substituted in. Streamlit reruns the script on every
slider move, so the figure is recomputed live -- no manual callback wiring.
"""
import numpy as np
import plotly.graph_objects as go
import sympy as sp

from modules.equation_engine import Equation, ProblemModel


def plottable_free_symbols(eq: Equation, fixed_symbols: set[str]) -> list[str]:
    if eq.sympy_eq is None:
        return []
    expr = eq.sympy_eq.lhs - eq.sympy_eq.rhs
    return sorted(s.name for s in expr.free_symbols if s.name not in fixed_symbols)


def build_plot(model: ProblemModel, eq: Equation, x_symbol: str,
                param_values: dict[str, float], x_range: tuple[float, float],
                y_target: str | None = None) -> go.Figure:
    """
    eq:      the equation to plot
    x_symbol: which free symbol is the x-axis
    param_values: values for every OTHER free symbol (from sliders/workspace)
    x_range: (min, max) for the x-axis sweep
    y_target: which symbol to solve for and plot on the y-axis (must be one
              of model.solve_for). If None or not solvable, falls back to
              plotting the equation's residual (lhs - rhs) against x, with a
              dashed zero-line marking where the equation is satisfied.
    """
    x = sp.Symbol(x_symbol)
    xs = np.linspace(x_range[0], x_range[1], 400)

    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    fig = go.Figure()

    if y_target and y_target != x_symbol:
        target = sp.Symbol(y_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify(x, solved[0][target], "numpy")
            ys = f(xs)
            fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines",
                                      name=f"{y_target} vs {x_symbol}"))
            fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_target)
            return fig

    # fallback: plot the residual of the equation itself
    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify(x, residual, "numpy")
    ys = f(xs)
    fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines", name=eq.name))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title=x_symbol, yaxis_title=f"{eq.name} residual (0 = satisfied)")
    return fig
