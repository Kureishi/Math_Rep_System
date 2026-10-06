"""
Turning a click on a plot into something the app can act on -- Streamlit-free.

`st.plotly_chart(fig, on_select="rerun", key=...)` hands back (and stores in `st.session_state[key]`) a
selection state whose `["selection"]["points"]` lists the points that are selected, each a dict with at
least `x` and `y` and the trace and point index. The functions here read that state and answer the
questions the UI actually has: which tornado bar, which grid cell, which graph node.

They deliberately identify a point by what it IS on the figure (a bar's label, a grid point's x and y, a
node's coordinates) rather than by trace or point indices, which shift whenever a figure gains or loses
a trace, and rather than by `customdata`, which not every Plotly front end hands back. Everything also
accepts a plain dict as well as Streamlit's attribute-style dict, and `None`, because the state is
empty until the first click.
"""
import math
from collections.abc import Iterable, Sequence
from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, default)


def selection_points(state: Any) -> list[Any]:
    """The selected points of a plotly selection state, or [] if nothing is selected."""
    points = _get(_get(state, "selection"), "points")
    if not points:
        return []
    return list(points)


def clicked_labels(state: Any, axis: str = "y") -> list[str]:
    """The distinct values of `axis` ("x" or "y") over the selected points, as strings, in the order
    they were selected -- e.g. the labels of the selected bars of a horizontal bar chart."""
    seen: list[str] = []
    for p in selection_points(state):
        v = _get(p, axis)
        if v is None:
            continue
        label = str(v)
        if label not in seen:
            seen.append(label)
    return seen


def clicked_symbol(state: Any, valid: Iterable[str], axis: str = "y") -> str | None:
    """The first selected label that is one of `valid`, else None. A stale or foreign selection (the
    chart was rebuilt for another target, say) is ignored rather than trusted."""
    allowed = set(valid)
    for label in clicked_labels(state, axis):
        if label in allowed:
            return label
    return None


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def clicked_grid_point(state: Any, xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """The (x, y) of the first selected point that sits on the grid `xs` x `ys`, snapped to the grid's
    own values (so floating-point round-tripping through the browser cannot yield an off-grid value)."""
    if not len(xs) or not len(ys):
        return None
    for p in selection_points(state):
        px, py = _as_float(_get(p, "x")), _as_float(_get(p, "y"))
        if px is None or py is None:
            continue
        sx = min(xs, key=lambda v: abs(v - px))
        sy = min(ys, key=lambda v: abs(v - py))
        # accept only a point within a fraction of a grid step of a grid line -- not an arbitrary click
        if abs(sx - px) <= _tolerance(xs) and abs(sy - py) <= _tolerance(ys):
            return float(sx), float(sy)
    return None


def _tolerance(values: Sequence[float]) -> float:
    ordered = sorted(set(float(v) for v in values))
    if len(ordered) < 2:
        return 1e-9
    smallest_step = min(b - a for a, b in zip(ordered, ordered[1:]))
    return max(smallest_step * 0.25, 1e-12)


def grid_value(xs: Sequence[float], ys: Sequence[float], z: Sequence[Sequence[float]],
               x: float, y: float) -> float | None:
    """z at the grid point (x, y), where z[i][j] belongs to ys[i], xs[j] (the layout Plotly heatmaps and
    parameter_sweep.sweep_result_to_grid use). None if the point is not on the grid or z has no value
    there (a NaN)."""
    try:
        j = min(range(len(xs)), key=lambda k: abs(xs[k] - x))
        i = min(range(len(ys)), key=lambda k: abs(ys[k] - y))
        value = z[i][j]
    except (ValueError, IndexError, TypeError):
        return None
    return _as_float(value)


def clicked_node_ids(state: Any, nodes: Iterable[Any]) -> list[str]:
    """The ids of the graph nodes (anything with .id, .x, .y) that the selected points sit on, matched by
    coordinates -- see the module docstring for why not by customdata."""
    nodes = list(nodes)
    found: list[str] = []
    for p in selection_points(state):
        px, py = _as_float(_get(p, "x")), _as_float(_get(p, "y"))
        if px is None or py is None:
            continue
        for n in nodes:
            if abs(n.x - px) < 1e-9 and abs(n.y - py) < 1e-9 and n.id not in found:
                found.append(n.id)
                break
    return found


def targets_for_node(node_id: str, edges: Iterable[Any], available_targets: Iterable[str]) -> list[str]:
    """Which solved targets a clicked dependency-graph node leads to.

    * a variable node ("var:x") leads to its own steps if x is a solved target -- a given input has none;
    * an equation node ("eq:name") leads to the solved targets it feeds, found along its outgoing edges."""
    available = list(available_targets)
    if node_id.startswith("var:"):
        symbol = node_id[len("var:"):]
        return [symbol] if symbol in available else []
    if node_id.startswith("eq:"):
        out: list[str] = []
        for e in edges:
            if e.source == node_id and e.target.startswith("var:"):
                symbol = e.target[len("var:"):]
                if symbol in available and symbol not in out:
                    out.append(symbol)
        return out
    return []
