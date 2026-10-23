"""
Light cones, null geodesics and Penrose diagrams -- Streamlit-free.

LIGHT CONES IN A COORDINATE PLANE. Pick any two coordinates (x_a, x_b) of a metric and hold the others fixed.
The induced 2-metric is h = [[g_aa, g_ab], [g_ab, g_bb]]; its null directions satisfy
g_aa da^2 + 2 g_ab da db + g_bb db^2 = 0, a quadratic with two roots -- the two edges of the light cone at that
point. Where the cone tilts, narrows or closes up is the causal structure made visible: in Schwarzschild
coordinates the cones pinch to vertical lines at r = 2M (the coordinates cannot follow light across the
horizon), while in Eddington-Finkelstein coordinates they stay open and tilt inward until, at r = 2M, one edge
points straight along the horizon. Cones are drawn in "display units": the two axes have different units, so
each direction is rescaled by `aspect` before its length is set -- only angles and ordering are meaningful.

NULL GEODESICS. For the 2-slice of a spherically symmetric metric (t, r), radial light rays are exactly the
integral curves of the cone edges, so they are traced by following one edge continuously (the root closest to
the previous direction, which keeps the curve on one family through places where a slope would be infinite).
The rays are NOT a substitute for the full geodesic integrator (gr_geodesics) -- they are the planar picture
only -- and a test checks that a radial ray from the integrator lies on the same curve.

PENROSE DIAGRAMS are drawn for the two cases with exact closed forms:
  * Minkowski: null coordinates u = t - r, v = t + r compactified by arctan;
  * Schwarzschild, maximally extended, from Kruskal-Szekeres coordinates (T, X), with
    (r/2M - 1) e^{r/2M} = X^2 - T^2 and tanh(t/4M) = T/X (region I): all four regions, both horizons (45-degree
    lines), the singularities r = 0 (T^2 - X^2 = 1) and null infinity. With U = T - X, V = T + X compactified by
    arctan the null rays stay at 45 degrees (the map acts on U and V separately, monotonically).
There is no general algorithm that compactifies an arbitrary metric, so other metrics get light cones and null
geodesics but not a Penrose diagram -- the module says so rather than drawing something misleading.
"""
import math
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from modules.gr_metrics import GRInputError, GRMetric


# ------------------------------------------------------------------------------------------ light cones

def _induced_function(metric: GRMetric, ia: int, ib: int, base: dict[sp.Symbol, float]):
    """A fast f(a, b) -> (g_aa, g_ab, g_bb) for the plane (ia, ib), the other coordinates and the parameters fixed at
    `base` (lambdified once: evaluating SymPy expressions point by point would make a cone grid take seconds)."""
    g = metric.g
    free = [metric.coords[ia], metric.coords[ib]]
    fixed = {k: v for k, v in base.items() if k not in free}
    expr = [sp.sympify(g[ia, ia]).subs(fixed), sp.sympify(g[ia, ib]).subs(fixed), sp.sympify(g[ib, ib]).subs(fixed)]
    fn = sp.lambdify(free, expr, modules="math")

    def induced(a: float, b: float) -> tuple[float, float, float]:
        try:
            vals = fn(a, b)
        except (ZeroDivisionError, ValueError, OverflowError):
            raise GRInputError("The metric is not finite at that point.") from None
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise GRInputError("The metric is not finite at that point.")
        return float(vals[0]), float(vals[1]), float(vals[2])
    return induced


def cone_edges(gaa: float, gab: float, gbb: float) -> list[tuple[float, float]] | None:
    """The two null directions (da, db), unit length in the (a, b) plane, of the quadratic form [[gaa, gab], [gab, gbb]]
    -- or None if the form is not Lorentzian (det >= 0: no real null directions, or a degenerate one)."""
    det = gaa * gbb - gab * gab
    if det >= -1e-14:
        return None
    # solve gaa x^2 + 2 gab x y + gbb y^2 = 0 for the direction (x, y)
    dirs = []
    disc = math.sqrt(gab * gab - gaa * gbb)
    if abs(gaa) > 1e-14 * max(1.0, abs(gbb), abs(gab)):
        for s in (1, -1):
            x, y = (-gab + s * disc) / gaa, 1.0                  # x = da/db
            dirs.append((x, y))
    else:                                                        # gaa = 0: directions db = 0 and (da/db) = -gbb/(2 gab)
        dirs.append((1.0, 0.0))
        dirs.append((-gbb / (2 * gab), 1.0))
    out = []
    for x, y in dirs:
        norm = math.hypot(x, y)
        out.append((x / norm, y / norm))
    return out


@dataclass
class LightCone:
    center: tuple[float, float]
    edges: list[tuple[float, float]]            # unit directions in DISPLAY units, future-pointing along `time_axis`
    open: bool                                  # False where the cone is degenerate / not Lorentzian


def light_cones(metric: GRMetric, plane: tuple[str, str], grid_a: np.ndarray, grid_b: np.ndarray,
                fixed: dict[str, float] | None = None, params: dict[str, float] | None = None,
                aspect: float = 1.0, time_axis: str | None = None) -> list[LightCone]:
    """Light cones on a grid of points in the (a, b) coordinate plane. `plane` names the two coordinates; the
    other coordinates are fixed at `fixed` (default: the metric's sample point). `time_axis` is the plane
    coordinate whose increase defines 'future' (default: the first of the two if it is the metric's time
    coordinate, else the second)."""
    names = metric.coord_names
    if plane[0] == plane[1] or plane[0] not in names or plane[1] not in names:
        raise GRInputError("Choose two different coordinates of the metric for the plane.")
    ia, ib = names.index(plane[0]), names.index(plane[1])
    time_axis = time_axis or (plane[0] if ia == 0 else plane[1] if ib == 0 else plane[0])
    base = dict(zip(metric.coords, metric.sample_point()))
    base.update({metric.coords[names.index(k)]: float(v) for k, v in (fixed or {}).items() if k in names})
    base.update(metric.param_subs(params))
    cones: list[LightCone] = []
    induced = _induced_function(metric, ia, ib, base)
    for a in grid_a:
        for b in grid_b:
            try:
                edges = cone_edges(*induced(float(a), float(b)))
            except GRInputError:
                edges = None
            if edges is None:
                cones.append(LightCone((float(a), float(b)), [], False))
                continue
            future = 0 if time_axis == plane[0] else 1
            disp = []
            for (da, db) in edges:
                if (da, db)[future] < 0 or ((da, db)[future] == 0):
                    da, db = -da, -db
                disp.append((da, db))
            # display units: rescale the b axis by `aspect`, renormalise, so a 45-degree edge looks like one
            disp = [_norm((da, db * aspect)) for da, db in disp]
            cones.append(LightCone((float(a), float(b)), disp, True))
    return cones


def _norm(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else v


def null_curve(metric: GRMetric, plane: tuple[str, str], start: tuple[float, float], branch: int, length: float,
               fixed: dict[str, float] | None = None, params: dict[str, float] | None = None, steps: int = 400,
               time_axis: str | None = None, bounds: tuple[tuple[float, float], tuple[float, float]] | None = None) -> np.ndarray:
    """A null curve in the plane, following cone edge `branch` (0 or 1) from `start` for coordinate length `length`
    (arc length in the (a, b) coordinates), as an (N, 2) array. Stops early at a degenerate cone or outside `bounds`."""
    names = metric.coord_names
    ia, ib = names.index(plane[0]), names.index(plane[1])
    base = dict(zip(metric.coords, metric.sample_point()))
    base.update({metric.coords[names.index(k)]: float(v) for k, v in (fixed or {}).items() if k in names})
    base.update(metric.param_subs(params))
    time_axis = time_axis or (plane[0] if ia == 0 else plane[1])
    future = 0 if time_axis == plane[0] else 1

    induced = _induced_function(metric, ia, ib, base)

    def direction(a: float, b: float, prev: tuple[float, float] | None) -> tuple[float, float] | None:
        try:
            edges = cone_edges(*induced(a, b))
        except GRInputError:
            return None
        if edges is None:
            return None
        edges = [e if e[future] > 0 else (-e[0], -e[1]) for e in edges]
        if prev is None:
            edges.sort(key=lambda e: e[1 - future])
            return edges[branch % 2]
        return max(edges, key=lambda e: e[0] * prev[0] + e[1] * prev[1])

    pts = [np.array(start, dtype=float)]
    h = length / steps
    d = direction(float(pts[0][0]), float(pts[0][1]), None)
    prev = d
    for _ in range(steps):
        if d is None:
            break
        x = pts[-1]
        k1 = np.array(d)
        mid = x + 0.5 * h * k1
        d2 = direction(float(mid[0]), float(mid[1]), d)
        if d2 is None:
            break
        k2 = np.array(d2)
        nxt = x + h * k2
        if bounds and not (bounds[0][0] <= nxt[0] <= bounds[0][1] and bounds[1][0] <= nxt[1] <= bounds[1][1]):
            break
        pts.append(nxt)
        d = direction(float(nxt[0]), float(nxt[1]), prev)
        prev = d if d is not None else prev
    return np.array(pts)


# ------------------------------------------------------------------------------------------ Penrose: Minkowski

@lru_cache(maxsize=2)
def minkowski_penrose(n_lines: int = 7, samples: int = 200) -> dict[str, list[np.ndarray]]:
    """Penrose diagram of Minkowski space in (X', T') = ((p - q)/2, (p + q)/2) with p = arctan(t + r),
    q = arctan(t - r). Returns polylines grouped as 'const_r', 'const_t', 'boundary' (all (N, 2) arrays of (X', T'))."""
    out: dict[str, list[np.ndarray]] = {"const_r": [], "const_t": [], "boundary": []}

    def pen(t: np.ndarray, r: np.ndarray) -> np.ndarray:
        p, q = np.arctan(t + r), np.arctan(t - r)
        return np.column_stack([(p - q) / 2, (p + q) / 2])

    ts = np.tan(np.linspace(-1.5, 1.5, samples))
    rs = np.tan(np.linspace(0.0, 1.5, samples))
    for r0 in np.tan(np.linspace(0.15, 1.35, n_lines)):
        out["const_r"].append(pen(ts, np.full_like(ts, r0)))
    for t0 in np.tan(np.linspace(-1.2, 1.2, n_lines)):
        out["const_t"].append(pen(np.full_like(rs, t0), rs))
    h = math.pi / 2
    s = np.linspace(0, 1, 100)
    out["boundary"] += [np.column_stack([s * h, (1 - s) * h]), np.column_stack([s * h, -(1 - s) * h]),
                        np.column_stack([np.zeros_like(s), np.linspace(-h, h, 100)])]
    return out


# ------------------------------------------------------------------------------------------ Penrose: Schwarzschild

def _kruskal_to_penrose(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    p, q = np.arctan(T + X), np.arctan(T - X)
    return np.column_stack([(p - q) / 2, (p + q) / 2])


def kruskal_radius(T: float, X: float, M: float) -> float:
    """r from (r/2M - 1) e^{r/2M} = X^2 - T^2 (the Lambert-W relation), by bracketing; r > 0 on this relation iff X^2 - T^2 > -1."""
    w = X * X - T * T
    if w <= -1:
        raise GRInputError("(T, X) is beyond the singularity r = 0 (T^2 - X^2 > 1).")
    fn = lambda rr: (rr / (2 * M) - 1) * math.exp(rr / (2 * M)) - w  # noqa: E731
    hi = 2 * M
    while fn(hi) <= 0:                      # (x - 1) e^x grows exponentially, so this takes a handful of steps and never overflows
        hi *= 1.5
    return brentq(fn, 0.0, hi, xtol=1e-14)


@lru_cache(maxsize=8)
def schwarzschild_penrose(M: float = 1.0, n_lines: int = 5, samples: int = 240) -> dict[str, list[np.ndarray]]:
    """The maximally extended Schwarzschild spacetime as Penrose-diagram polylines, in (X', T') coordinates:
    'horizon' (the two lines T = +-X), 'singularity' (T^2 - X^2 = 1, top and bottom), 'infinity' (the four null
    boundaries), 'const_r' and 'const_t' (curves in regions I and II/III/IV), and 'rays' (outgoing/ingoing radial
    null lines at 45 degrees). Every point is computed from Kruskal coordinates, not drawn by hand."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("horizon", "singularity", "infinity", "const_r", "const_t", "rays")}
    big = 1e3
    s = np.linspace(-big, big, 4001)
    out["horizon"] += [_kruskal_to_penrose(s, s), _kruskal_to_penrose(s, -s)]
    xs = np.sinh(np.linspace(-6, 6, samples))
    for sign in (1, -1):
        out["singularity"].append(_kruskal_to_penrose(sign * np.sqrt(1 + xs ** 2), xs))
    h = math.pi / 2
    sl = np.linspace(0, 1, 100)
    for sx in (1, -1):
        for st in (1, -1):
            out["infinity"].append(np.column_stack([sx * (h - sl * h), st * sl * h]))
    # constant r > 2M in regions I and IV: X^2 - T^2 = w(r),  T = w^0.5 sinh(s), X = +-w^0.5 cosh(s)
    for r0 in np.linspace(1.5 * M * 2, 8 * M * 2, n_lines):
        w = (r0 / (2 * M) - 1) * math.exp(r0 / (2 * M))
        sh = np.linspace(-4, 4, samples)
        for sx in (1, -1):
            out["const_r"].append(_kruskal_to_penrose(math.sqrt(w) * np.sinh(sh), sx * math.sqrt(w) * np.cosh(sh)))
    # constant r < 2M in regions II and III: T^2 - X^2 = -w(r) > 0
    for r0 in np.linspace(0.25 * 2 * M, 0.9 * 2 * M, n_lines):
        w = (r0 / (2 * M) - 1) * math.exp(r0 / (2 * M))            # negative
        sh = np.linspace(-4, 4, samples)
        for st in (1, -1):
            out["const_r"].append(_kruskal_to_penrose(st * math.sqrt(-w) * np.cosh(sh), math.sqrt(-w) * np.sinh(sh)))
    # constant t in regions I and IV: T = X tanh(t/4M)
    xx = np.linspace(0.0, big, 400)
    for t0 in np.linspace(-1.5, 1.5, n_lines) * 4 * M:
        for sx in (1, -1):
            out["const_t"].append(_kruskal_to_penrose(sx * xx * math.tanh(t0 / (4 * M)), sx * xx))
    # radial null rays: U = const and V = const lines
    tt = np.linspace(-big, big, 4001)
    for c in np.linspace(-3, 3, n_lines + 2):
        out["rays"].append(_kruskal_to_penrose((tt + c) / 2, (tt - c) / 2))      # V - U style: T - X = c
        out["rays"].append(_kruskal_to_penrose((tt + c) / 2, (c - tt) / 2))      # T + X = c
    return out


def penrose_supported(metric: GRMetric) -> str | None:
    """'minkowski' / 'schwarzschild' if a closed-form Penrose diagram exists for this metric, else None."""
    if metric.key in ("minkowski", "minkowski_spherical"):
        return "minkowski"
    if metric.key == "schwarzschild":
        return "schwarzschild"
    return None
