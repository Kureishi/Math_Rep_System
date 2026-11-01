"""
Metrics for the General-Relativity tools (modules/gr_*.py) -- Streamlit-free.

A `GRMetric` is a metric tensor g_ab(x) in named coordinates, with its free parameters (mass M, spin a,
charge Q, ...) and default values for them. Everything in the GR modules -- curvature and field equations
(gr_curvature), geodesics (gr_geodesics), coordinate changes (gr_transforms), light cones and Penrose
diagrams (gr_cones) -- takes one of these, so a metric is parsed and validated once, here.

CONVENTIONS (stated once, relied on everywhere): signature (-,+,+,+) with the time coordinate first;
G = c = 1 unless a module says otherwise; Riemann R^r_{smn} = d_m Gamma^r_{ns} - d_n Gamma^r_{ms} + ... (MTW),
which gives the 2-sphere a POSITIVE Ricci scalar 2/R^2; Einstein's equations G_ab + Lambda g_ab = 8 pi T_ab.

TWO WAYS TO GET A METRIC
  * PRESETS -- Minkowski, Schwarzschild, Reissner-Nordstrom, Kerr, Schwarzschild-de Sitter, flat FRW
    (power law and de Sitter) and Schwarzschild in ingoing Eddington-Finkelstein coordinates. Each carries
    the metadata other modules use (how its coordinates embed in plottable space, which closed-form results
    apply to it, a sample point where the metric is regular).
  * USER TEXT -- a grid of expressions. The text is first walked as a Python syntax tree and refused unless
    it is only numbers, names, + - * / ** and a short list of functions (the same guard the quantum modules
    use), and only then does SymPy see it. Any name that is not a coordinate is a PARAMETER (positive real,
    default 1.0), so `2*M/r` needs no declaration of M.

SYMBOLS. Coordinates are real symbols (the radial coordinate `r` is positive); parameters are positive.
Using fixed assumptions matters: SymPy treats Symbol('r') and Symbol('r', positive=True) as different, and a
`log(r - 2*M)` only simplifies sensibly when it knows r - 2M can be positive. Everything that parses an
expression against a metric builds its symbols from the metric's own, by name, so they always match.
"""
import ast
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from modules.quantum_common import _CONSTANTS, _FUNCTIONS, ExpressionError, check_expression_syntax

# Symbolic general relativity is much heavier than the rest of the app (a full Kerr curvature tensor is ~10 s),
# so it gets its own, longer bound than settings.computation_timeout_seconds.
GR_TIMEOUT = float(os.getenv("GR_TIMEOUT_SECONDS", "120"))
MAX_DIM = 5


class GRInputError(ValueError):
    """The input is not a usable metric / expression (the message says why, in terms the user can act on)."""


def coordinate_symbol(name: str) -> sp.Symbol:
    """The symbol for a coordinate called `name`: real, and positive for the radial coordinate `r`."""
    return sp.Symbol(name, positive=True) if name == "r" else sp.Symbol(name, real=True)


def parameter_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, positive=True)


@dataclass(eq=False)
class GRMetric:
    key: str
    name: str
    coords: tuple[sp.Symbol, ...]
    g: sp.Matrix
    params: dict[str, float] = field(default_factory=dict)      # parameter name -> default value
    description: str = ""
    # how coordinates map to plottable space: "spherical" (t, r, theta, phi), "oblate" (Boyer-Lindquist),
    # "cartesian" (t, x, y, z), "comoving" (FRW, plotted at fixed comoving radius), or "none"
    embedding: str = "none"
    # which closed-form results apply: "minkowski", "schwarzschild", "kerr", "rn", "static" (f(r) metrics),
    # "frw", or "" for a user metric
    family: str = ""
    sample: dict[str, float] = field(default_factory=dict)       # a coordinate point where the metric is regular
    # f(r) for static spherically symmetric metrics ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2
    f_of_r: sp.Expr | None = None

    @property
    def dim(self) -> int:
        return len(self.coords)

    @property
    def coord_names(self) -> list[str]:
        return [str(c) for c in self.coords]

    @property
    def param_symbols(self) -> dict[str, sp.Symbol]:
        return {name: parameter_symbol(name) for name in sorted(self.params)}

    def symbols_by_name(self) -> dict[str, sp.Symbol]:
        out: dict[str, sp.Symbol] = {str(c): c for c in self.coords}
        out.update(self.param_symbols)
        return out

    def param_subs(self, values: dict[str, float] | None = None) -> dict[sp.Symbol, float]:
        merged = dict(self.params)
        merged.update(values or {})
        return {sp.Symbol(k, positive=True): float(v) for k, v in merged.items() if k in self.params}

    def sample_point(self) -> list[float]:
        defaults = [0.7, 3.1, 1.1, 0.9, 1.3]
        return [float(self.sample.get(str(c), defaults[i % len(defaults)])) for i, c in enumerate(self.coords)]

    def parse(self, text: str) -> sp.Expr:
        """An expression in this metric's coordinates and parameters (same guard as the user-metric grid)."""
        return parse_in_symbols(text, self.symbols_by_name())


def parse_in_symbols(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    try:
        check_expression_syntax(text, symbols)
        local: dict = dict(symbols)
        local.update(_CONSTANTS)
        local.update(_FUNCTIONS)
        return parse_expr(text.strip().replace("^", "**"), local_dict=local, transformations=standard_transformations)
    except ExpressionError as e:
        raise GRInputError(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise GRInputError(f"Could not interpret '{text}': {e}") from None


def _names_in(text: str) -> set[str]:
    """Names used as variables (not function names or constants) in an expression string."""
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except SyntaxError:
        return set()
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in called and n.id not in _CONSTANTS}


def parse_user_metric(rows: list[list[str]], coord_names: list[str], param_defaults: dict[str, float] | None = None,
                      name: str = "User metric", embedding: str = "none") -> GRMetric:
    """A GRMetric from a grid of expression strings (see the module docstring for what is accepted)."""
    names = [c.strip() for c in coord_names if c.strip()]
    if not 2 <= len(names) <= MAX_DIM:
        raise GRInputError(f"Give between 2 and {MAX_DIM} coordinate names (got {len(names)}).")
    if len(set(names)) != len(names):
        raise GRInputError("Coordinate names must be distinct.")
    for nm in names:
        if not nm.isidentifier() or nm in _FUNCTIONS or nm in _CONSTANTS:
            raise GRInputError(f"'{nm}' cannot be used as a coordinate name.")
    n = len(names)
    if len(rows) != n or any(len(r) != n for r in rows):
        raise GRInputError(f"The metric must be a {n} x {n} grid, one row and column per coordinate.")
    used: set[str] = set()
    for row in rows:
        for cell in row:
            if not str(cell).strip():
                raise GRInputError("Every cell of the metric needs an entry (use 0 for zero).")
            used |= _names_in(str(cell))
    param_names = sorted(used - set(names))
    coords = tuple(coordinate_symbol(nm) for nm in names)
    symbols: dict[str, sp.Symbol] = {str(c): c for c in coords}
    symbols.update({p: parameter_symbol(p) for p in param_names})
    g = sp.Matrix(n, n, lambda i, j: parse_in_symbols(str(rows[i][j]), symbols))
    for i in range(n):
        for j in range(i + 1, n):
            if sp.simplify(g[i, j] - g[j, i]) != 0:
                raise GRInputError(f"The metric must be symmetric, but g[{i}][{j}] and g[{j}][{i}] differ.")
    defaults = {p: 1.0 for p in param_names}
    defaults.update({k: float(v) for k, v in (param_defaults or {}).items() if k in defaults})
    return GRMetric(key="user", name=name, coords=coords, g=g, params=defaults, embedding=embedding,
                    description="Metric entered by the user.")


# ------------------------------------------------------------------------------------------ presets

def _c(name: str) -> sp.Symbol:
    return coordinate_symbol(name)


def _p(name: str) -> sp.Symbol:
    return parameter_symbol(name)


def _static_spherical(key: str, name: str, f: sp.Expr, params: dict[str, float], description: str,
                      family: str, sample_r: float) -> GRMetric:
    t, r, th, ph = _c("t"), _c("r"), _c("theta"), _c("phi")
    g = sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2)
    return GRMetric(key, name, (t, r, th, ph), g, params, description, embedding="spherical", family=family,
                    sample={"t": 0.5, "r": sample_r, "theta": 1.1, "phi": 0.7}, f_of_r=f)


def minkowski() -> GRMetric:
    t, x, y, z = _c("t"), _c("x"), _c("y"), _c("z")
    return GRMetric("minkowski", "Minkowski (Cartesian)", (t, x, y, z), sp.diag(-1, 1, 1, 1), {},
                    "Flat spacetime in inertial coordinates.", embedding="cartesian", family="minkowski",
                    sample={"t": 0.3, "x": 0.5, "y": 0.8, "z": 1.2})


def minkowski_spherical() -> GRMetric:
    return _static_spherical("minkowski_spherical", "Minkowski (spherical)", sp.Integer(1), {},
                             "Flat spacetime in spherical coordinates (flat, but the metric is not constant).",
                             "minkowski", 2.0)


def schwarzschild() -> GRMetric:
    M, r = _p("M"), _c("r")
    return _static_spherical("schwarzschild", "Schwarzschild", 1 - 2 * M / r, {"M": 1.0},
                             "Vacuum outside a spherical mass M: horizon at r = 2M, curvature singularity at r = 0.",
                             "schwarzschild", 8.0)


def reissner_nordstrom() -> GRMetric:
    M, Q, r = _p("M"), _p("Q"), _c("r")
    return _static_spherical("reissner_nordstrom", "Reissner-Nordstrom", 1 - 2 * M / r + Q ** 2 / r ** 2,
                             {"M": 1.0, "Q": 0.5},
                             "A charged spherical mass (Gaussian units, G = c = 1): sourced by its own electric field.",
                             "rn", 8.0)


def schwarzschild_de_sitter() -> GRMetric:
    M, Lam, r = _p("M"), _p("Lambda"), _c("r")
    return _static_spherical("schwarzschild_de_sitter", "Schwarzschild-de Sitter", 1 - 2 * M / r - Lam * r ** 2 / 3,
                             {"M": 1.0, "Lambda": 0.01},
                             "A mass in a universe with cosmological constant: black-hole and cosmological horizons.",
                             "static", 8.0)


def kerr() -> GRMetric:
    t, r, th, ph = _c("t"), _c("r"), _c("theta"), _c("phi")
    M, a = _p("M"), _p("a")
    sig = r ** 2 + a ** 2 * sp.cos(th) ** 2
    dl = r ** 2 - 2 * M * r + a ** 2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * M * r / sig)
    g[0, 3] = g[3, 0] = -2 * M * a * r * sp.sin(th) ** 2 / sig
    g[1, 1] = sig / dl
    g[2, 2] = sig
    g[3, 3] = (r ** 2 + a ** 2 + 2 * M * a ** 2 * r * sp.sin(th) ** 2 / sig) * sp.sin(th) ** 2
    return GRMetric("kerr", "Kerr (Boyer-Lindquist)", (t, r, th, ph), g, {"M": 1.0, "a": 0.7},
                    "A rotating black hole: spin a (|a| <= M), ring singularity, ergosphere.",
                    embedding="oblate", family="kerr", sample={"t": 0.5, "r": 6.0, "theta": 1.1, "phi": 0.7})


def frw_power_law() -> GRMetric:
    t, r, th, ph = _c("t"), _c("r"), _c("theta"), _c("phi")
    p = _p("p")
    a = t ** p
    g = sp.diag(-1, a ** 2, a ** 2 * r ** 2, a ** 2 * r ** 2 * sp.sin(th) ** 2)
    return GRMetric("frw_power", "Flat FRW, a(t) = t^p", (t, r, th, ph), g, {"p": 2 / 3},
                    "Spatially flat expanding universe. p = 2/3 is matter-dominated, p = 1/2 radiation-dominated.",
                    embedding="comoving", family="frw", sample={"t": 1.5, "r": 1.0, "theta": 1.1, "phi": 0.7})


def de_sitter_flat() -> GRMetric:
    t, r, th, ph = _c("t"), _c("r"), _c("theta"), _c("phi")
    H = _p("H")
    a = sp.exp(H * t)
    g = sp.diag(-1, a ** 2, a ** 2 * r ** 2, a ** 2 * r ** 2 * sp.sin(th) ** 2)
    return GRMetric("de_sitter", "de Sitter (flat slicing)", (t, r, th, ph), g, {"H": 1.0},
                    "Exponentially expanding vacuum universe with Lambda = 3 H^2.",
                    embedding="comoving", family="frw", sample={"t": 0.5, "r": 1.0, "theta": 1.1, "phi": 0.7})


def schwarzschild_ingoing_ef() -> GRMetric:
    v, r, th, ph = _c("v"), _c("r"), _c("theta"), _c("phi")
    M = _p("M")
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * M / r)
    g[0, 1] = g[1, 0] = 1
    g[2, 2] = r ** 2
    g[3, 3] = r ** 2 * sp.sin(th) ** 2
    return GRMetric("schwarzschild_ef", "Schwarzschild (ingoing Eddington-Finkelstein)", (v, r, th, ph), g,
                    {"M": 1.0}, "Schwarzschild in coordinates that are regular across the future horizon.",
                    embedding="spherical", family="", sample={"v": 0.5, "r": 8.0, "theta": 1.1, "phi": 0.7})


PRESETS: dict[str, Callable[[], GRMetric]] = {
    "minkowski": minkowski,
    "minkowski_spherical": minkowski_spherical,
    "schwarzschild": schwarzschild,
    "reissner_nordstrom": reissner_nordstrom,
    "kerr": kerr,
    "schwarzschild_de_sitter": schwarzschild_de_sitter,
    "frw_power": frw_power_law,
    "de_sitter": de_sitter_flat,
    "schwarzschild_ef": schwarzschild_ingoing_ef,
}


def preset(key: str) -> GRMetric:
    if key not in PRESETS:
        raise GRInputError(f"Unknown preset '{key}'. Available: {', '.join(PRESETS)}.")
    return PRESETS[key]()


def metric_rows(metric: GRMetric) -> list[list[str]]:
    """The metric as a grid of strings (what the user-metric parser accepts), for loading a preset into the
    text grid. Round-trips: parse_user_metric(metric_rows(m), m.coord_names) is the same tensor."""
    return [[str(metric.g[i, j]) for j in range(metric.dim)] for i in range(metric.dim)]
