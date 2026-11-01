"""
Reproducible research output: a run becomes a runnable notebook / script with its parameters and seeds -- Streamlit-free.

modules/notebook_export.py turns a SOLVED PROBLEM into a notebook of its steps and closed-form functions. This module does the same for a
COMPUTATIONAL RUN of the app's tools (a lattice scan, an inspiral, a convergence study, an integrator comparison, a dimensional analysis, ...):

    RunRecord  = recipe + parameters + seed + library versions + the numbers the run produced
    notebook   = provenance table, then CODE CELLS that re-run exactly that computation from the parameters and the seed, then a
                 REPRODUCTION CHECK that compares the fresh numbers with the recorded ones and fails loudly if they differ.

How it stays honest:
  * the recorded numbers come from executing the SAME code cells the notebook contains (`make_record` execs the recipe code), never from
    a separate path, so notebook and record cannot disagree;
  * every parameter is a plain number / string / list literal (anything else is refused), so the generated code cannot carry anything but data;
  * a seed is always concrete: `seed=None` draws a fresh one, which is stored (the project's seed convention, modules/sm_common.py);
  * the notebook declares the versions of Python, numpy, scipy, sympy and mpmath it was made with, and the check tolerance, because
    bit-identical floating point is only promised on the same libraries;
  * the tests EXECUTE every recipe's exported notebook in a fresh namespace and require the reproduction check to pass, and edit a recorded
    value to prove the check can fail.
The notebook calls the app's own modules (`from modules import ...`): run it from the project root, or set MATH_REP_SYSTEM to its path.
"""
import ast
import datetime
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, field

from modules.cc_common import CrossCuttingError
from modules.notebook_export import _code_cell, _markdown_cell
from modules.sm_common import resolve_seed

_LIBS = ("numpy", "scipy", "sympy", "mpmath")


@dataclass
class Recipe:
    key: str
    title: str
    description: str
    defaults: dict
    code: str                       # runs with PARAMS and SEED defined; must define `summary` (dict[str, float | str])
    uses_seed: bool = True
    tolerance: float = 1e-9


RECIPES: dict[str, Recipe] = {r.key: r for r in [
    Recipe("partition", "Partition function of a two-level system",
           "Z, F, U, S and C of a two-level system at chosen temperatures, with the thermodynamic self-checks.",
           {"gap": 1.0, "g_ground": 1, "g_excited": 1, "kT": [0.25, 0.5, 1.0, 2.0]},
           '''from modules import sm_partition as sp_
spec = sp_.two_level(PARAMS["gap"], PARAMS["g_ground"], PARAMS["g_excited"])
th = sp_.thermodynamics(spec, PARAMS["kT"])
checks = sp_.verify_thermo(spec, th)
summary = {"checks_passed": float(checks.passed)}
for i, t in enumerate(PARAMS["kT"]):
    summary[f"U@{t:g}"] = float(th.U[i]); summary[f"S@{t:g}"] = float(th.S[i]); summary[f"C@{t:g}"] = float(th.C[i])''', uses_seed=False),
    Recipe("lattice_scan", "Metropolis temperature scan of a lattice model",
           "Energy, order parameter and Binder cumulant against temperature (Ising, Potts or XY), annealed from high to low T.",
           {"model": "ising", "L": 8, "h": 0.0, "q": 3, "temps": [1.8, 2.27, 2.8], "n_therm": 200, "n_meas": 1500},
           '''from modules import sm_lattice as sl
model = sl.LatticeModel(PARAMS["model"], PARAMS["L"], 1.0, PARAMS["h"], PARAMS["q"])
scan = sl.scan_temperatures(model, PARAMS["temps"], PARAMS["n_therm"], PARAMS["n_meas"], seed=SEED)
summary = {}
for t, p in zip(scan.temps, scan.points):
    summary[f"e@{t:g}"] = float(p.e); summary[f"m@{t:g}"] = float(p.m); summary[f"U4@{t:g}"] = float(p.U4); summary[f"tau_m@{t:g}"] = float(p.tau_m)'''),
    Recipe("einstein", "Fluctuation-dissipation (Einstein relation) measured",
           "Diffusion constant from a force-free ensemble and mobility from the drift under a force; D/mu must equal kT.",
           {"kT": 1.0, "gamma": 2.0, "force": 1.0, "dt": 0.02, "t_max": 10.0, "n_paths": 4000, "mass": 1.0},
           '''from modules import sm_stochastic as ss
res = ss.einstein_relation(PARAMS["kT"], PARAMS["gamma"], PARAMS["force"], PARAMS["dt"], PARAMS["t_max"], PARAMS["n_paths"], PARAMS["mass"], SEED)
summary = {"D": float(res.D), "mobility": float(res.mobility), "kT_estimate": float(res.kT_estimate), "checks_passed": float(res.checks.passed)}'''),
    Recipe("random_matrix", "Random-matrix level statistics",
           "Eigenvalue density against the semicircle and nearest-neighbour spacings against the Wigner surmises.",
           {"kind": "GOE", "N": 120, "n_samples": 20},
           '''from modules import sm_randommatrix as rm
rep = rm.analyse(PARAMS["kind"], PARAMS["N"], PARAMS["n_samples"], SEED)
summary = {"ratio": float(rep.ratio), "nearest": rep.nearest, "checks_passed": float(rep.checks.passed)}
for k, v in rep.distances.items():
    summary[f"KS_{k}"] = float(v)'''),
    Recipe("geodesic_precession", "Perihelion precession in Schwarzschild spacetime",
           "A bound orbit integrated as a geodesic, its periapsis advance compared with the exact elliptic-integral result.",
           {"M": 1.0, "r_a": 100.0, "r_p": 40.0, "orbits": 3},
           '''import math
from modules import gr_geodesics as gg, gr_potential as gp
from modules.gr_metrics import preset
M, ra, rp = PARAMS["M"], PARAMS["r_a"], PARAMS["r_p"]
E, L = gp.schwarzschild_orbit_from_turning_points(M, ra, rp)
system = gg.build_system(preset("schwarzschild"), {"M": M})
x0, u0 = gg.initial_from_conserved(system, ra, E, L, "timelike", inward=True)
period = 2 * math.pi * math.sqrt(((ra + rp) / 2) ** 3 / M)
res = gg.integrate_geodesic(system, x0, u0, (PARAMS["orbits"] + 0.2) * period)
adv = gg.periapsis_advance(res)
summary = {"E": float(E), "L": float(L), "advance_measured": float(sum(adv) / len(adv)), "advance_exact": float(gp.schwarzschild_precession(M, ra, rp)),
           "checks_passed": float(res.checks.passed)}''', uses_seed=False),
    Recipe("integrators", "Symplectic against non-symplectic integrators",
           "Energy drift of leapfrog, Yoshida, implicit midpoint, RK4 and RK45 on the pendulum over many periods.",
           {"theta0": 1.0, "t_end": 400.0, "dt": 0.1},
           '''import numpy as np
from modules import cc_conservation as cc
f = cc.HamiltonianFlow(1, lambda q, p: 0.5 * p[0] ** 2 - np.cos(q[0]), lambda q, p: np.array([np.sin(q[0])]), lambda q, p: np.array([p[0]]), True)
cmp = cc.compare_integrators(f, np.array([PARAMS["theta0"]]), np.array([0.0]), PARAMS["t_end"], PARAMS["dt"])
summary = {}
for r in cmp.runs:
    summary[f"max_rel_drift[{r.method}]"] = float(r.drift.max_rel); summary[f"kind[{r.method}]"] = r.drift.kind''', uses_seed=False),
    Recipe("convergence", "Order of convergence of an ODE method",
           "The observed order of Euler, Heun, midpoint or RK4 on a harmonic oscillator, from successive refinements.",
           {"method": "rk4", "t_end": 3.0, "steps0": 8, "levels": 6},
           '''import math
import numpy as np
from modules import cc_convergence as cv
r = cv.ode_convergence(lambda t, y: np.array([y[1], -y[0]]), [0.0, 1.0], PARAMS["t_end"], PARAMS["method"], PARAMS["steps0"], PARAMS["levels"],
                       lambda t: np.array([math.sin(t), math.cos(t)]))
summary = {"observed_order": float(r.observed_order), "gci": float(r.gci), "checks_passed": float(r.checks.passed)}''', uses_seed=False),
    Recipe("nondim", "Non-dimensionalisation of a damped, driven oscillator",
           "The dimensionless form, its scales and groups, and the numerical equivalence check.",
           {"m": 2.0, "c": 0.6, "k": 8.0, "F": 3.0, "w": 1.7, "x0": 0.5},
           '''from modules import cc_nondim as nd
pv = {k: PARAMS[k] for k in ("m", "c", "k", "F", "w")}
res = nd.nondimensionalize("m*x_tt + c*x_t + k*x = F*sin(w*t)", "x", ["t"],
                           {"x": "m", "t": "s", "m": "kg", "c": "kg/s", "k": "kg/s^2", "F": "N", "w": "1/s"}, pv, {"x": PARAMS["x0"], "x_t": 0.0})
summary = {"equation": str(res.dimensionless), "checks_passed": float(res.checks.passed)}
for k, v in nd.groups_numeric(res, pv).items():
    summary[k] = float(v)''', uses_seed=False),
    Recipe("crossval", "Symbolic against numerical: an integral",
           "A closed-form definite integral and scipy's quadrature of the same integrand.",
           {"expr": "exp(-x^2)*cos(a*x)", "lower": "-oo", "upper": "oo", "a": 1.5},
           '''from modules import cc_crossval as cx
cv = cx.validate_integral(PARAMS["expr"], "x", PARAMS["lower"], PARAMS["upper"], {"a": PARAMS["a"]})
summary = {"symbolic": float(cv.symbolic_value), "numeric": float(cv.numeric_value), "status": cv.status}''', uses_seed=False),
    Recipe("limit", "A limiting-case check",
           "A physical formula against its classical limit: the expansion, the leading term and the measured rate of convergence.",
           {"key": "rel_ke"},
           '''from modules import cc_limits as cl
rep = cl.check_preset(PARAMS["key"])
summary = {"expansion": rep.series_text, "predicted_exponent": float(rep.predicted_exponent or 0.0), "observed_exponent": float(rep.observed_exponent or 0.0),
           "checks_passed": float(rep.checks.passed)}''', uses_seed=False),
]}


@dataclass
class RunRecord:
    recipe: str
    params: dict
    seed: int | None
    summary: dict
    versions: dict
    created: str
    narrative: str = ""
    tolerance: float = 1e-9

    @property
    def title(self) -> str:
        return RECIPES[self.recipe].title


def _check_literal(value: object, path: str = "parameter") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        if isinstance(value, float) and value != value:
            raise CrossCuttingError(f"{path} is NaN.")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_literal(v, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise CrossCuttingError(f"{path} has a non-string key.")
            _check_literal(v, f"{path}.{k}")
        return
    raise CrossCuttingError(f"{path} must be a plain number, string, list or dict (got {type(value).__name__}).")


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in _LIBS:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = "unknown"
    return out


def _execute(code: str, params: dict, seed: int | None) -> dict:
    ns: dict = {"PARAMS": params, "SEED": seed, "__name__": "__repro__"}
    exec(compile(code, "<recipe>", "exec"), ns)                                    # noqa: S102 - the code is a fixed template; only literals vary
    return ns


def make_record(key: str, params: dict | None = None, seed: int | None = None, narrative: str = "") -> RunRecord:
    """Run the recipe once with the given parameters and seed and record what it produced."""
    if key not in RECIPES:
        raise CrossCuttingError(f"Unknown recipe '{key}'. Available: {', '.join(RECIPES)}.")
    r = RECIPES[key]
    merged = {**r.defaults, **(params or {})}
    unknown = set(merged) - set(r.defaults)
    if unknown:
        raise CrossCuttingError(f"Unknown parameter(s) for '{key}': {', '.join(sorted(unknown))}. Known: {', '.join(r.defaults)}.")
    for k, v in merged.items():
        _check_literal(v, k)
        if isinstance(r.defaults[k], (int, float)) and not isinstance(r.defaults[k], bool) and not isinstance(v, (int, float)):
            raise CrossCuttingError(f"Parameter '{k}' must be a number.")
    used_seed = resolve_seed(seed) if r.uses_seed else None
    ns = _execute(r.code, merged, used_seed)
    summary = ns.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise CrossCuttingError("The recipe produced no summary.")
    clean = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)) for k, v in summary.items()}
    return RunRecord(key, merged, used_seed, clean, _versions(), datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), narrative, r.tolerance)


# ------------------------------------------------------------------------------------------ building the notebook

def _cells(rec: RunRecord) -> list[tuple[str, str]]:
    """[(kind, text)] with kind 'md' or 'code'."""
    r = RECIPES[rec.recipe]
    vers = ", ".join(f"{k} {v}" for k, v in rec.versions.items())
    md = [f"# {r.title}", "", r.description, ""]
    if rec.narrative.strip():
        md += [rec.narrative.strip(), ""]
    md += ["| | |", "|---|---|", f"| recipe | `{rec.recipe}` |", f"| seed | `{rec.seed}` |" if rec.seed is not None else "| seed | not used (deterministic) |",
           f"| created | {rec.created} |", f"| libraries | {vers} |", f"| reproduction tolerance | {rec.tolerance:g} (relative) |", "",
           "Run this notebook from the project root (or set the environment variable `MATH_REP_SYSTEM` to its path). The last cell re-runs the "
           "computation and compares it with the numbers recorded when the notebook was exported; it raises an error if they differ.", ""]
    setup = ("import os, sys\nROOT = os.environ.get(\"MATH_REP_SYSTEM\", os.getcwd())\nsys.path.insert(0, ROOT)\n"
             "import numpy as np  # noqa: F401")
    params = f"PARAMS = {json.dumps(rec.params, indent=4, sort_keys=True)}\nSEED = {rec.seed!r}"
    expected = f"EXPECTED = {json.dumps(rec.summary, indent=4, sort_keys=True)}\nTOLERANCE = {rec.tolerance!r}"
    check = '''bad = []
for name, want in EXPECTED.items():
    got = summary.get(name)
    if isinstance(want, str):
        if got != want:
            bad.append((name, got, want))
    elif got is None or abs(got - want) > TOLERANCE * (1.0 + abs(want)):
        bad.append((name, got, want))
assert not bad, "NOT REPRODUCED: " + "; ".join(f"{n}: got {g!r}, recorded {w!r}" for n, g, w in bad)
print(f"reproduced {len(EXPECTED)} recorded values (relative tolerance {TOLERANCE:g})")'''
    return [("md", "\n".join(md)), ("code", setup), ("code", params), ("code", r.code), ("md", "## Reproduction check\n\nRecorded values, then the comparison."),
            ("code", expected), ("code", check)]


def build_research_notebook(rec: RunRecord) -> str:
    cells = [_markdown_cell(t) if k == "md" else _code_cell(t) for k, t in _cells(rec)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3"},
                                       "math_rep_system": {"recipe": rec.recipe, "seed": rec.seed, "versions": rec.versions}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1)


def build_script(rec: RunRecord) -> str:
    """The same cells as a plain Python script (markdown cells become comments)."""
    out = []
    for kind, text in _cells(rec):
        out.append("\n".join("# " + ln if ln.strip() else "#" for ln in text.splitlines()) if kind == "md" else text)
    return "\n\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------ verifying a notebook

def notebook_code(nb_json: str) -> str:
    try:
        nb = json.loads(nb_json)
        return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    except (ValueError, KeyError, TypeError) as exc:
        raise CrossCuttingError(f"Not a valid notebook: {exc}") from None


def verify_notebook(nb_json: str) -> tuple[bool, str]:
    """Execute all code cells of an exported notebook in a fresh namespace; (reproduced?, message). The current directory must be the project root."""
    code = notebook_code(nb_json)
    ns: dict = {"__name__": "__repro__"}
    old_path = list(sys.path)
    try:
        exec(compile(code, "<notebook>", "exec"), ns)                              # noqa: S102
        return True, "all recorded values were reproduced"
    except AssertionError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"the notebook failed to run: {type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = old_path


def literal_params_ok(source: str) -> bool:
    """True if `PARAMS = ...` in a code cell parses as a pure literal (used by the tests to show the generated code carries data only)."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PARAMS" for t in node.targets):
            try:
                ast.literal_eval(node.value)
                return True
            except ValueError:
                return False
    return False

