"""
Detects a SUVAT-style 1D kinematics setup among a solved problem's
variables and, when enough of the classic quantities (initial velocity,
acceleration, a time span) are numerically resolvable, builds a
position/velocity-vs-time trajectory for ui/results/summary.py's animated
motion diagram (see modules.plotter.build_motion_diagram).

Deliberately does NOT key off model.problem_domain (the LLM's free-text
domain label -- "kinematics", "physics", "motion problem", anything) since
that string isn't a reliable signal on its own. Instead it matches each
variable's own `meaning` field against a small set of substrings (the same
lightweight approach modules.named_formulas.py uses for its domain
keywords) -- "initial velocity", "acceleration", "time", and so on -- which
an LLM extraction consistently writes in something close to that form
regardless of what symbol names it picked (v_i/u/vi, a, t/t_total, ...).

This is intentionally conservative: returning None (skip the animation)
is always safe; a WRONG guess that this is kinematics and building a
trajectory from the wrong variables would actively mislead, which is worse
than the section simply not appearing.
"""
from dataclasses import dataclass

import numpy as np

from modules.equation_engine import ProblemModel, Variable
from modules.verifier import VerificationReport

_INITIAL_VELOCITY_HINTS = ("initial velocity", "starting velocity", "launch velocity")
_FINAL_VELOCITY_HINTS = ("final velocity", "ending velocity", "terminal velocity")
_ACCELERATION_HINTS = ("acceleration",)
_TIME_HINTS = ("time",)
_INITIAL_POSITION_HINTS = ("initial position", "starting position", "initial height", "starting height")
_DISPLACEMENT_HINTS = ("displacement", "distance travel", "distance cover", "position")


@dataclass
class MotionTrajectory:
    t_values: np.ndarray
    x_values: np.ndarray
    v_values: np.ndarray
    x_label: str = "position"
    x_unit: str = ""
    t_unit: str = ""


def _find_by_meaning(model: ProblemModel, hints: tuple[str, ...]) -> Variable | None:
    for v in model.variables:
        if v.is_function:
            continue
        low = v.meaning.lower()
        if any(h in low for h in hints):
            return v
    return None


def build_kinematics_trajectory(model: ProblemModel, report: VerificationReport,
                                   n_points: int = 150) -> MotionTrajectory | None:
    """Returns a MotionTrajectory built from x(t) = x0 + v_i*t + 0.5*a*t^2
    over [0, t_final], or None when the problem doesn't look like 1D
    constant-acceleration kinematics with enough resolvable quantities.
    Combines report.sympy_numeric_answers (this problem's SOLVED targets)
    with each variable's own known_value (this problem's GIVENS) into one
    numeric lookup -- a target variable's number_input default in
    render_variables() is NOT a reliable source here (it defaults to 0.0
    for anything that wasn't already known, regardless of what it solved
    to), so this reads the verified solved answers directly instead.
    """
    a_var = _find_by_meaning(model, _ACCELERATION_HINTS)
    t_var = _find_by_meaning(model, _TIME_HINTS)
    v_i_var = _find_by_meaning(model, _INITIAL_VELOCITY_HINTS)
    v_f_var = _find_by_meaning(model, _FINAL_VELOCITY_HINTS)
    if a_var is None or t_var is None or (v_i_var is None and v_f_var is None):
        return None

    values: dict[str, float] = dict(report.sympy_numeric_answers)
    for v in model.variables:
        if v.known_value is not None:
            values.setdefault(v.symbol, v.known_value)

    def resolve(var: Variable | None) -> float | None:
        return values.get(var.symbol) if var is not None else None

    a = resolve(a_var)
    t_final = resolve(t_var)
    v_i = resolve(v_i_var)
    v_f = resolve(v_f_var)
    if a is None or t_final is None or t_final <= 0:
        return None
    if v_i is None:
        if v_f is None:
            return None
        v_i = v_f - a * t_final  # back it out from v = v_i + a*t

    x0_var = _find_by_meaning(model, _INITIAL_POSITION_HINTS)
    x0 = resolve(x0_var) or 0.0

    ts = np.linspace(0.0, float(t_final), n_points)
    xs = x0 + v_i * ts + 0.5 * a * ts ** 2
    vs = v_i + a * ts

    # Position's unit is NOT the velocity variable's unit (a common bug
    # shape: "m/s" is velocity's unit, not position's) -- prefer an
    # explicitly-named displacement/position variable's own unit; failing
    # that, strip a "/<time_unit>" suffix from the velocity unit as a
    # best-effort guess (the overwhelmingly common case: "m/s" -> "m");
    # leave blank rather than show a wrong unit if neither works cleanly.
    disp_var = _find_by_meaning(model, _DISPLACEMENT_HINTS)
    v_var = v_i_var if v_i_var is not None else v_f_var
    assert v_var is not None  # guaranteed by the a_var/t_var/v_i_var/v_f_var check above
    v_unit = v_var.unit or ""
    t_unit = t_var.unit or ""
    if disp_var is not None and disp_var.unit:
        x_unit = disp_var.unit
    elif t_unit and v_unit.endswith(f"/{t_unit}"):
        x_unit = v_unit[: -len(f"/{t_unit}")]
    else:
        x_unit = ""

    return MotionTrajectory(t_values=ts, x_values=xs, v_values=vs, x_label="position",
                              x_unit=x_unit, t_unit=t_unit)
