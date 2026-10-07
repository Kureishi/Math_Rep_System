"""Shared builders for the report/export tests."""
import json

from modules.equation_engine import build_model
from modules.report_content import PlotSnapshot, ReportExtras
from modules.report_export import ExportContext
from modules.solve_compare import deterministic_report
from modules.solver import compute_steps
from tests.conftest import KINEMATICS_TWO_TARGET_JSON

def _make_png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 60, 60)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _make_png()      # a small real image: Word and PowerPoint read its size


def kinematics_ctx(**kw) -> ExportContext:
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = deterministic_report(model)
    return ExportContext(kw.pop("problem_text", "A car accelerates from 8 to 20 m/s in 6 s."), model, report,
                         compute_steps(model), **kw)


def linear_system_ctx() -> ExportContext:
    model = build_model({
        "problem_domain": "circuit", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
                      {"symbol": "y", "meaning": "y", "known_value": None, "unit": None}],
        "equations": [{"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
                      {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""}],
        "solve_for": ["x", "y"], "assumptions": []})
    return ExportContext("a linear system", model, deterministic_report(model), compute_steps(model))


def snapshot(title="A plot", caption="what it shows", figure_json="") -> PlotSnapshot:
    return PlotSnapshot(title, caption, PNG, figure_json)


def full_extras():
    from modules.report_content import FollowupItem, TutorItem
    return ReportExtras(
        followups=[FollowupItem("what if t doubles?", "what_if", "a becomes 1"), FollowupItem("why?", "conceptual", "because")],
        tutor=[TutorItem("a", "2", True, "Correct!", 3, 7)],
        monte_carlo=[{"target": "a", "n": 1000, "n_failed": 0, "seed": 42, "mean": 2.0, "std": 0.07, "p5": 1.9,
                      "p95": 2.1, "inputs": ["u ~ N(8, 0.5)"], "unit": "m/s^2", "samples": [1.9, 2.0, 2.1, 2.0]}],
        provenance={"reasoning_model": "test-model", "app_version": "9.9", "captured_at": "2026-01-01 00:00",
                    "temperature_extraction": 0.1, "temperature_narration": 0.4, "numeric_tolerance": 1e-6,
                    "cross_check_tolerance": 0.02, "max_verification_retries": 2, "computation_timeout_seconds": 10,
                    "python": "3.12", "libraries": {"sympy": "1.14.0"}, "secondary_model": ""},
        now={"captured_at": "2026-02-02 00:00", "app_version": "9.9"})
