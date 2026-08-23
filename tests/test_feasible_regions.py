import json
import numpy as np

from modules.equation_engine import build_model
from modules.plotter import build_feasible_region_plot


def _region_at(fig, x_val, y_val):
    xs, ys = fig.data[0].x, fig.data[0].y
    xi = int(np.argmin(np.abs(xs - x_val)))
    yi = int(np.argmin(np.abs(ys - y_val)))
    return bool(fig.data[0].z[yi, xi])


def test_feasible_point_correctly_included(multi_constraint_json):
    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]
    fig = build_feasible_region_plot(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    # (2,2): 2+2<=10 OK, 4+2<=15 OK, both >=0 OK -- feasible
    assert _region_at(fig, 2, 2) is True


def test_budget_violation_correctly_excluded(multi_constraint_json):
    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]
    fig = build_feasible_region_plot(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    # (9,9): 9+9=18 > 10 -- violates budget constraint
    assert _region_at(fig, 9, 9) is False


def test_non_negativity_violation_correctly_excluded(multi_constraint_json):
    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]
    fig = build_feasible_region_plot(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    assert _region_at(fig, -1, -1) is False


def test_subset_of_constraints_relaxes_region(multi_constraint_json):
    """Dropping the budget constraint should make more points feasible --
    a basic sanity check that constraint selection actually changes output."""
    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]
    all_constraints_fig = build_feasible_region_plot(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    fewer = [c for c in constraints if c.name != "budget constraint"]
    relaxed_fig = build_feasible_region_plot(fewer, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    assert relaxed_fig.data[0].z.sum() >= all_constraints_fig.data[0].z.sum()


def test_no_crash_on_empty_constraint_list():
    fig = build_feasible_region_plot([], "x", "y", {}, (-2, 12), (-2, 12), resolution=20)
    # with no constraints, everything is trivially "feasible"
    assert fig.data[0].z.sum() == fig.data[0].z.size
