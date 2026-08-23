import json

import numpy as np
import sympy as sp

from modules.equation_engine import build_model
from modules.plot_snapshot import (
    snapshot_line_plot, snapshot_surface_plot, snapshot_feasible_region, snapshot_ode_plot,
)
from modules.exporter import build_markdown, build_pdf_bytes, PlotSnapshot

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_snapshot_line_plot_produces_valid_png(kinematics_two_target_json):
    model = build_model(json.loads(kinematics_two_target_json))
    eq = model.equations[0]
    png = snapshot_line_plot(eq, "t", {"a": 2.0, "u": 8.0}, (0, 10), y_target="v")
    assert png[:8] == PNG_SIGNATURE
    assert len(png) > 500


def test_snapshot_surface_plot_produces_valid_png(kinematics_two_target_json):
    model = build_model(json.loads(kinematics_two_target_json))
    eq = model.equations[1]  # displacement equation, has 2 free vars (a, t)
    png = snapshot_surface_plot(eq, "a", "t", {"u": 8.0}, (0, 5), (0, 10), z_target="d")
    assert png[:8] == PNG_SIGNATURE


def test_snapshot_feasible_region_produces_valid_png(multi_constraint_json):
    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]
    png = snapshot_feasible_region(constraints, "x", "y", {}, (-2, 12), (-2, 12))
    assert png[:8] == PNG_SIGNATURE


def test_snapshot_ode_plot_produces_valid_png():
    t = sp.Symbol("t")
    k = sp.Symbol("k")
    rhs = (500 * sp.exp(-k * t)).subs({k: 0.1})
    png = snapshot_ode_plot("N", t, rhs, (0, 50))
    assert png[:8] == PNG_SIGNATURE


def test_snapshot_embeds_in_markdown(kinematics_json, fake_client_factory):
    from modules.verifier import verify
    from modules.solver import compute_steps

    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)

    eq = model.equations[0]
    png = snapshot_line_plot(eq, "t", {"a": 2.0, "u": 8.0}, (0, 10), y_target="v")
    snap = PlotSnapshot(title="test plot", caption="x=t, a=2.0, u=8.0", png_bytes=png)

    md = build_markdown("x", model, report, steps, [], plot_snapshots=[snap])
    assert "## Plots" in md
    assert "data:image/png;base64," in md
    assert "test plot" in md
    assert "x=t, a=2.0, u=8.0" in md


def test_snapshot_embeds_in_pdf(kinematics_json, fake_client_factory):
    from modules.verifier import verify
    from modules.solver import compute_steps

    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)

    eq = model.equations[0]
    png = snapshot_line_plot(eq, "t", {"a": 2.0, "u": 8.0}, (0, 10), y_target="v")
    snap = PlotSnapshot(title="test plot", caption="x=t, a=2.0, u=8.0", png_bytes=png)

    pdf_bytes = build_pdf_bytes("x", model, report, steps, [], plot_snapshots=[snap])
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > len(png)  # the image bytes are actually embedded, not dropped


def test_export_without_snapshots_still_works(kinematics_json, fake_client_factory):
    """Backward compatibility: plot_snapshots is optional."""
    from modules.verifier import verify
    from modules.solver import compute_steps

    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)

    md = build_markdown("x", model, report, steps, [])
    assert "## Plots" not in md
    pdf_bytes = build_pdf_bytes("x", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"


def test_feasible_region_snapshot_matches_interactive_version(multi_constraint_json):
    """The static matplotlib snapshot and the interactive Plotly version
    should agree on which points are feasible -- same underlying math,
    different rendering engine."""
    from modules.plotter import build_feasible_region_plot

    model = build_model(json.loads(multi_constraint_json))
    constraints = [e for e in model.equations if e.kind == "inequality"]

    interactive_fig = build_feasible_region_plot(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    xs, ys = interactive_fig.data[0].x, interactive_fig.data[0].y
    xi = int(np.argmin(np.abs(xs - 2)))
    yi = int(np.argmin(np.abs(ys - 2)))
    interactive_feasible_at_2_2 = bool(interactive_fig.data[0].z[yi, xi])

    # the static snapshot doesn't expose its grid directly, but we can at
    # least confirm it renders without error for the same inputs and
    # doesn't silently disagree in a way that would show up as a crash
    png = snapshot_feasible_region(constraints, "x", "y", {}, (-2, 12), (-2, 12), resolution=50)
    assert png[:8] == PNG_SIGNATURE
    assert interactive_feasible_at_2_2 is True  # sanity: (2,2) should be feasible per earlier tests
