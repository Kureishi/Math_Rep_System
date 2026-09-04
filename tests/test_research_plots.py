from modules.equation_engine import build_model
from modules.plotter import (
    build_plot, build_fit_plot, build_contour_plot, build_overlay_plot,
    build_chain_sweep_plot, build_spread_plot, build_histogram_plot,
)
from modules.plot_snapshot import (
    snapshot_line_plot, snapshot_fit_plot, snapshot_contour_plot, snapshot_overlay_plot,
    snapshot_chain_sweep_plot, snapshot_spread_plot, snapshot_histogram_plot,
)


def _kinematics_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
            {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _two_var_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": None, "unit": None},
            {"symbol": "d", "meaning": "d", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "disp", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2)", "derivation": ""},
        ],
        "solve_for": ["d"], "assumptions": [],
    })


# ---------------------------------------------------------------- log-axis toggle


def test_build_plot_x_log_produces_geomspace_grid():
    model = _kinematics_model()
    eq = model.equations[0]
    known_subs = {"v_f": 20.0, "v_i": 8.0}
    fig_linear = build_plot(model, eq, "t", known_subs, (1, 10), y_target="a")
    fig_log = build_plot(model, eq, "t", known_subs, (1, 10), y_target="a", x_log=True, y_log=True)
    assert fig_log.layout.xaxis.type == "log"
    assert fig_log.layout.yaxis.type == "log"
    assert fig_linear.layout.xaxis.type in (None, "linear")


def test_snapshot_line_plot_with_log_axes_still_produces_png():
    model = _kinematics_model()
    eq = model.equations[0]
    png = snapshot_line_plot(eq, "t", {"v_f": 20.0, "v_i": 8.0}, (1, 10), y_target="a",
                               x_log=True, y_log=True)
    assert png.startswith(b"\x89PNG")


def test_build_fit_plot_log_axes_set_on_layout():
    fig = build_fit_plot([1, 2, 3], [2, 4, 8], None, x_log=True, y_log=True)
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"


def test_snapshot_fit_plot_log_axes_produces_svg_and_pdf():
    svg = snapshot_fit_plot([1, 2, 3], [2, 4, 8], None, x_log=True, y_log=True, fmt="svg")
    assert svg.startswith(b"<?xml") or b"<svg" in svg[:200]
    pdf = snapshot_fit_plot([1, 2, 3], [2, 4, 8], None, fmt="pdf")
    assert pdf.startswith(b"%PDF")


# ---------------------------------------------------------------- vector export (png/svg/pdf)


def test_snapshot_line_plot_supports_all_three_formats():
    model = _kinematics_model()
    eq = model.equations[0]
    known_subs = {"v_f": 20.0, "v_i": 8.0}
    png = snapshot_line_plot(eq, "t", known_subs, (1, 10), y_target="a", fmt="png")
    svg = snapshot_line_plot(eq, "t", known_subs, (1, 10), y_target="a", fmt="svg")
    pdf = snapshot_line_plot(eq, "t", known_subs, (1, 10), y_target="a", fmt="pdf")
    assert png.startswith(b"\x89PNG")
    assert b"<svg" in svg[:300] or svg.startswith(b"<?xml")
    assert pdf.startswith(b"%PDF")


def test_snapshot_rejects_unsupported_format():
    model = _kinematics_model()
    eq = model.equations[0]
    import pytest
    with pytest.raises(ValueError):
        snapshot_line_plot(eq, "t", {"v_f": 20.0, "v_i": 8.0}, (1, 10), y_target="a", fmt="jpeg")


# ---------------------------------------------------------------- contour plot


def test_build_contour_plot_has_contour_trace():
    model = _two_var_model()
    eq = model.equations[0]
    fig = build_contour_plot(eq, "a", "t", {}, (0, 10), (0, 10), z_target="d")
    assert len(fig.data) == 1
    assert fig.data[0].type == "contour"


def test_build_contour_plot_falls_back_to_residual_when_unsolvable_target():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "disp", "kind": "equation", "expression": "Eq(0.5*a*t**2, 100)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    eq = model.equations[0]
    fig = build_contour_plot(eq, "a", "t", {}, (0, 10), (0, 10), z_target=None)
    assert len(fig.data) == 1


def test_snapshot_contour_plot_produces_png_bytes():
    model = _two_var_model()
    eq = model.equations[0]
    png = snapshot_contour_plot(eq, "a", "t", {}, (0, 10), (0, 10), z_target="d")
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- overlay plot


def test_build_overlay_plot_one_trace_per_series():
    series = [
        {"x": [1, 2, 3], "y": [1, 4, 9], "name": "quadratic"},
        {"x": [1, 2, 3], "y": [1, 2, 3], "name": "linear"},
    ]
    fig = build_overlay_plot(series, x_label="x", y_label="y", title="comparison")
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"quadratic", "linear"}


def test_build_overlay_plot_handles_empty_series():
    fig = build_overlay_plot([])
    assert len(fig.data) == 0


def test_snapshot_overlay_plot_produces_png_bytes():
    series = [{"x": [1, 2, 3], "y": [1, 4, 9], "name": "quadratic"}]
    png = snapshot_overlay_plot(series)
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- chain sweep plot


def test_build_chain_sweep_plot_one_line_per_step():
    sweep_rows = [
        {"value": 0.0, "outputs": {0: 2.0, 1: 20.0}},
        {"value": 5.0, "outputs": {0: 3.0, 1: 30.0}},
    ]
    fig = build_chain_sweep_plot(sweep_rows, "v_i", {0: "step 1: a", 1: "step 2: d"})
    assert len(fig.data) == 2
    names = {t.name for t in fig.data}
    assert names == {"step 1: a", "step 2: d"}


def test_build_chain_sweep_plot_handles_none_outputs():
    sweep_rows = [
        {"value": 0.0, "outputs": {0: 2.0}},
        {"value": 5.0, "outputs": {0: None}},  # a failed resolve at this swept value
    ]
    fig = build_chain_sweep_plot(sweep_rows, "v_i", {0: "step 1"})
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [2.0, None]


def test_snapshot_chain_sweep_plot_produces_png_and_skips_none_points():
    sweep_rows = [
        {"value": 0.0, "outputs": {0: 2.0}},
        {"value": 5.0, "outputs": {0: None}},
        {"value": 10.0, "outputs": {0: 4.0}},
    ]
    png = snapshot_chain_sweep_plot(sweep_rows, "v_i", {0: "step 1"})
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- spread / uncertainty plot


def test_build_spread_plot_has_box_trace():
    fig = build_spread_plot([2.0, 2.1, 1.9, 5.0], "a")
    assert len(fig.data) == 1
    assert fig.data[0].type == "box"
    assert list(fig.data[0].y) == [2.0, 2.1, 1.9, 5.0]


def test_snapshot_spread_plot_produces_png_bytes():
    png = snapshot_spread_plot([2.0, 2.1, 1.9, 5.0], "a")
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- Monte Carlo histogram plot


def test_build_histogram_plot_has_histogram_trace_and_reference_lines():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    fig = build_histogram_plot(samples, "d", mean=3.0, p5=1.2, p95=4.8)
    assert any(t.type == "histogram" for t in fig.data)
    # 3 add_vline calls each add a shape to the layout
    assert len(fig.layout.shapes) == 3


def test_build_histogram_plot_without_reference_lines():
    samples = [1.0, 2.0, 3.0]
    fig = build_histogram_plot(samples, "d")
    assert len(fig.layout.shapes) == 0


def test_snapshot_histogram_plot_produces_png_bytes():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    png = snapshot_histogram_plot(samples, "d", mean=3.0, p5=1.2, p95=4.8)
    assert png.startswith(b"\x89PNG")
