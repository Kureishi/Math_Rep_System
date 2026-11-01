"""Tests for modules/gr_metrics.py -- presets and the guarded user-metric parser."""
import pytest
import sympy as sp

from modules.gr_metrics import PRESETS, GRInputError, metric_rows, parse_user_metric, preset


def test_every_preset_is_a_symmetric_lorentzian_metric():
    import numpy as np
    from modules.gr_curvature import evaluate, numeric_subs
    for key in PRESETS:
        m = preset(key)
        assert m.g == m.g.T, key
        subs = numeric_subs(m)
        mat = np.array([[evaluate(m.g[i, j], subs) for j in range(m.dim)] for i in range(m.dim)])
        assert (np.linalg.eigvalsh(mat) < 0).sum() == 1, f"{key}: not Lorentzian at its sample point"


def test_unknown_preset_is_a_clear_error():
    with pytest.raises(GRInputError, match="Unknown preset"):
        preset("nope")


def test_user_metric_detects_parameters_and_uses_matching_symbols():
    m = parse_user_metric([["-(1-2*M/r)", "0"], ["0", "1/(1-2*M/r)"]], ["t", "r"])
    assert m.params == {"M": 1.0}
    # the same symbols as the presets use, so expressions from different sources combine
    assert m.g[0, 0].free_symbols == {sp.Symbol("M", positive=True), sp.Symbol("r", positive=True)}


def test_user_metric_round_trips_a_preset():
    k = preset("schwarzschild")
    m = parse_user_metric(metric_rows(k), k.coord_names)
    assert sp.simplify(m.g - k.g) == sp.zeros(4, 4)


@pytest.mark.parametrize("rows,coords,msg", [
    ([["1", "0"], ["0", "1"]], ["x"], "between 2 and"),
    ([["1", "0"], ["0", "1"]], ["x", "x"], "distinct"),
    ([["1", "0"]], ["x", "y"], "grid"),
    ([["1", ""], ["0", "1"]], ["x", "y"], "Every cell"),
    ([["1", "x"], ["0", "1"]], ["x", "y"], "symmetric"),
    ([["__import__('os').system('x')", "0"], ["0", "1"]], ["x", "y"], "not allowed|Unknown|calls"),
    ([["x.real", "0"], ["0", "1"]], ["x", "y"], "not allowed"),
    ([["sin(", "0"], ["0", "1"]], ["x", "y"], "Could not read"),
])
def test_user_metric_rejects_bad_input_with_a_reason(rows, coords, msg):
    with pytest.raises(GRInputError, match=msg):
        parse_user_metric(rows, coords)


def test_coordinate_names_cannot_shadow_functions():
    with pytest.raises(GRInputError, match="cannot be used"):
        parse_user_metric([["1", "0"], ["0", "1"]], ["sin", "y"])


def test_param_defaults_are_applied_and_unknown_ones_ignored():
    m = parse_user_metric([["-a", "0"], ["0", "1"]], ["t", "x"], {"a": 3.0, "zz": 9.0})
    assert m.params == {"a": 3.0}
