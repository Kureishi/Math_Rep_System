"""Reading a click on a plot (modules/plot_selection.py)."""
from dataclasses import dataclass

import pytest

from modules.plot_selection import (
    clicked_grid_point, clicked_labels, clicked_node_ids, clicked_symbol, grid_value, selection_points,
    targets_for_node,
)


def state(*points):
    return {"selection": {"points": list(points), "point_indices": [], "box": [], "lasso": []}}


class AttrDict(dict):
    """Streamlit's selection state supports both item and attribute access."""
    __getattr__ = dict.get


# ------------------------------------------------------------------ the raw points

@pytest.mark.parametrize("empty", [None, {}, {"selection": None}, {"selection": {}}, {"selection": {"points": []}}])
def test_no_selection_means_no_points(empty):
    assert selection_points(empty) == []
    assert clicked_labels(empty) == []
    assert clicked_symbol(empty, ["a"]) is None


def test_attribute_style_state_is_read_like_a_plain_dict():
    s = AttrDict(selection=AttrDict(points=[AttrDict(x=1.0, y="v0")]))
    assert clicked_labels(s) == ["v0"]


def test_labels_are_distinct_and_keep_selection_order():
    s = state({"x": 1, "y": "b"}, {"x": 2, "y": "a"}, {"x": 3, "y": "b"})
    assert clicked_labels(s) == ["b", "a"]


def test_a_point_without_the_axis_is_skipped_rather_than_crashing():
    assert clicked_labels(state({"x": 1}, {"x": 2, "y": "k"})) == ["k"]


# ------------------------------------------------------------------ tornado bars

def test_clicked_symbol_returns_the_first_valid_label():
    s = state({"x": 1, "y": "zzz"}, {"x": 2, "y": "u"}, {"x": 3, "y": "t"})
    assert clicked_symbol(s, ["t", "u"]) == "u"


def test_a_stale_selection_for_a_symbol_not_on_this_chart_is_ignored():
    assert clicked_symbol(state({"x": 1, "y": "gone"}), ["u", "t"]) is None


# ------------------------------------------------------------------ heatmap grid points

XS, YS = [1.0, 2.0, 3.0], [10.0, 20.0]


def test_a_grid_click_is_snapped_to_the_grids_own_values():
    assert clicked_grid_point(state({"x": 2.0000000001, "y": 19.9999999999}), XS, YS) == (2.0, 20.0)


def test_a_point_off_the_grid_is_not_a_grid_click():
    assert clicked_grid_point(state({"x": 2.5, "y": 15.0}), XS, YS) is None


def test_non_numeric_or_non_finite_coordinates_are_rejected():
    assert clicked_grid_point(state({"x": "a", "y": 10}), XS, YS) is None
    assert clicked_grid_point(state({"x": float("nan"), "y": 10}), XS, YS) is None
    assert clicked_grid_point(state({"x": 1, "y": None}), XS, YS) is None


def test_an_empty_grid_has_no_clickable_points():
    assert clicked_grid_point(state({"x": 1, "y": 1}), [], []) is None


def test_the_first_on_grid_point_wins_when_several_are_selected():
    s = state({"x": 9, "y": 9}, {"x": 3.0, "y": 10.0})
    assert clicked_grid_point(s, XS, YS) == (3.0, 10.0)


def test_grid_value_reads_z_with_rows_by_y_and_columns_by_x():
    z = [[11.0, 12.0, 13.0],        # y = 10
         [21.0, 22.0, 23.0]]        # y = 20
    assert grid_value(XS, YS, z, 2.0, 20.0) == 22.0
    assert grid_value(XS, YS, z, 3.0, 10.0) == 13.0


def test_grid_value_is_none_for_a_nan_cell_or_a_mismatched_grid():
    assert grid_value(XS, YS, [[float("nan")] * 3, [1.0] * 3], 1.0, 10.0) is None
    assert grid_value(XS, YS, [[1.0]], 3.0, 20.0) is None


# ------------------------------------------------------------------ graph nodes

@dataclass
class Node:
    id: str
    x: float
    y: float
    kind: str = "known"


@dataclass
class Edge:
    source: str
    target: str


NODES = [Node("var:v", 0.0, 0.0), Node("eq:vel", 1.0, 0.0, "equation"), Node("var:a", 2.0, 0.0, "unknown")]
EDGES = [Edge("var:v", "eq:vel"), Edge("eq:vel", "var:a")]


def test_nodes_are_identified_by_their_coordinates():
    assert clicked_node_ids(state({"x": 1.0, "y": 0.0}), NODES) == ["eq:vel"]
    assert clicked_node_ids(state({"x": 2.0, "y": 0.0}, {"x": 0.0, "y": 0.0}), NODES) == ["var:a", "var:v"]


def test_a_click_on_empty_canvas_selects_no_node():
    assert clicked_node_ids(state({"x": 1.5, "y": 0.7}), NODES) == []


def test_a_variable_node_leads_to_its_own_steps_only_if_it_was_solved():
    assert targets_for_node("var:a", EDGES, ["a", "d"]) == ["a"]
    assert targets_for_node("var:v", EDGES, ["a", "d"]) == []          # a given input has none


def test_an_equation_node_leads_to_the_unknowns_it_feeds():
    edges = EDGES + [Edge("eq:vel", "var:d")]
    assert targets_for_node("eq:vel", edges, ["a"]) == ["a"]
    assert targets_for_node("eq:vel", edges, ["a", "d"]) == ["a", "d"]
    assert targets_for_node("eq:vel", edges, []) == []


def test_an_unknown_node_id_format_leads_nowhere():
    assert targets_for_node("weird", EDGES, ["a"]) == []
