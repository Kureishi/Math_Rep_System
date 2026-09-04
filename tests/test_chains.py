import pytest

import modules.chains as chains_module
from modules.chains import InputBinding
from modules.equation_engine import build_model


def _kinematics_model(v_i_known=8.0):
    """a = (v_f - v_i) / t -- solves for acceleration."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity",
             "known_value": str(v_i_known) if v_i_known is not None else None, "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _downstream_model():
    """d = a * t2 -- a downstream problem whose 'a' will be wired to the
    upstream kinematics problem's solved acceleration."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
            {"symbol": "t2", "meaning": "second time interval", "known_value": "10", "unit": "s"},
            {"symbol": "d", "meaning": "extra distance", "known_value": None, "unit": "m"},
        ],
        "equations": [
            {"name": "dist", "kind": "equation", "expression": "Eq(d, a * t2)", "derivation": ""},
        ],
        "solve_for": ["d"], "assumptions": [],
    })


@pytest.fixture(autouse=True)
def _redirect_db(tmp_path, monkeypatch):
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "test_chains.db")


# ---------------------------------------------------------------- chain CRUD

def test_create_and_list_chains():
    cid = chains_module.create_chain("My physics homework")
    chains = chains_module.list_chains()
    assert len(chains) == 1
    assert chains[0]["id"] == cid
    assert chains[0]["name"] == "My physics homework"


def test_rename_chain():
    cid = chains_module.create_chain("old name")
    chains_module.rename_chain(cid, "new name")
    chains = chains_module.list_chains()
    assert chains[0]["name"] == "new name"


def test_delete_chain_removes_steps_too():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "problem 1", _kinematics_model(), "a")
    chains_module.delete_chain(cid)
    assert chains_module.list_chains() == []
    assert chains_module.load_chain(cid) is None


def test_load_nonexistent_chain_returns_none():
    assert chains_module.load_chain(99999) is None


# ---------------------------------------------------------------- add_step / resolving

def test_single_step_resolves_to_correct_value():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "car problem", _kinematics_model(), "a")
    chain = chains_module.load_chain(cid)
    assert len(chain.steps) == 1
    step = chain.steps[0]
    assert step.status == "ok"
    assert step.output_value == pytest.approx(2.0)  # (20-8)/6


def test_add_step_rejects_non_algebraic_or_unrequested_target():
    cid = chains_module.create_chain("chain")
    model = _kinematics_model()
    with pytest.raises(ValueError):
        chains_module.add_step(cid, "car problem", model, "v_f")  # not in solve_for


def test_two_step_chain_cascades_upstream_output_downstream():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")
    chains_module.add_step(
        cid, "downstream", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )
    chain = chains_module.load_chain(cid)
    assert chain.steps[0].output_value == pytest.approx(2.0)
    assert chain.steps[1].status == "ok"
    assert chain.steps[1].output_value == pytest.approx(2.0 * 10)  # a * t2


def test_editing_upstream_literal_binding_cascades_downstream():
    """The core spreadsheet-like behavior: changing an upstream input
    automatically re-solves everything downstream of it."""
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")
    chains_module.add_step(
        cid, "downstream", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )

    # override the upstream problem's own v_i input (literal binding on
    # step 0 itself) -- this should ripple through both steps
    chains_module.set_step_bindings(
        cid, 0, [InputBinding(symbol="v_i", source="literal", literal_value=0.0)],
    )
    chain = chains_module.load_chain(cid)
    new_a = (20 - 0) / 6
    assert chain.steps[0].output_value == pytest.approx(new_a)
    assert chain.steps[1].output_value == pytest.approx(new_a * 10)


def test_broken_upstream_binding_reports_error_not_crash():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")
    # references position 5, which doesn't exist
    chains_module.add_step(
        cid, "downstream", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=5, upstream_symbol="a")],
    )
    chain = chains_module.load_chain(cid)
    assert chain.steps[1].status == "error"
    assert chain.steps[1].error_detail is not None
    assert chain.steps[1].output_value is None


def test_binding_referencing_a_later_or_equal_position_is_rejected():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "self-referencing", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )
    chain = chains_module.load_chain(cid)
    assert chain.steps[0].status == "error"


def test_remove_step_shifts_positions_down():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "s0", _kinematics_model(), "a")
    chains_module.add_step(cid, "s1", _kinematics_model(v_i_known=5.0), "a")
    chains_module.add_step(cid, "s2", _kinematics_model(v_i_known=2.0), "a")

    chains_module.remove_step(cid, 0)
    chain = chains_module.load_chain(cid)
    assert len(chain.steps) == 2
    assert [s.position for s in chain.steps] == [0, 1]
    assert chain.steps[0].problem_text == "s1"
    assert chain.steps[1].problem_text == "s2"


def test_remove_step_leaves_broken_binding_reported_as_error_not_silently_fixed():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "s0", _kinematics_model(), "a")
    chains_module.add_step(
        cid, "s1", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )
    chains_module.remove_step(cid, 0)  # s1's binding now points at a step that no longer exists
    chain = chains_module.load_chain(cid)
    assert len(chain.steps) == 1
    assert chain.steps[0].status == "error"


def test_max_steps_per_chain_enforced(monkeypatch):
    monkeypatch.setattr(chains_module, "MAX_STEPS_PER_CHAIN", 2)
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "s0", _kinematics_model(), "a")
    chains_module.add_step(cid, "s1", _kinematics_model(), "a")
    with pytest.raises(ValueError):
        chains_module.add_step(cid, "s2", _kinematics_model(), "a")


# ---------------------------------------------------------------- suggest_bindings

def test_suggest_bindings_matches_by_symbol_name():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")
    chain = chains_module.load_chain(cid)

    downstream_model = _downstream_model()
    suggestions = chains_module.suggest_bindings(chain, downstream_model)
    assert len(suggestions) == 1
    assert suggestions[0].symbol == "a"
    assert suggestions[0].source == "upstream"
    assert suggestions[0].upstream_position == 0


def test_suggest_bindings_skips_already_known_variables():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")
    chain = chains_module.load_chain(cid)

    # a downstream model where 'a' is ALREADY known -- shouldn't get an
    # auto-suggested binding even though the name matches
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "acceleration", "known_value": "9.8", "unit": "m/s^2"},
            {"symbol": "t2", "meaning": "time", "known_value": None, "unit": "s"},
        ],
        "equations": [{"name": "x", "kind": "equation", "expression": "Eq(t2, a)", "derivation": ""}],
        "solve_for": ["t2"], "assumptions": [],
    })
    assert chains_module.suggest_bindings(chain, model) == []


# ---------------------------------------------------------------- sweep_step_binding

def test_sweep_step_binding_returns_one_row_per_value():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "upstream", _kinematics_model(),
        "a", bindings=[InputBinding(symbol="v_i", source="literal", literal_value=8.0)],
    )
    rows = chains_module.sweep_step_binding(cid, 0, "v_i", [0.0, 5.0, 10.0])
    assert len(rows) == 3
    assert [r["value"] for r in rows] == [0.0, 5.0, 10.0]
    # a = (20 - v_i) / 6
    assert rows[0]["outputs"][0] == pytest.approx((20 - 0.0) / 6)
    assert rows[1]["outputs"][0] == pytest.approx((20 - 5.0) / 6)
    assert rows[2]["outputs"][0] == pytest.approx((20 - 10.0) / 6)


def test_sweep_step_binding_cascades_to_downstream_steps():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "upstream", _kinematics_model(),
        "a", bindings=[InputBinding(symbol="v_i", source="literal", literal_value=8.0)],
    )
    chains_module.add_step(
        cid, "downstream", _downstream_model(), "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )
    rows = chains_module.sweep_step_binding(cid, 0, "v_i", [0.0, 6.0])
    a0 = (20 - 0.0) / 6
    a1 = (20 - 6.0) / 6
    assert rows[0]["outputs"][1] == pytest.approx(a0 * 10)
    assert rows[1]["outputs"][1] == pytest.approx(a1 * 10)


def test_sweep_step_binding_restores_original_bindings_after_sweep():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "upstream", _kinematics_model(),
        "a", bindings=[InputBinding(symbol="v_i", source="literal", literal_value=8.0)],
    )
    chains_module.sweep_step_binding(cid, 0, "v_i", [0.0, 100.0])
    chain = chains_module.load_chain(cid)
    literal = next(b for b in chain.steps[0].bindings if b.symbol == "v_i")
    assert literal.literal_value == 8.0
    # and the chain's actual stored/resolved state reflects the ORIGINAL
    # binding, not the last swept value
    assert chain.steps[0].output_value == pytest.approx((20 - 8.0) / 6)


def test_sweep_step_binding_rejects_missing_literal_binding():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(cid, "upstream", _kinematics_model(), "a")  # no bindings at all
    with pytest.raises(ValueError):
        chains_module.sweep_step_binding(cid, 0, "v_i", [0.0, 5.0])


def test_sweep_step_binding_rejects_empty_values():
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "upstream", _kinematics_model(),
        "a", bindings=[InputBinding(symbol="v_i", source="literal", literal_value=8.0)],
    )
    with pytest.raises(ValueError):
        chains_module.sweep_step_binding(cid, 0, "v_i", [])


def test_sweep_step_binding_enforces_max_points(monkeypatch):
    monkeypatch.setattr(chains_module, "MAX_SWEEP_POINTS", 3)
    cid = chains_module.create_chain("chain")
    chains_module.add_step(
        cid, "upstream", _kinematics_model(),
        "a", bindings=[InputBinding(symbol="v_i", source="literal", literal_value=8.0)],
    )
    with pytest.raises(ValueError):
        chains_module.sweep_step_binding(cid, 0, "v_i", [0.0, 1.0, 2.0, 3.0, 4.0])


def test_sweep_step_binding_unknown_chain_raises():
    with pytest.raises(ValueError):
        chains_module.sweep_step_binding(99999, 0, "v_i", [0.0, 1.0])
