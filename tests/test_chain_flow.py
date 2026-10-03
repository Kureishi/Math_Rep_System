"""modules/chain_flow.py: layout of a chain as a cascade, and the broken-link
conditions. Built from hand-made Chain objects (no database), plus one
round trip through the real resolve_chain."""
import pytest

import modules.chains as chains_module
from modules.chain_flow import ChainFlow, build_chain_flow, step_caption
from modules.chains import Chain, ChainStep, InputBinding
from tests.test_chains import _downstream_model, _kinematics_model


def _step(position, symbol="a", value=None, status="ok", error=None, bindings=()):
    return ChainStep(position=position, problem_text=f"step {position}", raw_json={}, output_symbol=symbol,
                     bindings=list(bindings), output_value=value, status=status, error_detail=error)


def _up(symbol, position):
    return InputBinding(symbol=symbol, source="upstream", upstream_position=position, upstream_symbol=symbol)


def _lit(symbol, value):
    return InputBinding(symbol=symbol, source="literal", literal_value=value)


def _chain(*steps):
    return Chain(id=1, name="demo", steps=list(steps))


# ------------------------------------------------------------------ layout

def test_steps_cascade_down_and_to_the_right():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0), _step(1, "d", 20.0), _step(2, "e", 5.0)))
    assert flow.name == "demo"
    assert [(s.x, s.y) for s in flow.steps] == [(0.0, 0.0), (2.0, -1.0), (4.0, -2.0)]
    assert [(s.position, s.symbol, s.value, s.status) for s in flow.steps][1] == (1, "d", 20.0, "ok")


def test_typed_in_inputs_stack_above_their_step():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0, bindings=[_lit("t", 6.0), _lit("m", 1.5)])))
    t, m = flow.literals
    assert (t.symbol, t.value, t.step) == ("t", 6.0, 0) and (m.symbol, m.value) == ("m", 1.5)
    assert t.x == m.x == 0.0 and t.y < m.y                                  # the second sits above the first
    assert flow.edges == []


def test_a_literal_binding_without_a_value_is_left_out():
    flow = build_chain_flow(_chain(_step(0, bindings=[InputBinding("t", "literal", literal_value=None)])))
    assert flow.literals == []


def test_an_upstream_binding_carries_the_upstream_steps_output():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0), _step(1, "d", 20.0, bindings=[_up("a", 0)])))
    (edge,) = flow.edges
    assert (edge.source, edge.target, edge.symbol, edge.carried, edge.broken) == (0, 1, "a", 2.0, False)


def test_an_input_symbol_can_differ_from_the_upstream_output_symbol():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0), _step(1, "d", 20.0,
                                                              bindings=[InputBinding("accel", "upstream", None, 0, "a")])))
    assert flow.edges[0].symbol == "accel" and flow.edges[0].carried == 2.0


@pytest.mark.parametrize("binding_position, steps, reason", [
    (5, [_step(0, "a", 2.0), _step(1, "d", 1.0)], "doesn't exist"),
    (None, [_step(0, "a", 2.0), _step(1, "d", 1.0)], "doesn't exist"),
    (1, [_step(0, "a", 2.0), _step(1, "d", 1.0)], "itself or a later step"),
    (2, [_step(0, "a", 2.0), _step(1, "d", 1.0), _step(2, "e", 1.0)], "itself or a later step"),
    (0, [_step(0, "a", None, status="error", error="boom"), _step(1, "d", None)], "step 1 produced no value"),
])
def test_unusable_upstream_bindings_are_flagged_broken_with_a_reason(binding_position, steps, reason):
    steps[1].bindings = [InputBinding("a", "upstream", None, binding_position, "a")]
    flow = build_chain_flow(_chain(*steps))
    (edge,) = [e for e in flow.edges if e.target == 1]
    assert edge.broken and reason in edge.reason and edge.carried is None


# ------------------------------------------------------------------ captions

def test_caption_for_a_step_that_received_values():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0),
                                    _step(1, "d", 20.0, bindings=[_up("a", 0), _lit("t2", 10.0)])))
    assert step_caption(flow, 1) == "Step 2 receives a = 2 from step 1; t2 = 10 typed in; result: d = 20."


def test_caption_for_a_first_step_with_nothing_wired_in():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0)))
    assert step_caption(flow, 0) == "Step 1 receives no overridden inputs; result: a = 2."


def test_caption_for_a_failed_step_and_a_broken_link():
    flow = build_chain_flow(_chain(_step(0, "a", 2.0), _step(1, "d", None, status="error", error="Solve failed: x",
                                                              bindings=[InputBinding("a", "upstream", None, 7, "a")])))
    caption = step_caption(flow, 1)
    assert "a could not be carried (points at a step that doesn't exist)" in caption
    assert caption.endswith("result: failed: Solve failed: x.")


def test_caption_for_an_unsolved_step():
    assert step_caption(build_chain_flow(_chain(_step(0, "a", None, status="stale"))), 0).endswith("not solved yet.")


# ------------------------------------------------------------------ against the real chain machinery

@pytest.fixture
def real_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains.db")
    cid = chains_module.create_chain("kinematics")
    chains_module.add_step(cid, "car accelerates", _kinematics_model(), "a")
    chains_module.add_step(cid, "then cruises", _downstream_model(), "d", bindings=[_up("a", 0)])
    return chains_module.resolve_chain(cid)


def test_flow_of_a_resolved_chain_carries_the_computed_numbers(real_chain):
    flow = build_chain_flow(real_chain)
    assert [s.value for s in flow.steps] == [pytest.approx(2.0), pytest.approx(20.0)]      # a = (20-8)/6, d = a*10
    (edge,) = flow.edges
    assert edge.carried == pytest.approx(2.0) and not edge.broken
    assert step_caption(flow, 1) == "Step 2 receives a = 2 from step 1; result: d = 20."


def test_flow_of_an_empty_chain_has_nothing_in_it():
    assert build_chain_flow(_chain()) == ChainFlow(name="demo")
