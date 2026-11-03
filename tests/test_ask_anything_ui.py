"""One lean end-to-end check that the sidebar's "Ask anything" box routes, prefills and explains."""
from tests.test_ui_smoke import _app


def _ask(at, text):
    at.sidebar.text_area(key="ask_query").set_value(text)
    at.run(timeout=120)


def test_a_clear_request_jumps_to_the_mode_and_prefills_it():
    at = _app()
    at.run(timeout=120)
    _ask(at, "is sin(x)**2 + cos(x)**2 equivalent to 1?")
    assert not at.exception
    assert at.session_state["app_mode"] == "🔁 Check equivalence"
    assert at.session_state["equiv_expr1"] == "sin(x)**2 + cos(x)**2" and at.session_state["equiv_expr2"] == "1"
    assert any("compares two expressions" in c.value for c in at.sidebar.caption)


def test_pasted_data_lands_in_discover_ready_to_run():
    at = _app()
    at.run(timeout=120)
    _ask(at, "find the differential equation behind this\nt,x\n0,1\n0.1,0.9\n0.2,0.82\n0.3,0.74\n0.4,0.67")
    assert not at.exception and at.session_state["app_mode"] == "🧭 Discover equations"
    assert at.session_state["discover_paste"].startswith("t,x")


def test_an_unsure_request_offers_choices_instead_of_guessing():
    at = _app()
    at.run(timeout=120)
    before = at.session_state["app_mode"]
    _ask(at, "curvature")
    assert not at.exception and at.session_state["app_mode"] == before
    assert any(b.key and b.key.startswith("ask_pick_") for b in at.sidebar.button)
    next(b for b in at.sidebar.button if b.key == "ask_pick_🧮 Tensor calculus").click()
    at.run(timeout=120)
    assert at.session_state["app_mode"] == "🧮 Tensor calculus"
