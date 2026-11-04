"""One lean end-to-end check of the Posterior page: run a demo, read the verdict; refuse a bad model with a reason."""
from tests.test_ui_smoke import _app

MODE = "🎲 Posterior sampling (MCMC)"


def _open():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    return at


def test_a_demo_runs_and_reports_convergence_and_the_parameters():
    at = _open()
    next(b for b in at.button if b.key == "post_run").click()
    at.run(timeout=180)
    assert not at.exception
    assert any("Converged" in m.value for m in at.markdown)
    assert at.dataframe and not [e for e in at.error if "LM Studio" not in e.value]        # (the sidebar's own LM Studio notice is not ours)


def test_a_model_with_an_unknown_symbol_is_explained():
    at = _open()
    next(r for r in at.radio if r.key == "post_source").set_value("Paste or upload data")
    at.run(timeout=120)
    next(t for t in at.text_area if t.key == "post_paste").set_value("x,y\n0,1\n1,2\n2,3\n3,4\n4,5\n5,6")
    at.run(timeout=120)                                                    # the model fields appear once the data parses
    next(t for t in at.text_input if t.key == "post_expr").set_value("a*x + z")
    next(t for t in at.text_input if t.key == "post_names").set_value("a")
    at.run(timeout=120)
    next(b for b in at.button if b.key == "post_run").click()
    at.run(timeout=120)
    assert not at.exception and any("not listed" in e.value for e in at.error)
