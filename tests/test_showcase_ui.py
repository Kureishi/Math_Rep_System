"""One lean end-to-end check of the showcase page; the physics is tested in test_showcase.py."""
from tests.test_ui_smoke import _app

MODE = "🎬 Visual showcase"


def test_the_page_renders_the_orbit_and_renders_a_black_hole_with_its_check():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    assert any("Check: the conserved quantity" in s.value for s in at.success)           # the orbit tab computes on load
    next(b for b in at.button if b.key == "showcase_lens_go").click()
    at.run(timeout=120)
    assert not at.exception
    assert any("the shadow's radius" in s.value for s in at.success)


def test_an_unstable_orbit_is_explained():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    next(s for s in at.slider if s.key == "showcase_orbit_p").set_value(7.0)
    next(s for s in at.slider if s.key == "showcase_orbit_e").set_value(0.9)
    at.run(timeout=120)
    assert not at.exception and any("plunges" in e.value for e in at.error)
