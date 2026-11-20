"""The word-problem page's mode-suggestion banner: it appears for text that clearly belongs to another mode, never for ordinary word problems, and its button
switches mode with the prefill the Ask-anything box would have used."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import router

APP = str(Path(__file__).parent.parent / "app.py")

# ordinary word problems, from every area the solver handles: none may be offered a different mode
WORD_PROBLEMS = [
    "A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its acceleration, and how far does it travel in that time?",
    "A rectangle has a measured length of 12.4 ± 0.2 m and a measured width of 5.1 ± 0.1 m. What is its area?",
    "A shipping company charges a flat $5 fee plus $0.80 per kilogram. For what package weights is the total cost under $25?",
    "A force of (10, 4) N acts on an object as it moves through a displacement of (5, 3) m. Calculate the work done by the force.",
    "A population of bacteria grows at a rate proportional to its current size, with a growth rate of 0.3 per hour. If the population starts at 1000, what is the population after 5 hours?",
    "A savings account starts with $1000 and earns 5% interest per year, compounded annually, with an additional $500 deposited at the end of each year. Find a formula for the balance after n years.",
    "A cylindrical can must hold a volume of 500 cm³. Find the radius and height that minimize the surface area of material used.",
    "A surveyor measures a triangular plot of land. Two sides are 120 m and 85 m, with an angle of 55 degrees between them. Find the length of the third side and the other two angles.",
    "A ball is launched at 20 m/s at a 30-degree angle. Find its maximum height.",
    "A 2 kg mass hangs from a spring with constant 50 N/m. Find the period of oscillation.",
    "A pendulum of length 1.5 m swings through a small angle. What is its period?",
    "A 10 ohm resistor and a 5 mH inductor are in series with a 12 V source. Find the current after 1 ms.",
    "A tank holds 200 L of water with 5 kg of salt. Brine with 0.1 kg/L flows in at 4 L/min. Find the salt after 30 minutes.",
    "A gas at 300 K and 2 atm is heated at constant volume to 450 K. What is the new pressure?",
    "How many grams of NaCl are needed to make 250 mL of a 0.5 M solution?",
    "A wave on a string has frequency 40 Hz and wavelength 0.5 m. What is its speed?",
    "Light of wavelength 600 nm passes through a slit 0.1 mm wide onto a screen 2 m away. Find the width of the central maximum.",
    "A satellite orbits Earth at 400 km altitude. Find its orbital period.",
    "A particle moves along x(t) = t^3 - 6t^2 + 9t. When is it at rest?",
    "Find the dimensions of a rectangle with perimeter 40 that has maximum area.",
    "A block slides down a frictionless incline of 30 degrees. Find its acceleration.",
    "A capacitor of 10 microfarads discharges through a 1 kilohm resistor. How long until its charge falls to 10 percent?",
    "How much heat is needed to raise 2 kg of water from 20 C to 80 C?",
    "Two resistors of 4 ohm and 6 ohm are connected in parallel. What is the equivalent resistance?",
    "A 0.2 M solution of acetic acid has pKa 4.76. Find its pH.",
]
# requests that clearly belong to another mode
CROSS_MODE = [
    ("Balance Fe + O2 -> Fe2O3", "🧪 Chemistry"),
    ("double slit interference fringe spacing for 550 nm light", "🌊 Waves & optics lab"),
    ("Is sin(x)**2 + cos(x)**2 equivalent to 1?", "🔁 Check equivalence"),
    ("figure-eight three-body orbit with a symplectic integrator", "🌀 Dynamics & chaos lab"),
    ("series rlc circuit damping ratio and resonant frequency", "🔌 Circuits & control lab"),
    ("energy levels of the infinite square well", "⚛️ Quantum mechanics"),
    ("triangle with sides 7, 9 and 5: find the angles", "📐 Geometry"),
]


@pytest.mark.parametrize("text", WORD_PROBLEMS)
def test_an_ordinary_word_problem_is_never_offered_another_mode(text):
    assert router.suggest_mode(text) is None


@pytest.mark.parametrize("text, mode", CROSS_MODE)
def test_a_request_that_belongs_elsewhere_is_offered_its_mode(text, mode):
    found = router.suggest_mode(text)
    assert found is not None and found.mode == mode and found.confidence >= router.BANNER_CONFIDENCE and found.reason


@pytest.mark.parametrize("text", ["", "   ", "pH", "balance", "x"])
def test_text_too_short_to_judge_gets_no_suggestion(text):
    assert router.suggest_mode(text) is None


def test_the_suggestion_threshold_is_bolder_than_the_automatic_jump_but_not_by_much():
    assert 0.4 <= router.BANNER_CONFIDENCE < router.AUTO_GO_CONFIDENCE


# ------------------------------------------------------------------------------------------ on the page

def _page(text=None):
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    if text is not None:
        at.text_area[0].set_value(text)
        at.run()
    return at


def _banner(at):
    return [i for i in at.info if "looks like a request for" in i.value]


def _button(at, key):
    return next((b for b in at.button if b.key == key), None)


def test_the_banner_appears_for_a_reaction_and_says_where_and_why():
    at = _page("Balance Fe + O2 -> Fe2O3")
    assert not at.exception
    banner = _banner(at)
    assert banner and "🧪 Chemistry" in banner[0].value and "sure" in banner[0].value and "without LM Studio" in banner[0].value
    assert _button(at, "route_banner_go") is not None and _button(at, "route_banner_stay") is not None


def test_there_is_no_banner_for_an_ordinary_word_problem():
    at = _page(WORD_PROBLEMS[0])
    assert not at.exception and not _banner(at) and _button(at, "route_banner_go") is None


def test_the_switch_button_changes_mode_and_prefills_what_the_ask_box_would():
    at = _page("Balance Fe + O2 -> Fe2O3")
    expected = router.route("Balance Fe + O2 -> Fe2O3")
    _button(at, "route_banner_go").click()
    at.run()
    assert not at.exception
    assert at.session_state["app_mode"] == "🧪 Chemistry"
    for key, value in expected.prefill.items():
        assert at.session_state[key] == value
    assert at.subheader and any("hemistry" in h.value for h in at.subheader)             # the chemistry page is what is on screen now


def test_keep_it_here_hides_the_banner_for_that_text_only():
    at = _page("Balance Fe + O2 -> Fe2O3")
    _button(at, "route_banner_stay").click()
    at.run()
    assert not at.exception and not _banner(at) and at.session_state["app_mode"] == "📝 Word problem solver"
    at.text_area[0].set_value("double slit interference fringe spacing for 550 nm light")
    at.run()
    assert _banner(at) and "Waves" in _banner(at)[0].value                  # a different request is a new question
    at.text_area[0].set_value("Balance Fe + O2 -> Fe2O3")
    at.run()
    assert not _banner(at)                                                  # ...and the dismissed one stays dismissed, even re-typed with different spacing/case


def test_the_dismissal_ignores_case_and_spacing():
    at = _page("Balance Fe + O2 -> Fe2O3")
    _button(at, "route_banner_stay").click()
    at.run()
    at.text_area[0].set_value("balance   FE + o2 ->   Fe2O3 ")
    at.run()
    assert not _banner(at)


def test_a_solved_problem_on_screen_suppresses_the_banner():
    from tests.report_helpers import kinematics_ctx
    ctx = kinematics_ctx()
    at = _page()
    at.session_state["model"], at.session_state["report"] = ctx.model, ctx.report
    at.session_state["steps"], at.session_state["problem_text"] = ctx.steps_by_target, ctx.problem_text
    at.run()
    at.text_area[0].set_value("Balance Fe + O2 -> Fe2O3")
    at.run()
    assert not at.exception and not _banner(at)


def test_the_banner_does_not_stop_represent_and_solve_from_being_available():
    at = _page("Balance Fe + O2 -> Fe2O3")
    assert any("Represent" in b.label for b in at.button)
