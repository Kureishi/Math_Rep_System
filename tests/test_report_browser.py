"""The interactive HTML report in a real browser (skipped where Playwright and Chromium aren't installed)."""
import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from modules.report_export import render_export  # noqa: E402
from tests.report_helpers import full_extras, kinematics_ctx  # noqa: E402


@pytest.fixture(scope="module")
def page_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("report") / "report.html"
    path.write_bytes(render_export("html", kinematics_ctx(extras=full_extras())).data)
    return path


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001 -- browsers not downloaded in this environment
            pytest.skip(f"Chromium unavailable: {e}")
        yield b
        b.close()


def test_the_page_loads_with_no_errors_and_draws_every_live_plot(browser, page_path):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    requests = []
    page.on("request", lambda r: requests.append(r.url))
    page.goto(page_path.as_uri())
    page.wait_for_selector(".plot .main-svg")
    assert errors == []
    assert page.evaluate("document.querySelectorAll('.plot').length") == 3
    assert page.evaluate("document.querySelectorAll('.plot').length === document.querySelectorAll('.plot .main-svg').length / 3")
    assert [u for u in requests if not u.startswith(("file:", "data:"))] == [], "no network requests"


def test_a_plot_is_interactive_not_a_picture(browser, page_path):
    page = browser.new_page()
    page.goto(page_path.as_uri())
    page.wait_for_selector(".plot .modebar")
    assert page.evaluate("document.querySelectorAll('.plot .modebar-btn').length") > 3        # zoom, pan, reset ...


def test_clicking_an_equation_gives_copy_feedback(browser, page_path):
    page = browser.new_page()
    page.goto(page_path.as_uri())
    page.click(".eq >> nth=0")
    page.wait_for_function("document.querySelector('.eq').classList.contains('copied')")


def test_equations_are_visible_in_dark_mode(browser, page_path):
    page = browser.new_page(color_scheme="dark")
    page.goto(page_path.as_uri())
    fill = page.evaluate("getComputedStyle(document.querySelector('.eq svg path, .eq svg use')).fill")
    body = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert fill != body and fill not in ("rgb(0, 0, 0)",)


def test_the_navigation_jumps_to_a_section(browser, page_path):
    page = browser.new_page()
    page.goto(page_path.as_uri())
    page.click('nav a[href="#sec-steps"]')
    assert page.evaluate("location.hash") == "#sec-steps"
