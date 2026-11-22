"""Find chart text drawn on top of other chart text, in a real browser.

Unit tests can check a figure's layout NUMBERS (panel gap, margins) but not whether the browser's rendered tick labels, axis titles and subplot titles actually collide: that
depends on font metrics and on how many tick labels there are. This measures the rendered boxes: every SVG text in every chart on the page, in both colour themes, and reports
pairs that overlap by more than 1 px in each direction (a button's own label does not count as overlapping the button).

    streamlit run scripts/dark_mode_gallery.py --server.port 8599 &
    python scripts/chart_text_overlap_scan.py --port 8599

Exits 1 if anything overlaps. Add a figure to the gallery page to cover it. Needs `pip install playwright && playwright install chromium` (a dev tool).
"""
import argparse
import sys

from playwright.sync_api import sync_playwright

JS = r"""
() => {
  const result = [];
  document.querySelectorAll('.js-plotly-plot').forEach((chart, index) => {
    const items = [];
    for (const t of chart.querySelectorAll('svg text')) {
      const r = t.getBoundingClientRect();
      if (r.width < 2 || r.height < 2 || t.closest('.updatemenu-button')) continue;
      const cs = getComputedStyle(t);
      if (cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) continue;
      items.push({label: t.textContent.slice(0, 40), x: r.x, y: r.y, w: r.width, h: r.height});
    }
    for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j];
      const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x), oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (ox > 1 && oy > 1) result.push({chart: index + 1, pair: `${a.label}  <->  ${b.label}`, overlap: `${Math.round(ox)} x ${Math.round(oy)} px`});
    }
  });
  return result;
}
"""

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8599)
args = ap.parse_args()
found = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for mode in ("light", "dark"):                    # the browser's colour scheme: Streamlit's "Use system setting" theme follows it
        page = browser.new_page(viewport={"width": 1500, "height": 1400}, color_scheme=mode)
        page.goto(f"http://localhost:{args.port}/")
        page.wait_for_selector(".js-plotly-plot", timeout=60000)
        page.wait_for_timeout(3500)
        for hit in page.evaluate(JS):
            found.append((mode, hit))
    browser.close()
print(f"{len(found)} overlapping pair(s) of chart text")
for mode, hit in found:
    print(f"  [{mode}] chart {hit['chart']}: {hit['pair']}  ({hit['overlap']})")
sys.exit(1 if found else 0)
