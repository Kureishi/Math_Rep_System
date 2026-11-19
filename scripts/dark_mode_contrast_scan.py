"""Measure the text contrast of every visible element of the gallery page, in several UI states, and report anything below WCAG AA (4.5 : 1).

Why a browser: the dark theme is a stylesheet laid over Streamlit's light one, and what a rule does depends on the state of the element it lands on (a popover's
trigger is white at rest but dark-tinted when hovered, focused or open). Reading the CSS cannot tell you that; computing each element's colours in a real
browser can. States covered: rest, each tab, an open selectbox / multiselect list, each popover open and after it closes (the trigger keeps focus), hovered
buttons / segments / expander headers, and keyboard focus on an open expander header. CSS `filter: brightness()` on an element or its ancestors is applied to the
text colour, as the browser paints it.

    streamlit run scripts/dark_mode_gallery.py --server.port 8599 &
    python scripts/dark_mode_contrast_scan.py --port 8599            # dark mode (default); --light for light mode
    python scripts/dark_mode_contrast_scan.py --port 8599 --dialog   # the Share-style modal, which is its own surface (add --light for light mode)

Exits 1 if anything is below 4.5 : 1. Inactive (disabled) controls are exempt, as WCAG allows, and a slider's value label is skipped (see the comment in the script). Needs `pip install playwright` and `playwright install chromium` (a dev tool: not needed to run the app).
"""
import argparse
import json
import re
import sys

from playwright.sync_api import sync_playwright
JS = r"""
(page) => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if(!m) return null; const p = m[1].split(',').map(x=>parseFloat(x)); return {r:p[0],g:p[1],b:p[2],a:p.length>3?p[3]:1}; };
  const lum = c => { const f = v => { v/=255; return v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055,2.4)}; return 0.2126*f(c.r)+0.7152*f(c.g)+0.0722*f(c.b); };
  const over = (top, bot) => ({r: top.r*top.a+bot.r*(1-top.a), g: top.g*top.a+bot.g*(1-top.a), b: top.b*top.a+bot.b*(1-top.a), a:1});
  const bgOf = el => { let layers=[]; for(let n=el; n; n=n.parentElement){ const c=parse(getComputedStyle(n).backgroundColor); if(c && c.a>0){ layers.push(c); if(c.a>=1) break; } } let base=page; /* the page colour: what shows through a transparent stack */ for(const l of layers.reverse()) base=over(l,base); return base; };
  const out=[]; const seen=new Set();
  const walker=document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while(walker.nextNode()){
    const t=walker.currentNode; const txt=t.textContent.trim(); if(!txt) continue;
    const el=t.parentElement; if(!el) continue;
    const st=getComputedStyle(el); const r=el.getBoundingClientRect();
    if(st.visibility==='hidden'||st.display==='none'||r.width<2||r.height<2||parseFloat(st.opacity)===0) continue;
    if(['SCRIPT','STYLE','NOSCRIPT'].includes(el.tagName)) continue;
    // WCAG exempts inactive controls; and a slider's value label sits INSIDE the thumb in the DOM but is drawn above it, so the background walk would wrongly use the thumb's fill
    if(el.closest('[disabled],[aria-disabled="true"],:disabled,[data-testid="stSliderThumbValue"]')) continue;
    const fg=parse(el instanceof SVGElement ? st.fill : st.color); if(!fg) continue; const bg=bgOf(el);       // chart text is SVG: coloured by fill, not color
    let k=1; for(let n=el;n;n=n.parentElement){ const f=getComputedStyle(n).filter; const m=f&&f.match(/brightness\(([\d.]+)\)/); if(m) k*=parseFloat(m[1]); }
    fg.r*=k; fg.g*=k; fg.b*=k; const fgc=over(fg,bg);
    const L1=lum(fgc), L2=lum(bg); const ratio=(Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05);
    if(ratio<4.5){ let n=el, path=[]; for(let i=0;i<4&&n;i++,n=n.parentElement){ const id=n.getAttribute&&n.getAttribute('data-testid'); path.push((id?('['+id+']'):n.tagName.toLowerCase())); }
      const key=txt.slice(0,30)+path.join('<'); if(seen.has(key)) continue; seen.add(key);
      out.push({text:txt.slice(0,40), ratio:Math.round(ratio*100)/100, fg:(el instanceof SVGElement ? st.fill : st.color), bg:`rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)})`, path:path.join(' < ')}); }
  }
  return out;
}
"""
def scan(pg, label, results):
    for r in pg.evaluate(JS, PAGE):
        r['state']=label; results.append(r)
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8599)
ap.add_argument("--light", action="store_true", help="scan light mode instead of dark")
ap.add_argument("--json", help="write every measurement to this file")
ap.add_argument("--dialog", action="store_true", help="scan the Share-style dialog instead of the page: it is a modal, its own surface, so it is measured on its own")
args = ap.parse_args()
dark = "0" if args.light else "1"
BASE = f"http://localhost:{args.port}/"
PAGE = {"r": 255, "g": 255, "b": 255, "a": 1} if args.light else {"r": 14, "g": 17, "b": 23, "a": 1}      # the page colour behind everything
results=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000})
    if args.dialog:
        pg.goto(f"{BASE}?dark={dark}&dialog=1"); pg.wait_for_selector('[data-testid="stDialog"]',timeout=40000); pg.wait_for_timeout(2500)
        scan(pg,"dialog open",results)
        dlg = pg.locator('[data-testid="stDialog"]')
        dlg.locator('[data-testid="stExpander"] details > summary').first.hover(); pg.wait_for_timeout(300); scan(pg,"dialog: hover closed expander header",results)
        dlg.locator('[data-testid="stExpander"] details > summary').first.click(); pg.wait_for_timeout(300); scan(pg,"dialog: open expander header",results)
        dlg.locator('[data-testid="stExpander"] details > summary').first.hover(); pg.wait_for_timeout(300); scan(pg,"dialog: hover open expander header",results)
        pg.mouse.move(5,5)
        dlg.get_by_role("button",name="Options").click(); pg.wait_for_timeout(500); scan(pg,"dialog: popover open",results); pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
        scan(pg,"dialog: popover closed, trigger focused",results)
        dlg.get_by_role("button",name="Options").hover(); pg.wait_for_timeout(300); scan(pg,"dialog: hover popover trigger",results)
        dlg.get_by_role("button",name="PDF").hover(); pg.wait_for_timeout(300); scan(pg,"dialog: hover download button",results)
        for tab in ("Answer card","Copy","Send to a chain","Export"):
            pg.locator('[data-testid="stDialog"]').get_by_role("tab",name=re.compile(tab)).click(); pg.wait_for_timeout(600); scan(pg,"dialog tab "+tab,results)
        b.close()
        bad={}
        for r in results: bad.setdefault((r['text'],r['path']),[]).append(r)
        print(f"{len(bad)} distinct low-contrast text spots in the dialog (ratio < 4.5)")
        for (t,pth),rs in sorted(bad.items(), key=lambda kv: kv[1][0]['ratio']):
            r=rs[0]; print(f"{r['ratio']:>5}  fg {r['fg']:<22} on {r['bg']:<18} '{t}'  [{', '.join(sorted({x['state'] for x in rs}))}]  {pth}")
        sys.exit(1 if bad else 0)
    pg.goto(f"{BASE}?dark={dark}"); pg.wait_for_selector('[data-testid="stExpander"]',timeout=40000); pg.wait_for_timeout(2500)
    scan(pg,"initial",results)
    for tab in ("Explore","Practice","Verify"):
        pg.get_by_role("tab",name=tab).click(); pg.wait_for_timeout(600); scan(pg,"tab "+tab,results)
    pg.get_by_role("tab",name="Practice").click(); pg.wait_for_timeout(400)
    # open a select dropdown menu
    sel=pg.locator('[data-testid="stSelectbox"] input[role="combobox"]:visible').first; sel.click(); pg.wait_for_timeout(500); scan(pg,"selectbox open",results); pg.keyboard.press("Escape")
    ms=pg.locator('[data-testid="stMultiSelect"] input[role="combobox"]:visible').first; ms.click(); pg.wait_for_timeout(500); scan(pg,"multiselect open",results); pg.keyboard.press("Escape")
    # popovers
    for name in ("Explain step 7","Copy a"):
        pg.get_by_role("button",name=name).click(); pg.wait_for_timeout(500); scan(pg,"popover "+name,results); pg.keyboard.press("Escape"); pg.wait_for_timeout(300); scan(pg,"popover closed, trigger focused "+name,results)
    # hover states
    for name in ("Secondary button","Explain step 7"):
        pg.get_by_role("button",name=name).hover(); pg.wait_for_timeout(300); scan(pg,"hover "+name,results)
    pg.get_by_role("tab",name="Practice").click(); pg.wait_for_timeout(300)
    pg.locator('[data-testid="stButtonGroup"] button[role="radio"]').nth(1).hover(); pg.wait_for_timeout(300); scan(pg,"hover unselected segment",results)
    pg.mouse.move(5,5)
    pg.locator('[data-testid="stExpander"] details[open] > summary').first.focus(); pg.keyboard.press("Shift+Tab"); pg.keyboard.press("Tab"); pg.wait_for_timeout(300); scan(pg,"keyboard focus on open expander header",results)
    pg.locator('[data-testid="stExpander"] details[open] > summary').first.hover(); pg.wait_for_timeout(300); scan(pg,"hover open expander header",results)
    pg.locator('[data-testid="stExpander"] details:not([open]) > summary').first.hover(); pg.wait_for_timeout(300); scan(pg,"hover closed expander header",results)
    pg.get_by_role("tab",name="Explore").click(); pg.wait_for_timeout(400); scan(pg,"explore again",results)
    b.close()
bad={}
for r in results: bad.setdefault((r['text'],r['path']),[]).append(r)
print(f"{len(bad)} distinct low-contrast text spots (ratio < 4.5)")
for (t,pth),rs in sorted(bad.items(), key=lambda kv: kv[1][0]['ratio']):
    r=rs[0]; print(f"{r['ratio']:>5}  fg {r['fg']:<22} on {r['bg']:<18} '{t}'  [{', '.join(sorted({x['state'] for x in rs}))}]  {pth}")
if args.json:
    json.dump(results, open(args.json, "w", encoding="utf-8"))
sys.exit(1 if bad else 0)
