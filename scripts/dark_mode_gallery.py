"""A page that shows one of nearly every Streamlit component, under the app's real stylesheet and Streamlit's own theme: light or dark follows the browser's
colour scheme (the scripts below set it with Playwright's color_scheme), because .streamlit/config.toml defines both [theme.light] and [theme.dark].

    streamlit run scripts/dark_mode_gallery.py --server.port 8599
    python scripts/dark_mode_contrast_scan.py --port 8599          # needs: pip install playwright && playwright install chromium

It is the page scripts/dark_mode_contrast_scan.py measures. Add a component here when the app starts using one the gallery lacks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st
from ui.build_stamp import LOADED
from ui.theme import badge_row, inject_base_styles, page_helper_iframe

st.set_page_config(layout="wide")
inject_base_styles()
dark = st.context.theme.type == "dark"           # what ui.a11y.is_dark() reads, so charts follow the page
with st.sidebar:
    st.caption(f"build {LOADED}")                 # the scans compare this with the files on disk and refuse to run against a stale server
    page_helper_iframe()                          # sets data-mrs-theme, which switches the light-only contrast fixes off in dark
    with st.expander("LM Studio", expanded=True):
        st.success("Connected to LM Studio.")
        st.error("Could not reach LM Studio at http://localhost:1234/v1. Open LM Studio, load a model, and click Start Server.")
        st.button("Re-check connection")
        st.selectbox("Reasoning model", ["qwen3", "other"])
    with st.expander("Advanced settings"):
        st.slider("Retries", 0, 5, 2)
st.title("Component gallery")
c = st.columns(3)
with c[0]:
    st.button("Secondary button")
    st.button("Primary button", type="primary")
    st.button("Disabled button", disabled=True)
    st.download_button("Download", data=b"x", file_name="x.txt")
with c[1]:
    with st.popover("Explain step 7"):
        st.write("Popover body text")
    with st.popover("Copy a"):
        st.code("a = 2")
with c[2]:
    st.toggle("A toggle"); st.checkbox("A checkbox"); st.radio("Radio", ["one", "two"], horizontal=True)
with st.expander("Open expander (expanded=True)", expanded=True):
    st.write("Body text inside the expander")
    with st.expander("Nested expander"):
        st.write("nested body")
with st.expander("Closed expander"):
    st.write("hidden")
t1, t2, t3 = st.tabs(["Verify", "Explore", "Practice"])
with t1:
    st.info("An info box"); st.success("A success box"); st.warning("A warning box"); st.error("An error box")
    st.caption("A caption")
with t2:
    st.metric("Metric", "42", "+3")
    st.metric("Falling metric", "7", "-2")
    st.dataframe(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    st.table(pd.DataFrame({"a": [1, 2]}))
    st.code("print('code block')")
    st.json({"k": [1, 2]})
with t3:
    st.selectbox("Select", ["alpha", "beta"]); st.multiselect("Multi", ["a", "b"], default=["a"])
    st.text_input("Text", "value"); st.number_input("Number", value=3); st.text_area("Area", "words")
    st.slider("Slider", 0, 10, 4); st.select_slider("Select slider", [1, 2, 3], value=2)
    st.file_uploader("Upload"); st.date_input("Date"); st.color_picker("Colour")
    st.progress(40, text="progress"); st.segmented_control("Seg", ["a", "b"], default="a")
with st.status("A status box", expanded=True):
    st.write("inside status")
st.toast("A toast message", icon="✅")
badge_row([("badge", "neutral"), ("pass", "pass")])
st.chat_message("assistant").write("chat bubble")
# charts are SVG: their text is coloured by `fill`, not CSS `color`, and the app themes them in Python (ui/a11y.show_figure), so they need their own coverage
import numpy as np
from modules.plotter import build_motion_diagram
from ui.a11y import show_figure
import plotly.graph_objects as go
t = np.linspace(0, 6, 61)
show_figure(build_motion_diagram(t, 8 * t + t ** 2, 8 + 2 * t, x_label="position", x_unit="m", t_unit="s", a_values=np.full_like(t, 2.0), n_strobes=6), width="stretch", key="gallery_motion")
fig = go.Figure(go.Scatter(x=t, y=np.sin(t), name="sin"))
fig.update_layout(title="A line chart", xaxis_title="time (s)", yaxis_title="value")
show_figure(fig, width="stretch", key="gallery_line")
@st.dialog("Share this solution", width="large")
def share_dialog():
    """A stand-in for ui/share.py's dialog: a modal is its own surface (white in Streamlit's light theme), with tabs, labels, captions, checkboxes and download buttons."""
    t1, t2, t3, t4 = st.tabs(["⬇️ Export", "🖼️ Answer card", "📋 Copy", "🔗 Send to a chain"])
    with t1:
        st.selectbox("What to include", ["Full audit", "Lab report", "Homework solution"], help="The sections each report contains")
        st.checkbox("Number the figures and add an automatic description under each", help="Figure 1. …")
        cols = st.columns(3)
        for i, name in enumerate(["Problem", "Results", "Derived equations", "Variables", "Verification detail", "Step-by-step solution"]):
            cols[i % 3].checkbox(name, value=True, key=f"dlg_{i}")
        st.caption("Nothing to include for: Matrix representation, Vectors, Plots.")
        st.markdown("Tip: 📷 “Include this plot in the report” under a plot adds it to the Plots section.")
        with st.expander("Details"):
            st.write("Expander text inside the dialog")
        with st.popover("Options"):
            st.write("Popover text inside the dialog")
        st.markdown("**Download** -- built when you click")
        c = st.columns(3)
        c[0].download_button("PDF", data=b"x", file_name="a.pdf"); c[0].caption("Typeset equations; fixed layout for printing or sending.")
        c[1].button("Word", disabled=True); c[1].caption("Unavailable: pip install python-docx (python-docx: Word (.docx) export).")
    with t2:
        st.caption("One image with the problem, the answer and the verdict.")
        st.toggle("Dark card"); st.code("a = 2 m/s²", language=None)
    with t3:
        st.text_area("Text", "a = 2"); st.radio("Format", ["Plain", "LaTeX"], horizontal=True)
    with t4:
        st.text_input("Chain name"); st.button("Send")


if st.query_params.get("dialog") == "1":
    share_dialog()
st.divider()
st.markdown("**Bold** markdown, `inline code`, [a link](https://example.com), and a table:\n\n| a | b |\n|---|---|\n| 1 | 2 |")

# multi-panel figures: the layouts most likely to put chart text on top of other chart text (a 2-row figure with an axis title and a subplot title between the rows,
# a 3-column one with a long subplot title), drawn through show_figure exactly as the app does, so scripts/chart_text_overlap_scan.py covers them too
from modules import waves as wv, waves_plots as wp
from plotly.subplots import make_subplots
show_figure(wp.diffraction_figure(wv.diffraction_1d("double slit", 550e-9, 1.5, 50e-6, 250e-6)), width="stretch", key="gallery_diffraction")
three = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, subplot_titles=("Norm − 1", "Energy ⟨H⟩ − E₀", "Ehrenfest: ⟨x⟩ and ∫⟨p⟩/m dt"))
for col in (1, 2, 3):
    three.add_trace(go.Scatter(x=[0, 1, 2], y=[0, col, 0], name=f"series {col}"), row=1, col=col)
    three.update_xaxes(title_text="time (s)", row=1, col=col)
show_figure(three, width="stretch", key="gallery_three_panel")
