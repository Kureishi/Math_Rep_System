"""A page that shows one of nearly every Streamlit component, in light or dark mode (?dark=1), under the app's real stylesheet.

    streamlit run scripts/dark_mode_gallery.py --server.port 8599
    python scripts/dark_mode_contrast_scan.py --port 8599          # needs: pip install playwright && playwright install chromium

It is the page scripts/dark_mode_contrast_scan.py measures. Add a component here when the app starts using one the gallery lacks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st
from ui.theme import inject_base_styles, dark_mode_css, badge_row

st.set_page_config(layout="wide")
inject_base_styles()
dark = st.query_params.get("dark") == "1"
if dark:
    st.markdown(dark_mode_css(), unsafe_allow_html=True)
with st.sidebar:
    with st.expander("LM Studio", expanded=True):
        st.success("Connected to LM Studio.")
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
st.divider()
st.markdown("**Bold** markdown, `inline code`, [a link](https://example.com), and a table:\n\n| a | b |\n|---|---|\n| 1 | 2 |")
