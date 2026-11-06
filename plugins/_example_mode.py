"""An example plugin: a complete, working mode in one file.

HOW TO USE IT: copy this file to plugins/my_mode.py (a name NOT starting with "_") and restart the app: a new mode appears in
the sidebar, in the command palette (search "hello") and in the "Ask anything" router. Delete the copy to remove it.

A plugin needs two things: a declaration (`MODE = ModeSpec(...)`) and the page function it names. They can live in the same file,
as here, with `module` set to "plugins.<this file's name without .py>".
"""
import streamlit as st

from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="👋 Hello plugin",
    keywords=("hello", "example", "plugin", "greeting"),
    module="plugins.my_mode", function="render_hello_tab",     # <- the name this file has once you copy it
    needs_client=False,
    order=9000,
)


def render_hello_tab() -> None:
    st.subheader("👋 Hello from a plugin")
    st.write("This page lives in plugins/my_mode.py. Nothing else in the project was edited to add it.")
    name = st.text_input("Your name", key="hello_plugin_name")
    if name:
        st.success(f"Hello, {name}!")
