"""Icon-only buttons: every button whose label is empty or a single symbol must have a `help` tooltip (it is the only name a mouse user or a screen reader gets),
and the stylesheet must draw a visible keyboard focus ring on buttons."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALLS = {"button", "download_button", "form_submit_button", "link_button", "toggle", "page_link"}


def _icon_only_without_help() -> list[str]:
    found = []
    for path in sorted([*(ROOT / "ui").rglob("*.py"), ROOT / "app.py"]):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CALLS):
                continue
            label = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "label"), None)
            if not (isinstance(label, ast.Constant) and isinstance(label.value, str) and len(label.value.strip()) <= 2):
                continue
            if "help" not in {k.arg for k in node.keywords}:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}  {node.func.attr}({label.value!r})")
    return found


def test_every_icon_only_button_has_a_help_label():
    assert _icon_only_without_help() == []


def test_the_audit_itself_can_fail(tmp_path):
    """Guard against the audit passing because it finds nothing: it does find the history icon buttons."""
    src = (ROOT / "ui" / "history_panel.py").read_text(encoding="utf-8")
    assert src.count('help="') >= 4 and "icon=" in src


def test_buttons_have_a_visible_keyboard_focus_ring():
    import streamlit as st
    from ui import theme
    seen: list[str] = []
    real = st.markdown
    st.markdown = lambda body, *a, **k: seen.append(body)                 # type: ignore[assignment]
    try:
        theme.inject_base_styles()
    finally:
        st.markdown = real                                                  # type: ignore[assignment]
    assert ":focus-visible" in seen[0] and "outline: 2px solid currentColor" in seen[0]
