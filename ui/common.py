"""
Helpers shared by more than one page: upload-size guard, the "include this plot in the
report" / download-format buttons, query-param <-> session-state syncing (so a page URL
restores its mode), click-persisted compute buttons, tooltip headers, live expression
parse preview, and the save/load-template bar.

Moved verbatim out of app.py; the only change is that the previously underscore-private
ones (persist_on_click, tooltip_header, ...) lost the underscore now that they are imported
across modules.
"""
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from modules.templates import save_template, list_templates, load_template
from modules.exporter import PlotSnapshot


LIVE_PREVIEW_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB -- matches .streamlit/config.toml's


def restore_from_query_param(key: str) -> None:
    """Call BEFORE creating a widget with this key, so a value present
    in the page's URL (from a shared/bookmarked link) becomes that
    widget's initial value. A no-op if the widget already has session
    state (e.g. the person has already interacted with it this
    session) or nothing relevant is in the URL."""
    if key not in st.session_state and key in st.query_params:
        st.session_state[key] = st.query_params[key]


def sync_query_param(key: str) -> None:
    """Call AFTER creating a widget with this key, to write its current
    value into the URL -- so the page's URL can be copied, bookmarked,
    or shared and reopening it restores this exact state (see
    restore_from_query_param, its counterpart on load)."""
    if key in st.session_state:
        st.query_params[key] = str(st.session_state[key])


                                             # server-side maxUploadSize; this is a second,
                                             # defense-in-depth check with a clearer,
                                             # upload-specific message rather than relying
                                             # solely on Streamlit's own generic rejection


def check_upload_size(uploaded_file) -> bool:
    """Returns True if uploaded_file is within MAX_UPLOAD_SIZE_BYTES,
    showing a clear error and returning False otherwise. Call this
    before doing any processing on an uploaded file."""
    if uploaded_file is None:
        return True
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.error(f"'{uploaded_file.name}' is {size_mb:.0f} MB, which is over the 500 MB upload limit. "
                  "Please upload a smaller file.")
        return False
    return True


def snapshot_button(key: str, title: str, caption: str, render_fn):
    """Renders a small 'include this plot in the report' control under a
    plot. render_fn is a zero-arg callable producing PNG bytes -- kept
    lazy so the (potentially slow) matplotlib re-render only happens when
    the user actually opts in, not on every script rerun."""
    existing = st.session_state["plot_snapshots"].get(key)
    if existing:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(f"✅ Included in the exported report as \"{existing.title}\"")
        with c2:
            if st.button("Remove", key=f"remove_snap_{key}"):
                del st.session_state["plot_snapshots"][key]
                st.session_state["pdf_bytes"] = None  # cached PDF is now stale
                st.rerun()
    else:
        if st.button("📸 Include this plot in the report", key=f"include_snap_{key}"):
            try:
                png = render_fn()
                st.session_state["plot_snapshots"][key] = PlotSnapshot(title=title, caption=caption, png_bytes=png)
                st.session_state["pdf_bytes"] = None
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't capture a snapshot of this plot: {e}")


def format_download_button(key: str, file_stem: str, render_fn):
    """A 'download this plot' control offering PNG (raster) or SVG/PDF
    (vector) -- vector formats scale to any size without pixelating,
    which matters for dropping a figure straight into a paper or slide
    deck. Distinct from snapshot_button() above, which always captures
    PNG specifically for embedding in this app's own Markdown/PDF report
    -- this is a direct, ad-hoc download of just this one plot, in
    whichever format the person actually wants it in. render_fn is a
    one-arg callable: fmt -> bytes."""
    fmt = st.selectbox("Format", ["png", "svg", "pdf"], key=f"fmt_{key}", label_visibility="collapsed")
    mime = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[fmt]
    if st.button(f"⬇️ Download plot ({fmt.upper()})", key=f"download_{key}"):
        try:
            data = render_fn(fmt)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't render this plot as {fmt}: {e}")
        else:
            st.download_button(f"Save {file_stem}.{fmt}", data=data, file_name=f"{file_stem}.{fmt}",
                                 mime=mime, key=f"save_{key}")


def persist_on_click(button_label: str, button_key: str, session_key: str, ready: bool, compute_fn):
    """Runs compute_fn() and stores the result in st.session_state when
    the button is clicked, then ALWAYS reads back from session_state
    (returning None if nothing's been computed yet) rather than only
    rendering within the same script-run the button was clicked in.

    This fixes a real, previously-shipped bug (found while adding the
    statistical-inference layer to the curve-fitting tab): st.button()
    only returns True on the EXACT rerun triggered by clicking it -- any
    OTHER widget interaction on the page (a slider, a different button,
    even an unrelated checkbox) sees it as False on that rerun. Gating
    a whole result display on `if st.button(...):` directly means the
    result silently disappears the moment the person touches any other
    interactive element added alongside it (e.g. a time-animation slider
    on the result itself) -- exactly the kind of thing this round of
    interactivity upgrades adds throughout the PDE and tensor-calculus
    tabs, so every "Solve"/"Analyze" button in both was converted to
    this pattern rather than just the ones visibly breaking today."""
    if st.button(button_label, key=button_key) and ready:
        st.session_state[session_key] = compute_fn()
    return st.session_state.get(session_key)


def tooltip_header(text: str, explanation: str, level: str = "**") -> None:
    """A hover-tooltip label for a STATIC piece of text (a results
    header, a jargon term) -- distinct from the existing st.caption()
    pattern used throughout this app, which is always-visible inline
    text rather than something a person hovers to reveal. Streamlit's
    `help=` parameter (used elsewhere in this file) only exists on
    interactive widgets, not on st.write/st.markdown output, so a
    static header needs a different mechanism: the browser's own
    native title-attribute tooltip via a plain HTML span, which needs
    no JS and degrades harmlessly (just shows no tooltip) anywhere
    that doesn't render it."""
    import html
    st.markdown(f'<span title="{html.escape(explanation)}" style="border-bottom: 1px dotted; cursor: help;">'
                f'{level}{html.escape(text)}{level}</span>', unsafe_allow_html=True)


def live_parse_preview(expr_str: str, extra_symbols: list[str] | None = None) -> None:
    """Best-effort inline feedback on a symbolic-expression text input,
    shown immediately below it: the parsed LaTeX if it parses cleanly,
    or a plain-language parse error if not -- so a typo shows up right
    where it was made instead of only after clicking Solve/Compute.
    Deliberately silent (no output at all) for an empty string, so a
    not-yet-filled-in field doesn't show an error before the person has
    had a chance to type anything."""
    if not expr_str or not expr_str.strip():
        return
    try:
        local_dict = {s: sp.Symbol(s) for s in (extra_symbols or [])}
        parsed = parse_expr(expr_str, local_dict=local_dict,
                              transformations=LIVE_PREVIEW_TRANSFORMS)
        st.caption(f"parsed as: \\({sp.latex(parsed)}\\)")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"⚠️ doesn't parse yet: {exc}")


def render_template_bar(category: str, collect_fn):
    """A compact 'save current inputs as template' / 'load a saved
    template' bar, reusable across modes. `collect_fn()` returns
    {session_state_key: value} for whatever this mode's own widgets are
    keyed by -- loading a template just writes its payload straight
    back into those same keys, before this mode's own widgets are
    (re)created further down the SAME function -- the same "write
    session_state before the widget exists this run" approach
    render_quick_start_tab() established, just simpler here: a template
    bar only needs to beat its OWN mode's widgets (created later in the
    same function), not cross into a different mode's (Quick Start's
    case, which needed the pending-state+rerun indirection because the
    sidebar's mode radio is created earlier in the script than ANY
    mode's own body)."""
    with st.expander("📋 Templates"):
        col1, col2 = st.columns(2)
        with col1:
            existing = list_templates(category)
            options = {f"{t.name} ({t.created_at[:10]})": t.id for t in existing}
            if options:
                choice = st.selectbox("Load a saved template", list(options.keys()),
                                        key=f"{category}_template_choice")
                if st.button("Load", key=f"{category}_template_load"):
                    tpl = load_template(options[choice])
                    if tpl is not None:
                        for k, v in tpl.payload.items():
                            st.session_state[k] = v
                        st.rerun()
            else:
                st.caption("No saved templates yet.")
        with col2:
            new_name = st.text_input("Save current inputs as...", key=f"{category}_template_name")
            if st.button("Save", key=f"{category}_template_save") and new_name.strip():
                save_template(new_name, category, collect_fn())
                st.success(f"Saved '{new_name}'.")
                st.toast(f"Template '{new_name}' saved", icon="📋")
