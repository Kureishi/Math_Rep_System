"""
The sidebar, moved out of app.py as one function. It reads only `client` and `ws`, and hands
back the two values the main area needs (see SidebarState).
"""
import streamlit as st
from config import settings
from modules.llm_client import LMStudioClient
from modules.command_palette import search as palette_search
from modules.project_bundle import export_bundle, import_bundle
from modules.settings_profiles import save_profile, list_profiles, load_profile, delete_profile, apply_profile
from modules import history, chains
from ui.common import restore_from_query_param, sync_query_param
from modules.command_palette import MODE_LABELS
from dataclasses import dataclass
from modules.workspace import Workspace


@dataclass
class SidebarState:
    """What the rest of the script needs back from the sidebar: which mode is selected, and
    whether the LM Studio server is reachable (which gates the Solve button)."""
    mode: str
    ok: bool


def render_sidebar(client: LMStudioClient, ws: Workspace) -> SidebarState:
    """The whole left sidebar: command-palette jump box, dark-mode toggle, mode navigation,
    unit-system preference, LM Studio connection status, advanced settings, settings profiles,
    problem history, workspace variables, active chain, recent error patterns, project bundle
    export/import."""
    with st.sidebar:
        # ---- command palette: fuzzy-searchable quick jump across modes.
        # Placed BEFORE the mode radio deliberately: a "jump" click can then
        # set st.session_state["app_mode"] directly, in the SAME run, since
        # no widget with that key has been instantiated yet this run --
        # unlike Quick Start's cross-mode jumps (triggered from deep inside
        # a DIFFERENT mode's own body, rendered well after the radio
        # already exists), which need the pending-state+rerun indirection
        # near the top of this file. See command_palette.py for the fuzzy-
        # matching logic and why this is scoped to mode-level jumps only
        # (Streamlit's st.tabs() has no "jump to sub-tab" API to hook into).
        st.text_input("🔍 Jump to... (Ctrl/Cmd+K)", key="palette_query",
                       placeholder="e.g. heat equation, bayesian, curvature")
        palette_query = st.session_state.get("palette_query", "")
        if palette_query.strip():
            palette_matches = palette_search(palette_query, limit=4)
            if palette_matches:
                for mode_option in palette_matches:
                    if st.button(mode_option, key=f"palette_jump_{mode_option}", width="stretch"):
                        st.session_state["app_mode"] = mode_option
                        st.rerun()
            else:
                st.caption("No matching mode.")
        # best-effort Ctrl/Cmd+K focus shortcut: reaches out of the
        # component's sandboxed iframe into the parent page (a standard, if
        # slightly fragile, trick for this in Streamlit -- there's no
        # supported API for a custom global keyboard shortcut) to focus the
        # search box above. If a future Streamlit DOM structure change
        # breaks the selector, this silently just doesn't focus anything --
        # it can't throw a visible error into the app either way.
        st.iframe(src="""
            <script>
            (function() {
                const doc = window.parent.document;
                doc.addEventListener('keydown', function(e) {
                    const isK = e.key === 'k' || e.key === 'K';
                    if ((e.metaKey || e.ctrlKey) && isK) {
                        e.preventDefault();
                        const inputs = doc.querySelectorAll('input[type="text"]');
                        for (const el of inputs) {
                            if (el.placeholder && el.placeholder.includes('heat equation')) {
                                el.focus();
                                break;
                            }
                        }
                    }
                });
            })();
            </script>
        """, height=1)

        # ---- best-effort in-app dark mode: Streamlit's OWN theme setting
        # (hamburger menu -> Settings -> Choose app theme) already supports
        # light/dark/system, but it's not very discoverable and can't be
        # toggled from Python code -- this is a separate, explicit,
        # visible switch that injects CSS overrides at runtime instead.
        # Targets stable data-testid selectors (Streamlit's own recommended
        # hook for custom CSS across versions) rather than CSS custom
        # properties, since there's no way to verify from here which
        # variable names a given Streamlit version actually exposes for
        # override -- data-testid attributes are the more reliably stable
        # target. Like the Ctrl+K shortcut above, this can't raise a
        # visible error even if a future Streamlit DOM change breaks a
        # selector -- it would just silently stop visually applying.
        st.session_state.setdefault("dark_mode", False)
        st.checkbox("🌙 Dark mode", key="dark_mode")
        if st.session_state["dark_mode"]:
            st.markdown("""
                <style>
                [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                    background-color: #0E1117;
                    color: #E8E8E8;
                }
                [data-testid="stSidebar"] {
                    background-color: #1C1F26;
                    color: #E8E8E8;
                }
                [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
                [data-testid="stCaptionContainer"], label, .stMarkdown, h1, h2, h3, h4, h5, h6 {
                    color: #E8E8E8 !important;
                }
                [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
                [data-testid="stTextArea"] textarea, [data-baseweb="select"] {
                    background-color: #262B36 !important;
                    color: #E8E8E8 !important;
                }
                </style>
            """, unsafe_allow_html=True)

        # ---- navigation: which tool is active. Kept near the very top of
        # the sidebar (rather than a horizontal radio competing with the main
        # input box, as it used to be) so switching tools doesn't require
        # scrolling past whatever's currently in the main content area.
        restore_from_query_param("app_mode")
        mode = st.radio("Mode", MODE_LABELS, key="app_mode")
        sync_query_param("app_mode")

        # ---- global unit-system preference: applied wherever a numeric
        # answer's unit is already known (currently the word-problem-solver
        # results and dimensional analysis) via unit_conversion.preferred_conversion,
        # so switching this once doesn't mean re-choosing "show me in feet"
        # every single time -- see unit_conversion.py's module additions.
        st.session_state.setdefault("preferred_unit_system", "None")
        st.selectbox("Preferred unit system", ["None", "SI", "imperial"],
                      key="preferred_unit_system",
                      help="When an answer's unit is known, also highlight it converted to this "
                           "system -- leave as None to just see the answer in whatever unit the "
                           "problem itself used.")

        # ---- persistent status panels: these matter across MULTIPLE
        # problems in a session (a recurring mistake, a chain in progress),
        # not just the one currently on screen, so they live in the sidebar
        # rather than inside a single problem's own tabs/expanders where
        # they'd only be visible while that specific problem is showing.
        # Placed right under navigation -- these are quick-glance, day-to-
        # day status the person actually wants to see without first
        # scrolling past the (set-once, rarely-touched) connection config
        # below, which matters most on a phone's narrower sidebar overlay.
        error_patterns = history.summarize_error_patterns()
        if error_patterns:
            st.divider()
            st.header("📊 Recent patterns")
            for p in error_patterns[:3]:
                st.warning(p["message"])
            st.caption("See the Practice tab on a solved problem to target these with a worksheet.")

        active_chain_id = st.session_state.get("active_chain_id")
        if active_chain_id is not None:
            active_chain = chains.load_chain(active_chain_id)
            if active_chain is not None:
                st.divider()
                st.header("🔗 Active chain")
                st.caption(f"**{active_chain.name}** -- {len(active_chain.steps)} step(s)")
                for step in active_chain.steps:
                    icon = {"ok": "✅", "error": "❌", "stale": "➖"}.get(step.status, "➖")
                    value = f"{step.output_value:.6g}" if step.output_value is not None else "--"
                    st.caption(f"{icon} step {step.position + 1}: {step.output_symbol} = {value}")
                if st.button("Open in Problem chains", key="sidebar_open_chain",
                              on_click=lambda: st.session_state.update(app_mode="🔗 Problem chains")):
                    st.rerun()

        st.divider()

        # ---- LM Studio connection: collapsed by default once things are
        # working -- this is "set once and forget" content, and unlike the
        # status panels above it doesn't change from problem to problem, so
        # it shouldn't cost a full screen of scrolling on every visit. Left
        # expanded automatically when there's actually a problem to see.
        ok, msg = client.is_available()
        with st.expander("LM Studio", expanded=not ok):
            (st.success if ok else st.error)(msg)

            loaded_models = client.list_models() if ok else []
            if ok and not loaded_models:
                st.warning("Connected, but no models are loaded. Load one in LM Studio's Developer tab.")
            elif loaded_models:
                # Fall back to whatever's actually loaded if config.py's default
                # isn't among the currently-served models, instead of silently
                # trying to call a model that doesn't exist.
                default_reasoning = settings.reasoning_model if settings.reasoning_model in loaded_models else loaded_models[0]
                settings.reasoning_model = st.selectbox(
                    "Reasoning model", loaded_models,
                    index=loaded_models.index(default_reasoning),
                    help="Used for equation extraction, verification cross-checks, narration, and scenarios.",
                )

                default_vision = settings.vision_model if settings.vision_model in loaded_models else loaded_models[0]
                settings.vision_model = st.selectbox(
                    "Vision model", loaded_models,
                    index=loaded_models.index(default_vision),
                    help="Used to transcribe problem statements from uploaded images. Pick a multimodal "
                         "model here -- a text-only model will error on image input; use OCR fallback instead.",
                )

        with st.expander("⚙️ Advanced settings"):
            st.caption("Tune verification strictness and generation behavior without editing config.py "
                        "or restarting the app. Changes apply to the next problem you solve.")

            settings.temperature_extraction = st.slider(
                "Extraction temperature", 0.0, 1.0, settings.temperature_extraction, 0.05,
                help="How much freedom the model has when converting text into equations. Lower is more "
                     "deterministic and faithful to the problem; raise it only if extraction feels too rigid.",
            )
            settings.temperature_narration = st.slider(
                "Narration temperature", 0.0, 1.0, settings.temperature_narration, 0.05,
                help="Controls the wording of step-by-step explanations only -- never affects the math "
                     "itself, since SymPy computes that independently.",
            )
            settings.max_verification_retries = st.number_input(
                "Max verification retries", min_value=0, max_value=5,
                value=settings.max_verification_retries, step=1,
                help="How many times to re-prompt the model with the failure reason if self-verification "
                     "fails, before giving up and showing the result with a warning.",
            )
            settings.numeric_tolerance = st.select_slider(
                "Numeric balance tolerance",
                options=[1e-9, 1e-6, 1e-3, 1e-2],
                value=settings.numeric_tolerance if settings.numeric_tolerance in (1e-9, 1e-6, 1e-3, 1e-2) else 1e-6,
                format_func=lambda x: f"{x:.0e}",
                help="How close a residual must be to zero to count as 'balances' when checking an "
                     "equation against known values. Tighter (smaller) catches more subtle errors but "
                     "may flag harmless floating-point rounding as a failure.",
            )
            cross_check_pct = st.slider(
                "Independent cross-check tolerance", 0.5, 10.0, settings.cross_check_tolerance * 100, 0.5,
                format="%.1f%%",
                help="How far apart the derived answer and the independent re-solve can be before "
                     "verification flags a disagreement. Wider tolerates more model imprecision; "
                     "narrower catches subtler derivation errors but may false-flag on rounding.",
            )
            settings.cross_check_tolerance = cross_check_pct / 100
            settings.computation_timeout_seconds = st.slider(
                "Computation timeout (seconds)", 1.0, 60.0, settings.computation_timeout_seconds, 1.0,
                help="How long a single symbolic computation (solving, eigenvalues, dsolve, equivalence "
                     "checking, proof steps, etc.) is allowed to run before it's abandoned as timed out, "
                     "rather than freezing the app. Raise it for problems you know are legitimately heavy "
                     "(large matrices, gnarly ODEs); lower it for a snappier UI on modest hardware. A "
                     "timed-out computation's background thread keeps running until it naturally finishes "
                     "-- Python can't forcibly kill it -- but the UI itself is never blocked past this limit.",
            )

            if st.button("Reset to defaults"):
                from config import Settings
                defaults = Settings()
                settings.temperature_extraction = defaults.temperature_extraction
                settings.temperature_narration = defaults.temperature_narration
                settings.max_verification_retries = defaults.max_verification_retries
                settings.numeric_tolerance = defaults.numeric_tolerance
                settings.cross_check_tolerance = defaults.cross_check_tolerance
                settings.computation_timeout_seconds = defaults.computation_timeout_seconds
                st.rerun()

            # ---- named settings profiles: unlike the sliders above (a
            # single live-tweaked object that resets to config.py's
            # hardcoded defaults every session), these persist across
            # sessions -- switch between e.g. "strict" and "fast
            # exploratory" verification tuning without re-typing every
            # slider by hand each time. See settings_profiles.py.
            st.divider()
            st.caption("Save/load named presets of the settings above -- these persist across "
                        "sessions, unlike the sliders themselves.")
            existing_profiles = list_profiles()

            load_cols = st.columns([2, 1])
            with load_cols[0]:
                selected_profile = st.selectbox("Load a saved profile", ["(choose one)"] + existing_profiles,
                                                  key="settings_profile_select", label_visibility="collapsed")
            with load_cols[1]:
                if st.button("Load", key="settings_profile_load") and selected_profile != "(choose one)":
                    profile = load_profile(selected_profile)
                    if profile is not None:
                        apply_profile(profile, settings)
                        st.rerun()

            save_cols = st.columns([2, 1])
            with save_cols[0]:
                new_profile_name = st.text_input("Save current settings as...",
                                                    key="settings_profile_new_name",
                                                    placeholder="e.g. strict verification",
                                                    label_visibility="collapsed")
            with save_cols[1]:
                if st.button("💾 Save as...", key="settings_profile_save") and new_profile_name.strip():
                    save_profile(new_profile_name, settings)
                    st.toast(f"Saved settings profile '{new_profile_name}'", icon="⚙️")
                    st.rerun()

            if existing_profiles:
                del_cols = st.columns([2, 1])
                with del_cols[0]:
                    delete_target = st.selectbox("Delete a profile", ["(choose one)"] + existing_profiles,
                                                    key="settings_profile_delete_select",
                                                    label_visibility="collapsed")
                with del_cols[1]:
                    if st.button("🗑️ Delete", key="settings_profile_delete") and delete_target != "(choose one)":
                        delete_profile(delete_target)
                        st.rerun()

        st.divider()
        st.header("Variable Workspace")
        if ws.entries:
            for name, entry in list(ws.entries.items()):
                c1, c2 = st.columns([3, 1])
                with c1:
                    new_name = st.text_input(
                        "Name", value=name, key=f"wsname_{name}", label_visibility="collapsed",
                    )
                    st.caption(f"= {entry.value:.6g} {entry.unit or ''}  \n_{entry.source}_")
                with c2:
                    if st.button("✕", key=f"rm_{name}"):
                        ws.remove(name)
                        st.rerun()
                if new_name != name:
                    ok_rename, err = ws.rename(name, new_name)
                    if ok_rename:
                        st.rerun()
                    else:
                        st.warning(err)
            st.caption("Rename a variable above to reuse it under a new name, or reference it by its "
                        "current name in a new problem below (e.g. \"using d = ...\").")
        else:
            st.caption("No stored variables yet. Solve a problem and extract a value to reuse it here.")

        # ---- session/project bundling: everything above (workspace) plus
        # everything below (history) plus every chain gets bundled into ONE
        # portable JSON file -- something that can be archived alongside a
        # paper, emailed to a collaborator, or reloaded on a different
        # machine to pick up exactly where a session left off. None of that
        # currently survives moving machines on its own: history.db/
        # chains.db are local SQLite files, and the workspace is pure
        # in-session state. See project_bundle.py.
        st.divider()
        st.header("📦 Project")
        with st.expander("Export / import a project bundle"):
            st.caption("Bundles your solved-problem history, chains, and workspace into one portable "
                        "file, for archiving, handing off to a collaborator, or picking up on another "
                        "machine.")
            if st.button("Prepare export", key="bundle_export_button"):
                st.session_state["bundle_export_json"] = export_bundle(workspace_entries=ws.entries)
            bundle_export_json = st.session_state.get("bundle_export_json")
            if bundle_export_json:
                st.download_button("⬇️ Download project bundle", data=bundle_export_json,
                                     file_name="math_rep_project.json", mime="application/json",
                                     key="bundle_download")

            st.divider()
            uploaded_bundle = st.file_uploader("Import a project bundle", type=["json"], key="bundle_upload")
            if uploaded_bundle is not None and st.button("Import", key="bundle_import_button"):
                import_summary = import_bundle(uploaded_bundle.getvalue().decode("utf-8"), ws)
                st.success(f"Imported {import_summary.history_imported} problem(s), "
                            f"{import_summary.chains_imported} chain(s), and "
                            f"{import_summary.workspace_imported} workspace value(s).")
                st.toast("Project bundle imported", icon="📦")
                for err in import_summary.errors:
                    st.warning(err)
                if (import_summary.history_imported or import_summary.chains_imported
                        or import_summary.workspace_imported):
                    st.rerun()

        st.divider()
        st.header("History")
        recent = history.list_recent()
        if recent:
            for entry in recent:
                badge = "✅" if entry["passed"] else "⚠️"
                label = entry["problem_text"][:45] + ("..." if len(entry["problem_text"]) > 45 else "")
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    st.caption(f"{badge} **{entry['domain'] or '—'}** -- {entry['timestamp'][:16].replace('T', ' ')}")
                    st.caption(label)
                with c2:
                    if st.button("↺", key=f"load_{entry['id']}", help="Load this problem"):
                        loaded = history.load(entry["id"])
                        if loaded is not None:
                            p_text, l_model, l_report, l_steps, l_scenarios = loaded
                            st.session_state.update(
                                problem_text=p_text, model=l_model, report=l_report,
                                steps=l_steps, scenarios=l_scenarios, pdf_bytes=None, plot_snapshots={},
                            )
                            st.rerun()
                with c3:
                    if st.button("✕", key=f"delhist_{entry['id']}", help="Delete from history"):
                        history.delete(entry["id"])
                        st.rerun()
        else:
            st.caption("No solved problems yet -- they'll be saved here automatically.")
    return SidebarState(mode=mode, ok=ok)
