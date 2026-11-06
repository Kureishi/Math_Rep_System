# Plugins: your own modes, without touching the project

Any `*.py` file in this folder whose name does not start with `_` is loaded at startup and may add one or more modes. The built-in
modes work the same way (they are the files in `modules/modes/`); a plugin is simply a mode that lives outside the repository's
tracked files (`plugins/*.py` is git-ignored, except the `_example*.py` files).

**To add one:** copy `_example_mode.py` to `my_mode.py`, edit it, restart the app. Fix `module="plugins.my_mode"` to match the file name.

A plugin file defines `MODE = ModeSpec(...)` (or `MODES = [...]` for several) and the page function it names (`function=...`). The
fields are documented in `modules/mode_spec.py`. The page runs inside Streamlit like any other; it receives the LM Studio client if
`needs_client=True`.

**If a plugin is broken** the app still starts: that file is skipped and the reason is shown on the Health check page. A plugin cannot
reuse a built-in mode's label (the built-in wins).

**Order:** `order` sets the sidebar position (built-ins are 10, 20, ... 240; plugins default to 5000, after them).
