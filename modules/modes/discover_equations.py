from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🧭 Discover equations",
    keywords=("sindy", "equation discovery", "discover", "system identification", "data driven", "find the law", "identify dynamics", "time series", "sparse regression", "differential equation from data"),
    module="ui.discover", function="render_discover_tab",
    needs_client=False,
    order=160,
)
