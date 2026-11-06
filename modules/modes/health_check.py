from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🩺 Health check",
    keywords=("health check", "diagnose", "diagnostics", "not working", "troubleshoot", "setup", "lm studio", "cannot connect", "first run", "model not loaded", "broken"),
    module="ui.health", function="render_health_tab",
    needs_client=True,
    order=180,
)
