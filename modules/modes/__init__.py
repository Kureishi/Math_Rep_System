"""One file per app mode. To add a mode, create a file here with a declaration:

    from modules.mode_spec import ModeSpec

    MODE = ModeSpec(
        label="🧪 My new mode",                      # shown in the sidebar; also the mode's identity
        keywords=("what", "people", "might", "type"),  # for the command palette and the "Ask anything" router
        module="ui.my_mode", function="render_my_mode_tab",
        needs_client=False,                           # True if the page takes the LM Studio client
        order=250,                                    # sidebar position (built-ins are spaced by 10)
    )

and write the page in ui/my_mode.py. Nothing else in the project needs to change: the sidebar, the palette, the router and the
page dispatch are all derived from these files (see modules/mode_registry.py). Files starting with "_" are ignored.

Your own modes that you do not want in the repository can live in the top-level plugins/ folder instead; see plugins/README.md.
"""
