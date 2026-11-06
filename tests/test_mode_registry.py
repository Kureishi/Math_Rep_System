"""Mode discovery: one file per mode, derived everywhere, and a broken file never takes the app down."""
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from modules import command_palette, health, mode_registry as mr
from modules.mode_spec import ModeSpec

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A scratch package of mode files, importable by a unique name, cleaned up afterwards."""
    name = f"scratchmodes_{abs(hash(str(tmp_path))) % 10 ** 8}"
    folder = tmp_path / name
    folder.mkdir()
    (folder / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, folder
    for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[mod]


def spec_file(folder: Path, filename: str, label: str, order: int = 100, keywords="()", extra=""):
    (folder / filename).write_text(textwrap.dedent(f'''
        from modules.mode_spec import ModeSpec
        MODE = ModeSpec(label={label!r}, keywords={keywords}, module="ui.discover", function="render_discover_tab", order={order}){extra}
    '''), encoding="utf-8")


# ------------------------------------------------------------------------------------------ the real modes

def test_the_builtin_modes_all_load_without_errors():
    assert mr.REGISTRY.errors == []
    assert len(mr.REGISTRY.specs) >= 24


def test_labels_and_orders_are_unique_and_the_list_is_sorted():
    specs = mr.REGISTRY.specs
    assert len({s.label for s in specs}) == len(specs)
    builtin = [s for s in specs if s.label not in mr.REGISTRY.plugin_labels]
    assert len({s.order for s in builtin}) == len(builtin)
    assert [(s.order, s.label) for s in specs] == sorted((s.order, s.label) for s in specs)


def test_exactly_one_mode_is_the_default_page_with_no_module():
    default = [s for s in mr.REGISTRY.specs if not s.module]
    assert [s.label for s in default] == ["📝 Word problem solver"]
    assert all(s.module and s.function for s in mr.REGISTRY.specs if s.label != default[0].label)


def test_the_sidebar_palette_and_dispatch_are_all_derived_from_the_same_declarations():
    import ui
    assert command_palette.MODE_LABELS == mr.REGISTRY.labels
    assert [e.mode for e in command_palette._ENTRIES] == mr.REGISTRY.labels
    assert list(ui.PAGE_TARGETS) == [s.label for s in mr.REGISTRY.specs if s.module]
    assert all(ui.PAGE_TARGETS[s.label] == (s.module, s.function, s.needs_client) for s in mr.REGISTRY.specs if s.module)


def test_every_builtin_mode_has_its_own_file_declaring_exactly_that_mode():
    files = sorted(p for p in (ROOT / "modules" / "modes").glob("*.py") if not p.name.startswith("_"))
    assert len(files) == len([s for s in mr.REGISTRY.specs if s.label not in mr.REGISTRY.plugin_labels])


def test_palette_search_still_finds_modes_by_their_keywords():
    assert command_palette.search("heat equation")[0] == "🌡️ PDE solver"
    assert command_palette.search("mcmc")[0] == "🎲 Posterior sampling (MCMC)"


# ------------------------------------------------------------------------------------------ adding a mode adds one file

def test_dropping_in_one_file_adds_a_mode_and_changes_nothing_else(pkg):
    name, folder = pkg
    spec_file(folder, "alpha.py", "🅰️ Alpha", 10, '("first",)')
    before = mr.discover(name, None)
    spec_file(folder, "beta.py", "🅱️ Beta", 20, '("second",)')
    after = mr.discover(name, None)
    assert before.labels == ["🅰️ Alpha"] and after.labels == ["🅰️ Alpha", "🅱️ Beta"] and after.errors == []
    assert after.spec("🅱️ Beta").keywords == ("second",)


def test_modes_are_ordered_by_their_order_then_label_whatever_the_file_names(pkg):
    name, folder = pkg
    spec_file(folder, "a.py", "Zeta", 30)
    spec_file(folder, "b.py", "Alpha", 30)
    spec_file(folder, "c.py", "Mid", 20)
    assert mr.discover(name, None).labels == ["Mid", "Alpha", "Zeta"]


def test_one_file_can_declare_several_modes(pkg):
    name, folder = pkg
    (folder / "many.py").write_text(textwrap.dedent('''
        from modules.mode_spec import ModeSpec
        MODES = [ModeSpec(label="One", module="ui.discover", function="render_discover_tab", order=1),
                 ModeSpec(label="Two", module="ui.discover", function="render_discover_tab", order=2)]
    '''), encoding="utf-8")
    assert mr.discover(name, None).labels == ["One", "Two"]


def test_files_starting_with_an_underscore_are_ignored(pkg):
    name, folder = pkg
    spec_file(folder, "_draft.py", "Draft")
    spec_file(folder, "real.py", "Real")
    assert mr.discover(name, None).labels == ["Real"]


# ------------------------------------------------------------------------------------------ failure is contained

def test_a_file_that_cannot_be_imported_is_skipped_with_a_reason_and_the_rest_load(pkg):
    name, folder = pkg
    (folder / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    spec_file(folder, "fine.py", "Fine")
    reg = mr.discover(name, None)
    assert reg.labels == ["Fine"] and any("broken" in e and "cannot be imported" in e and "SyntaxError" in e for e in reg.errors)


def test_a_file_that_raises_on_import_is_skipped(pkg):
    name, folder = pkg
    (folder / "boom.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    reg = mr.discover(name, None)
    assert reg.labels == [] and any("boom" in e and "RuntimeError" in e for e in reg.errors)


def test_a_file_that_declares_no_mode_is_reported(pkg):
    name, folder = pkg
    (folder / "empty.py").write_text("X = 1\n", encoding="utf-8")
    assert any("declares no mode" in e for e in mr.discover(name, None).errors)


@pytest.mark.parametrize("snippet, message", [
    ('ModeSpec(label="", module="ui.x", function="f")', "label must be a non-empty string"),
    ('ModeSpec(label="A", keywords=["list"], module="ui.x", function="f")', "keywords must be a tuple"),
    ('ModeSpec(label="A", module="ui.x")', "given together"),
    ('ModeSpec(label="A", module="not a module", function="f")', "dotted module path"),
    ('ModeSpec(label="A", module="ui.x", function="f", order="late")', "order must be an integer"),
    ('ModeSpec(label="A", module="ui.x", function="f", needs_client="yes")', "needs_client"),
    ('"just a string"', "must contain ModeSpec"),
])
def test_an_invalid_declaration_is_rejected_with_the_reason(pkg, snippet, message):
    name, folder = pkg
    (folder / "bad.py").write_text(f"from modules.mode_spec import ModeSpec\nMODE = {snippet}\n", encoding="utf-8")
    reg = mr.discover(name, None)
    assert reg.labels == [] and any(message in e for e in reg.errors)


def test_a_duplicate_label_keeps_the_first_and_names_both_files(pkg):
    name, folder = pkg
    spec_file(folder, "first.py", "Same", 10)
    spec_file(folder, "second.py", "Same", 20)
    reg = mr.discover(name, None)
    assert reg.labels == ["Same"] and reg.spec("Same").order == 10
    assert any("already used by" in e and "first" in e and "second" in e for e in reg.errors)


def test_a_missing_package_is_an_error_not_a_crash():
    reg = mr.discover("no_such_package_anywhere", None)
    assert reg.specs == [] and any("cannot be imported" in e for e in reg.errors)


# ------------------------------------------------------------------------------------------ plugins

@pytest.fixture
def plugin_dir(tmp_path):
    folder = tmp_path / "plugdir"
    folder.mkdir()
    yield folder
    for mod in [m for m in sys.modules if m.startswith("plugins.plug_")]:
        del sys.modules[mod]


def write_plugin(folder: Path, stem: str, label: str, function="page", defines=True, order=7000):
    body = f'''
        from modules.mode_spec import ModeSpec
        MODE = ModeSpec(label={label!r}, keywords=("plug",), module="plugins.{stem}", function={function!r}, order={order})
        {"def page(): return 'ok'" if defines else ""}
    '''
    (folder / f"{stem}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_a_plugin_with_its_own_page_function_is_discovered_and_importable_by_name(plugin_dir):
    import importlib
    write_plugin(plugin_dir, "plug_one", "🔌 One")
    reg = mr.discover("modules.modes", plugin_dir)
    assert "🔌 One" in reg.labels and reg.plugin_labels == ["🔌 One"] and reg.errors == []
    assert importlib.import_module("plugins.plug_one").page() == "ok"
    assert reg.labels.index("🔌 One") == len(reg.labels) - 1                          # plugins come after the built-ins


def test_a_plugin_cannot_replace_a_builtin_mode(plugin_dir):
    write_plugin(plugin_dir, "plug_clash", "🧭 Discover equations")
    reg = mr.discover("modules.modes", plugin_dir)
    assert reg.plugin_labels == [] and reg.spec("🧭 Discover equations").module == "ui.discover"
    assert any("already used by" in e for e in reg.errors)


def test_a_plugin_naming_a_function_it_does_not_define_is_rejected(plugin_dir):
    write_plugin(plugin_dir, "plug_nofunc", "🔌 Missing", function="render_it", defines=False)
    reg = mr.discover("modules.modes", plugin_dir)
    assert "🔌 Missing" not in reg.labels and any("does not exist" in e for e in reg.errors)


def test_a_broken_plugin_never_stops_the_app_or_the_other_plugins(plugin_dir):
    (plugin_dir / "plug_bad.py").write_text("import not_a_real_module_xyz\n", encoding="utf-8")
    write_plugin(plugin_dir, "plug_good", "🔌 Good")
    reg = mr.discover("modules.modes", plugin_dir)
    assert "🔌 Good" in reg.labels and any("plug_bad.py" in e and "ModuleNotFoundError" in e for e in reg.errors)
    assert len(reg.specs) >= 25


def test_underscore_plugins_and_a_missing_plugin_folder_are_fine(plugin_dir, tmp_path):
    write_plugin(plugin_dir, "_hidden", "Hidden")
    assert "Hidden" not in mr.discover("modules.modes", plugin_dir).labels
    assert mr.discover("modules.modes", tmp_path / "nope").errors == []


def test_the_shipped_example_plugin_works_once_copied(plugin_dir):
    import importlib
    text = (ROOT / "plugins" / "_example_mode.py").read_text(encoding="utf-8")
    (plugin_dir / "plug_example.py").write_text(text.replace("plugins.my_mode", "plugins.plug_example"), encoding="utf-8")
    reg = mr.discover("modules.modes", plugin_dir)
    assert reg.errors == [] and "👋 Hello plugin" in reg.labels
    assert callable(importlib.import_module("plugins.plug_example").render_hello_tab)


def test_a_plugin_is_findable_by_the_palette_logic(plugin_dir):
    write_plugin(plugin_dir, "plug_find", "🔌 Findable")
    reg = mr.discover("modules.modes", plugin_dir)
    assert "plug" in reg.spec("🔌 Findable").keywords


# ------------------------------------------------------------------------------------------ the health check

def test_the_health_check_reports_loaded_modes_and_skipped_files(monkeypatch):
    ok = health.check_plugins()
    assert ok.status in ("ok", "info") and "modes loaded" in ok.detail
    bad = mr.Registry(specs=list(mr.REGISTRY.specs), errors=["plugin x.py: cannot be loaded (ValueError: nope)"])
    monkeypatch.setattr(mr, "REGISTRY", bad)
    c = health.check_plugins()
    assert c.status == "warn" and "x.py" in c.detail and "plugins/README.md" in c.fix


def test_modespec_problems_is_empty_for_a_good_spec():
    assert ModeSpec(label="A", keywords=("x",), module="ui.discover", function="render_discover_tab").problems() == []
    assert ModeSpec(label="Default").problems() == []
