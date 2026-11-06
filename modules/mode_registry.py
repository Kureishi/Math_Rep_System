"""
Mode discovery -- one file per mode, nothing shared to edit.

The sidebar's mode list, the command palette, the "Ask anything" router and the page dispatch table all used to be maintained
by hand in two shared files, so every new mode edited the same lines as every other, which is where most merge conflicts came
from. They are now DERIVED from the mode declarations this module discovers:

    modules/modes/*.py   the built-in modes, one tiny file each: `MODE = ModeSpec(...)`  (or `MODES = [...]`)
    plugins/*.py         optional, yours: the same declaration, in a file that can also hold the page function itself

To add a mode: create one file with a `MODE = ModeSpec(...)`. Nothing else changes.

Failure is contained. A file that cannot be imported, declares no mode, declares an invalid one, or reuses a label is skipped
with a message in `REGISTRY.errors` (shown by the Health check), and every other mode still loads: a broken plugin must not take
the app down. A built-in file with an error is a bug and is caught by tests/test_mode_registry.py.
"""
import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from modules.mode_spec import ModeSpec

ROOT = Path(__file__).resolve().parent.parent
BUILTIN_PACKAGE = "modules.modes"
PLUGIN_DIR = ROOT / "plugins"
PLUGIN_PACKAGE = "plugins"


@dataclass
class Registry:
    specs: list[ModeSpec] = field(default_factory=list)       # in display order
    errors: list[str] = field(default_factory=list)           # one message per skipped file or mode
    sources: dict[str, str] = field(default_factory=dict)     # label -> the file that declared it
    plugin_labels: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.specs]

    def spec(self, label: str) -> ModeSpec:
        return next(s for s in self.specs if s.label == label)


def _specs_of(module: ModuleType, source: str, errors: list[str]) -> list[ModeSpec]:
    """The ModeSpec(s) a module declares through MODE or MODES."""
    found: list = []
    if hasattr(module, "MODE"):
        found.append(module.MODE)
    if hasattr(module, "MODES"):
        try:
            found.extend(module.MODES)
        except TypeError:
            errors.append(f"{source}: MODES must be a list of ModeSpec")
            return []
    if not found:
        errors.append(f"{source}: declares no mode (define MODE = ModeSpec(...) or MODES = [...])")
        return []
    out = []
    for spec in found:
        if not isinstance(spec, ModeSpec):
            errors.append(f"{source}: MODE/MODES must contain ModeSpec objects, not {type(spec).__name__}")
            continue
        out.append(spec)
    return out


def _load_package(package: str, errors: list[str]) -> list[tuple[ModeSpec, str, ModuleType]]:
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{package}: cannot be imported ({type(exc).__name__}: {exc})")
        return []
    out = []
    for info in sorted(pkgutil.iter_modules(getattr(pkg, "__path__", [])), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        name = f"{package}.{info.name}"
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- one broken file must not stop the rest
            errors.append(f"{name}: cannot be imported ({type(exc).__name__}: {exc})")
            continue
        out += [(spec, name, module) for spec in _specs_of(module, name, errors)]
    return out


def _load_plugins(directory: Path, errors: list[str]) -> list[tuple[ModeSpec, str, ModuleType]]:
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        name = f"{PLUGIN_PACKAGE}.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError("not a loadable Python file")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module                       # so the page can be imported by name later
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(name, None)
            errors.append(f"plugin {path.name}: cannot be loaded ({type(exc).__name__}: {exc})")
            continue
        out += [(s, f"plugin {path.name}", module) for s in _specs_of(module, f"plugin {path.name}", errors)]
    return out


def build(candidates: Iterable[tuple[ModeSpec, str, ModuleType]], errors: list[str], plugin_sources: set[str] | None = None) -> Registry:
    """Validate the declarations, reject duplicates (the first declaration of a label wins) and sort into display order."""
    reg = Registry(errors=errors)
    taken: dict[str, str] = {}
    for spec, source, module in candidates:
        bad = spec.problems()
        if bad:
            errors.append(f"{source}: invalid mode {getattr(spec, 'label', '?')!r}: " + "; ".join(bad))
            continue
        if spec.label in taken:
            errors.append(f"{source}: the label {spec.label!r} is already used by {taken[spec.label]}; skipped")
            continue
        if spec.module == module.__name__ and not callable(getattr(module, spec.function, None)):
            errors.append(f"{source}: {spec.label!r} names a page function {spec.function!r} that does not exist in {module.__name__}")
            continue
        taken[spec.label] = source
        reg.specs.append(spec)
        if plugin_sources and source in plugin_sources:
            reg.plugin_labels.append(spec.label)
    reg.specs.sort(key=lambda s: (s.order, s.label))
    reg.sources = taken
    return reg


def discover(package: str = BUILTIN_PACKAGE, plugin_dir: Path | None = PLUGIN_DIR) -> Registry:
    """Find every declared mode: the built-ins in `package`, then any plugins in `plugin_dir`."""
    errors: list[str] = []
    built_in = _load_package(package, errors)
    plugins = _load_plugins(plugin_dir, errors) if plugin_dir is not None else []
    return build(built_in + plugins, errors, {src for _, src, _ in plugins})


REGISTRY: Registry = discover()
