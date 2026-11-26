"""
A tiny persistent "calculator memory" so a value derived from one problem
(e.g. a computed velocity) can be pulled into a later, unrelated
calculation -- lives in Streamlit's session_state so it survives across
interactions within a session.
"""
from dataclasses import dataclass


@dataclass
class WorkspaceEntry:
    name: str
    value: float
    source: str  # which problem/equation it came from
    unit: str | None = None


class Workspace:
    def __init__(self, session_state):
        self._state = session_state
        if "workspace_entries" not in self._state:
            self._state["workspace_entries"] = {}

    @property
    def entries(self) -> dict[str, WorkspaceEntry]:
        return self._state["workspace_entries"]

    def store(self, name: str, value: float, source: str, unit: str | None = None):
        self.entries[name] = WorkspaceEntry(name=name, value=value, source=source, unit=unit)

    def rename(self, old_name: str, new_name: str) -> tuple[bool, str]:
        new_name = new_name.strip()
        if old_name not in self.entries:
            return False, f"'{old_name}' not found."
        if not new_name:
            return False, "Name can't be empty."
        if not new_name.isidentifier():
            return False, f"'{new_name}' isn't a valid variable name (letters/numbers/underscore, can't start with a digit)."
        if new_name != old_name and new_name in self.entries:
            return False, f"'{new_name}' already exists in the workspace."
        if new_name == old_name:
            return True, ""
        entry = self.entries.pop(old_name)
        entry.name = new_name
        self.entries[new_name] = entry
        return True, ""

    def remove(self, name: str):
        self.entries.pop(name, None)

    def as_context_string(self) -> str | None:
        """Renders stored values as text the LLM can use as known inputs
        when a new problem references them by name."""
        if not self.entries:
            return None
        return "\n".join(
            f"- {e.name} = {e.value:.6g} {e.unit or ''} (previously solved from: {e.source})"
            for e in self.entries.values()
        )

    def as_substitution_dict(self) -> dict[str, float]:
        return {k: v.value for k, v in self.entries.items()}


# ---------------------------------------------------------------------------------------------- binding workspace values after extraction
import re  # noqa: E402

_QUOTES = "'\"“”‘’`"
_BACK_REFERENCE = re.compile(r"\b(workspace|from before|earlier|previous(?:ly)?|stored|saved|last result)\b", re.IGNORECASE)


def references_workspace_value(problem_text: str, name: str) -> tuple[bool, bool]:
    """(explicit, likely): does the problem text point at the workspace value called `name`?
    explicit  the name is written in quotes ('a'), the way a person names a workspace variable.
    likely    a name of two or more characters appears as a word AND the text says "workspace", "from before", "earlier", ... (a single letter is not matched this way: 'a' is also an article)."""
    quoted = re.search(rf"[{_QUOTES}]\s*{re.escape(name)}\s*[{_QUOTES}]", problem_text) is not None
    word = len(name) >= 2 and re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", problem_text) is not None
    return quoted, bool(word and _BACK_REFERENCE.search(problem_text))


def bind_workspace_values(model, entries: dict[str, WorkspaceEntry] | None, problem_text: str) -> list[str]:
    """Give the model's inputs the workspace values the problem refers to, whatever the language model did with them.

    The model is TOLD the workspace values (`as_context_string`), but a model that ignores them leaves the variable unknown (the symptom: "using the acceleration ('a' in workspace) from before"
    came back with a = 0 unknown, and the problem could not be solved). So after extraction, for each workspace name the problem points at (see references_workspace_value) that the model
    has as an input without a value, the value is filled in from the workspace, its unit is filled in if the model gave none, and the name is removed from `solve_for` when the problem named it explicitly.
    Returns a note per binding (also added to model.assumptions, so it is visible in the result). Never overwrites a value the model did give."""
    notes: list[str] = []
    for name, entry in (entries or {}).items():
        explicit, likely = references_workspace_value(problem_text, name)
        if not (explicit or likely):
            continue
        var = next((v for v in model.variables if v.symbol == name), None)
        if var is None or var.known_value is not None or var.is_function:
            continue
        var.known_value = float(entry.value)
        if not var.unit and entry.unit:
            var.unit = entry.unit
        if explicit and name in model.solve_for and len(model.solve_for) > 1:
            model.solve_for = [s for s in model.solve_for if s != name]
        unit = f" {entry.unit}" if entry.unit else ""
        note = f"{name} = {entry.value:.6g}{unit} was taken from the workspace (from: {entry.source[:60]})"
        model.assumptions.append(note)
        notes.append(note)
    return notes
