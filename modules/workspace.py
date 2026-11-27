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
        # NOT the text of the problem each value came from: a model reading "8 m/s ... over 6 seconds" there took those numbers (v0 = 8, t = 6) into the NEW problem.
        return "\n".join(f"- {e.name} = {e.value:.6g} {e.unit or ''} (a value from an earlier calculation)".replace("  ", " ") for e in self.entries.values())

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
    Returns a note per binding (also added to model.assumptions, so it is visible in the result). A value the model gave is kept unless the problem names the workspace variable explicitly (in quotes)."""
    notes: list[str] = []
    for name, entry in (entries or {}).items():
        explicit, likely = references_workspace_value(problem_text, name)
        var = next((v for v in model.variables if v.symbol == name), None)
        if not (explicit or likely) and var is not None and is_follow_up(problem_text):
            # no name in the text, but the variable's MEANING is ("the acceleration from before"): a word of 4+ letters of its description
            lowered = problem_text.lower()
            likely = any(w in lowered for w in re.findall(r"[a-z]{4,}", (var.meaning or "").lower()))
        if not (explicit or likely):
            continue
        if var is None or var.is_function:
            continue
        # A value the model gave stays when the text only LIKELY points at the workspace; an explicit quoted name ('a' in workspace) is the person's own instruction, so the workspace
        # value replaces whatever the model put there (the symptom: a model that ignored the workspace and wrote 0 for the unknown).
        if var.known_value is not None and not (explicit and var.known_value != float(entry.value)):
            continue
        replaced = var.known_value
        var.known_value = float(entry.value)
        if not var.unit and entry.unit:
            var.unit = entry.unit
        if explicit and name in model.solve_for and len(model.solve_for) > 1:
            model.solve_for = [s for s in model.solve_for if s != name]
        unit = f" {entry.unit}" if entry.unit else ""
        note = f"{name} = {entry.value:.6g}{unit} was taken from the workspace" + (f" (the model had {replaced:g})" if replaced is not None else "")
        model.assumptions.append(note)
        notes.append(note)
    return notes


# ---------------------------------------------------------------------------------------------- follow-up problems ("the car from before, 10 seconds instead of 6")
_FOLLOW_UP = re.compile(
    r"\b(workspace|(?:from|as|like) before|(?:from )?earlier (?:problem|result|calculation|example)|previous(?:ly)? (?:problem|result|calculation|example)|last (?:problem|result)"
    r"|instead of|rather than|same (?:car|ball|object|setup|system|problem|circuit|mass|spring|rectangle|triangle))\b", re.IGNORECASE)
_REPLACED = re.compile(r"(-?\d+(?:\.\d+)?)\s*[A-Za-z%°/²³^·]*\s+(?:instead of|rather than)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def is_follow_up(problem_text: str) -> bool:
    """Does the text say it builds on something earlier ('from before', 'in workspace', 'instead of 6', 'the same car')?"""
    return bool(_FOLLOW_UP.search(problem_text or ""))


def earlier_entries(previous_model, workspace_entries: dict[str, WorkspaceEntry] | None) -> dict[str, WorkspaceEntry]:
    """The values of the problem solved just before (its given inputs) and of the workspace (results kept from earlier), by symbol: what a follow-up may mean by 'the car from before'.
    The workspace wins where both have a symbol (it holds the later, solved values)."""
    out: dict[str, WorkspaceEntry] = {}
    for v in getattr(previous_model, "variables", None) or []:
        if v.known_value is not None and not v.is_function and not v.is_vector:
            out[v.symbol] = WorkspaceEntry(name=v.symbol, value=float(v.known_value), source="the previous problem", unit=v.unit)
    out.update(workspace_entries or {})
    return out


def earlier_context(previous_text: str | None, entries: dict[str, WorkspaceEntry]) -> str | None:
    """A labelled block for the extraction prompt, added ONLY for a follow-up. It states that these values are context and may be used only if the new problem refers to them, because
    an unlabelled copy of the earlier text made a model take its numbers into the new problem (v0 = 8, t = 6)."""
    if not entries:
        return None
    values = "; ".join(f"{e.name} = {e.value:.6g}{(' ' + e.unit) if e.unit else ''}" for e in entries.values())
    head = f'The earlier problem was: "{previous_text.strip()[:300]}". ' if previous_text else ""
    return (f"EARLIER PROBLEM (context for a follow-up; use it only where the new problem refers to it, e.g. 'the car from before'): {head}Its values: {values}. "
            "Where the new problem gives a different value (\"10 seconds instead of 6\"), use the new one.")


def bind_earlier_values(model, entries: dict[str, WorkspaceEntry] | None, problem_text: str) -> list[str]:
    """For a follow-up, fill the inputs the model left without a value from the earlier problem (same symbol), then apply 'N instead of M' changes.

    Only inputs that are NOT targets and have no value are filled, so nothing the problem or the model stated is overridden; 'N instead of M' changes the one input whose value is M."""
    notes: list[str] = []
    if not is_follow_up(problem_text):
        return notes
    for name, entry in (entries or {}).items():
        var = next((v for v in model.variables if v.symbol == name), None)
        if var is None or var.is_function or var.known_value is not None or name in model.solve_for:
            continue
        var.known_value = float(entry.value)
        if not var.unit and entry.unit:
            var.unit = entry.unit
        unit = f" {entry.unit}" if entry.unit else ""
        notes.append(f"{name} = {entry.value:.6g}{unit} was taken from the earlier problem")
    for m in _REPLACED.finditer(problem_text):
        new, old = float(m.group(1)), float(m.group(2))
        holders = [v for v in model.variables if v.known_value is not None and abs(v.known_value - old) < 1e-12 and not v.is_function]
        if len(holders) == 1 and new != old:
            holders[0].known_value = new
            notes.append(f"{holders[0].symbol} is {new:g}, not {old:g}: the problem says \"{m.group(0).strip()}\"")
    model.assumptions.extend(notes)
    return notes
