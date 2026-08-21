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
