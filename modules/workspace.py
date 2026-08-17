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

    def remove(self, name: str):
        self.entries.pop(name, None)

    def as_substitution_dict(self) -> dict[str, float]:
        return {k: v.value for k, v in self.entries.items()}
