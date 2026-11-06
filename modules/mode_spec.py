"""The declaration of one app mode. No imports, so any file can use it without circularity."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    """Everything the app needs to know about a mode, declared in ONE place (its own file).

    label         the exact text shown in the sidebar and used as the mode's identity (emoji + name)
    keywords      words the command palette and the "Ask anything" router match against, beyond the label
    module        dotted path of the module holding the page function (e.g. "ui.discover"); "" for the default page,
                  which app.py renders when no other page matches
    function      the page function's name in that module
    needs_client  True if the page takes the LM Studio client as its argument (it talks to the model)
    order         position in the sidebar: lower comes first. Built-in modes are spaced by 10 so a new one can slot
                  between two others without renumbering anything; plugins default to the end
    """
    label: str
    keywords: tuple[str, ...] = ()
    module: str = ""
    function: str = ""
    needs_client: bool = False
    order: int = 5000

    def problems(self) -> list[str]:
        """What is wrong with this spec (empty if it is fine)."""
        out = []
        if not isinstance(self.label, str) or not self.label.strip():
            out.append("label must be a non-empty string")
        if not isinstance(self.keywords, tuple) or not all(isinstance(k, str) and k.strip() for k in self.keywords):
            out.append("keywords must be a tuple of non-empty strings")
        if not isinstance(self.module, str) or not isinstance(self.function, str):
            out.append("module and function must be strings")
        elif bool(self.module) != bool(self.function):
            out.append("module and function must be given together (or both left empty for the default page)")
        elif self.module and not all(part.isidentifier() for part in self.module.split(".")):
            out.append(f"'{self.module}' is not a dotted module path")
        elif self.function and not self.function.isidentifier():
            out.append(f"'{self.function}' is not a function name")
        if not isinstance(self.needs_client, bool):
            out.append("needs_client must be True or False")
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            out.append("order must be an integer")
        return out
