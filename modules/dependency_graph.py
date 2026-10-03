"""
Dependency graph: shows which variables feed into which equations, and
which equations determine which unknowns -- useful once a problem has
several equations (a matrix system, a coupled ODE pair, a chain of
sequential substitutions) and it's not obvious at a glance which pieces
depend on which.

A fixed three-column layout (known inputs -> equations -> unknowns)
rather than a generic force-directed graph: that's literally the
information flow this app's own solving pipeline follows (knowns get
substituted into equations, equations get solved for unknowns), and it
avoids pulling in a graph-layout dependency (e.g. networkx) for graphs
that are always small -- a handful of equations/variables -- and
naturally three-tiered anyway.
"""
from dataclasses import dataclass, field

import sympy as sp

from modules.equation_engine import ProblemModel


@dataclass
class GraphNode:
    id: str
    label: str
    kind: str    # "known" | "unknown" | "equation"
    x: float
    y: float


@dataclass
class GraphEdge:
    source: str  # a GraphNode.id
    target: str  # a GraphNode.id


def build_dependency_graph(model: ProblemModel) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Builds a bipartite known-vars / equations / unknown-vars graph.
    Scoped to "equation"/"ode"/"recurrence"-kind relations with a
    parsed sympy_eq -- inequality-kind relations and objectives don't
    have the same clean "these symbols flow in, this symbol flows out"
    structure to diagram the same way."""
    known_syms = sorted({v.symbol for v in model.variables if v.known_value is not None})
    unknown_syms = sorted({v.symbol for v in model.variables
                            if v.known_value is None and not v.is_function and not v.is_vector})
    eqs = [e for e in model.equations
           if e.kind in ("equation", "ode", "recurrence") and e.sympy_eq is not None]

    nodes: list[GraphNode] = []
    for i, s in enumerate(known_syms):
        nodes.append(GraphNode(id=f"var:{s}", label=s, kind="known", x=0.0, y=float(i)))
    for i, e in enumerate(eqs):
        nodes.append(GraphNode(id=f"eq:{e.name}", label=e.name, kind="equation", x=1.0, y=float(i)))
    for i, s in enumerate(unknown_syms):
        nodes.append(GraphNode(id=f"var:{s}", label=s, kind="unknown", x=2.0, y=float(i)))

    known_set, unknown_set = set(known_syms), set(unknown_syms)
    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str]] = set()

    for e in eqs:
        assert e.sympy_eq is not None  # guaranteed by eqs's filter above
        eq_id = f"eq:{e.name}"
        all_syms = sorted(s.name for s in e.sympy_eq.free_symbols)

        # if this equation is written as "Eq(target, expr)" (a single
        # bare unknown symbol on the LHS -- the common shape for how
        # these equations get extracted), that symbol is what the
        # equation PRODUCES; every other symbol it uses -- known or
        # unknown -- is an INPUT to it. This distinguishes "a flows out
        # of accel" from "a flows INTO disp" for a symbol used in more
        # than one equation, rather than drawing a misleading eq->var
        # edge from every equation that merely references an unknown.
        # For anything not in that shape (e.g. a genuinely coupled
        # system like 2x+3y=8, with no single unknown isolated on one
        # side), falls back to treating every unknown it mentions as
        # something it helps produce.
        produced = None
        if (isinstance(e.sympy_eq.lhs, sp.Symbol) and e.sympy_eq.lhs.name in unknown_set):
            produced = e.sympy_eq.lhs.name

        for sym in all_syms:
            if sym == produced:
                edge = (eq_id, f"var:{sym}")
            elif sym in known_set or sym in unknown_set:
                edge = (f"var:{sym}", eq_id)
            else:
                continue
            if produced is None and sym in unknown_set:
                # no clear single "produced" symbol -- this equation
                # helps determine every unknown it mentions
                edge = (eq_id, f"var:{sym}")
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(GraphEdge(*edge))

    return nodes, edges


# ---------------------------------------------------------------- solve order

@dataclass
class SolveStage:
    """One step of the order in which the graph's quantities become
    determined. `kind` is:
      "solve"          one equation determines one new variable
      "simultaneous"   equations that share several unknowns, determined together
      "underdetermined" like simultaneous but with fewer equations than unknowns
                       (the unknowns are treated as determined so the replay can
                       continue, but nothing actually pins them down)
      "check"          everything this equation could determine is already known,
                       so it can only confirm (or contradict) an earlier result
      "differential"   an ODE or recurrence: determines a function, not a plotted variable
      "unresolved"     what is left once nothing else can be determined (a cycle,
                       or an input that never becomes known); `missing` says which
    """
    index: int
    kind: str
    equations: list[str]                 # equation node ids
    produces: list[str]                  # variable node ids newly determined
    needs: list[str]                     # variable node ids this stage reads
    missing: list[str] = field(default_factory=list)


def solve_order(model: ProblemModel, nodes: list[GraphNode] | None = None,
                 edges: list[GraphEdge] | None = None) -> list[SolveStage]:
    """The order in which the known inputs let each equation, and so each
    unknown, be determined: an equation can run once every variable feeding
    into it is known, and what it produces then feeds the equations after it.

    This is the DEPENDENCY order implied by the graph -- the earliest each
    quantity can be determined from the givens -- not a trace of the
    solver's own internal sequence (which substitutes into each target
    separately). Equations that become ready together are listed in the
    order they appear in the model.

    Readiness is evaluated once per round, and stages within a round are
    processed one after another, so two equations that could each determine
    the same variable give a "solve" and then a "check" rather than two
    solves."""
    if nodes is None or edges is None:
        nodes, edges = build_dependency_graph(model)
    eq_ids = [n.id for n in nodes if n.kind == "equation"]
    eq_kind = {f"eq:{e.name}": e.kind for e in model.equations}
    inputs: dict[str, list[str]] = {eq: [] for eq in eq_ids}
    outputs: dict[str, list[str]] = {eq: [] for eq in eq_ids}
    for edge in edges:
        if edge.source.startswith("var:") and edge.target in inputs:
            inputs[edge.target].append(edge.source)
        elif edge.source in outputs and edge.target.startswith("var:"):
            outputs[edge.source].append(edge.target)

    determined = {n.id for n in nodes if n.kind == "known"}
    remaining = list(eq_ids)
    position = {eq: i for i, eq in enumerate(eq_ids)}
    stages: list[SolveStage] = []

    def needs_of(group: list[str]) -> list[str]:
        seen: list[str] = []
        for eq in group:
            for var in inputs[eq]:
                if var not in seen:
                    seen.append(var)
        return seen

    while remaining:
        ready = [eq for eq in remaining if all(v in determined for v in inputs[eq])]
        if not ready:
            gap = sorted({v for eq in remaining for v in inputs[eq] if v not in determined})
            stages.append(SolveStage(len(stages), "unresolved", list(remaining), [], needs_of(remaining),
                                      missing=gap))
            break

        wave: list[SolveStage] = []
        coupled = [eq for eq in ready if len(outputs[eq]) >= 2 and eq_kind.get(eq) not in ("ode", "recurrence")]
        for eq in ready:
            if eq in coupled:
                continue
            fresh = [v for v in outputs[eq] if v not in determined]
            if eq_kind.get(eq) in ("ode", "recurrence"):
                kind = "differential"
            elif not fresh:
                kind = "check"
            else:
                kind = "solve"
            wave.append(SolveStage(0, kind, [eq], fresh, needs_of([eq])))
            determined.update(fresh)

        # equations sharing unknowns are determined together, as one stage per connected group
        groups: list[list[str]] = []
        for eq in coupled:
            touching = [g for g in groups if any(set(outputs[eq]) & set(outputs[other]) for other in g)]
            merged = [eq] + [other for g in touching for other in g]
            groups = [g for g in groups if g not in touching] + [sorted(merged, key=position.get)]  # type: ignore[arg-type]
        for group in groups:
            unknowns = sorted({v for eq in group for v in outputs[eq]})
            fresh = [v for v in unknowns if v not in determined]
            kind = "simultaneous" if len(group) >= len(unknowns) else "underdetermined"
            if not fresh:
                kind = "check"
            wave.append(SolveStage(0, kind, group, fresh, needs_of(group)))
            determined.update(fresh)

        wave.sort(key=lambda st: min(position[eq] for eq in st.equations))
        for st in wave:
            st.index = len(stages)
            stages.append(st)
        done = {eq for st in wave for eq in st.equations}
        remaining = [eq for eq in remaining if eq not in done]
    return stages


def describe_stage(stage: SolveStage, nodes: list[GraphNode]) -> str:
    """One line saying what a stage does, in the model's own names."""
    label = {n.id: n.label for n in nodes}

    def names(ids: list[str]) -> str:
        return ", ".join(label.get(i, i) for i in ids)

    eqs = names(stage.equations)
    reads = f" (needs {names(stage.needs)})" if stage.needs else ""
    if stage.kind == "solve":
        return f"solve {names(stage.produces)} from {eqs}{reads}"
    if stage.kind == "simultaneous":
        return f"solve {names(stage.produces)} together from {eqs}{reads}"
    if stage.kind == "underdetermined":
        return f"{eqs} has more unknowns ({names(stage.produces)}) than equations -- nothing pins them down{reads}"
    if stage.kind == "check":
        return f"{eqs} only confirms what is already known{reads}"
    if stage.kind == "differential":
        return f"solve the differential/recurrence equation {eqs}{reads}"
    return f"can't determine {eqs}: still needs {names(stage.missing)}"


# ---------------------------------------------------------------- replay frames

@dataclass
class ReplayFrame:
    """What the dependency graph looks like after `index` stages of the
    solve order have run (index 0 = just the givens)."""
    index: int
    title: str
    states: dict[str, str]                       # node id -> "known" | "pending" | "done" | "current" | "blocked"
    active_edges: list[tuple[str, str]]          # the edges the current stage reads along / writes along


def replay_frames(nodes: list[GraphNode], edges: list[GraphEdge], stages: list[SolveStage]) -> list[ReplayFrame]:
    """One frame for the givens plus one per stage, for animating the solve
    order (modules.plotter.build_solve_order_replay)."""
    label = {n.id: n.label for n in nodes}
    kinds = {n.id: n.kind for n in nodes}
    known_ids = [n.id for n in nodes if n.kind == "known"]
    all_ids = [n.id for n in nodes]

    def blank() -> dict[str, str]:
        return {i: ("known" if kinds[i] == "known" else "pending") for i in all_ids}

    frames = [ReplayFrame(0, "Givens: " + (", ".join(label[i] for i in known_ids) or "none"), blank(), [])]
    determined: set[str] = set()
    executed: set[str] = set()
    for k, st in enumerate(stages, start=1):
        states = blank()
        for i in determined | executed:
            states[i] = "done"
        mark = "blocked" if st.kind == "unresolved" else "current"
        for i in st.equations + st.produces:
            states[i] = mark
        active = [(e.source, e.target) for e in edges
                  if (e.target in st.equations and e.source in st.needs)
                  or (e.source in st.equations and e.target in st.produces)]
        frames.append(ReplayFrame(k, f"Step {k} of {len(stages)}: {describe_stage(st, nodes)}", states, active))
        determined.update(st.produces)
        executed.update(st.equations)
    return frames
