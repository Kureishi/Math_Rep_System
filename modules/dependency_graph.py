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
from dataclasses import dataclass

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
