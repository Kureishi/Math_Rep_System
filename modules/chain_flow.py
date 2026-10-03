"""
Value flow through a problem chain: which step's answer feeds which later
step, and what number it carried.

modules.chains resolves a chain top to bottom, feeding each step's output
into later steps through "upstream" bindings. That is easy to lose track of
in a list of expanders; this lays the chain out as a cascade (each step a
little further down and right than the last), draws every carried value as
a labelled link, and shows the inputs that were typed in by hand as small
nodes above their step -- so the path a number took from the first step to
the last can be followed, and animated one step at a time.

Pure layout and data: no plotting, and no database access (it reads a Chain
already loaded by chains.load_chain).
"""
from dataclasses import dataclass, field

from modules.chains import Chain

COLUMN_WIDTH = 2.0
ROW_DROP = 1.0
LITERAL_RISE = 0.9
LITERAL_GAP = 0.55


@dataclass
class FlowStep:
    position: int
    symbol: str
    value: float | None
    status: str                 # "ok" | "error" | "stale"
    error: str | None
    x: float
    y: float


@dataclass
class FlowLiteral:
    step: int                   # the position of the step this input feeds
    symbol: str
    value: float
    x: float
    y: float


@dataclass
class FlowEdge:
    source: int                 # upstream step position
    target: int                 # downstream step position
    symbol: str                 # the input symbol, in the downstream step
    carried: float | None       # the value that travelled (None if the upstream step has none)
    broken: bool = False
    reason: str = ""


@dataclass
class ChainFlow:
    name: str
    steps: list[FlowStep] = field(default_factory=list)
    literals: list[FlowLiteral] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)


def build_chain_flow(chain: Chain) -> ChainFlow:
    """Lays out `chain` as a cascade. An upstream binding is flagged
    `broken` when it points at a step that doesn't exist, at itself or at a
    later step, or at an upstream step that never produced a value -- the
    same conditions modules.chains.resolve_chain reports as errors."""
    flow = ChainFlow(name=chain.name)
    by_position = {s.position: s for s in chain.steps}
    for step in chain.steps:
        x = step.position * COLUMN_WIDTH
        y = -step.position * ROW_DROP
        flow.steps.append(FlowStep(step.position, step.output_symbol, step.output_value, step.status,
                                    step.error_detail, x, y))
        stack = 0
        for b in step.bindings:
            if b.source == "literal" and b.literal_value is not None:
                flow.literals.append(FlowLiteral(step.position, b.symbol, float(b.literal_value), x,
                                                  y + LITERAL_RISE + LITERAL_GAP * stack))
                stack += 1
            elif b.source == "upstream":
                up = b.upstream_position
                if up is None or up not in by_position:
                    flow.edges.append(FlowEdge(up if up is not None else -1, step.position, b.symbol, None,
                                                broken=True, reason="points at a step that doesn't exist"))
                elif up >= step.position:
                    flow.edges.append(FlowEdge(up, step.position, b.symbol, None, broken=True,
                                                reason="points at itself or a later step"))
                elif by_position[up].output_value is None:
                    flow.edges.append(FlowEdge(up, step.position, b.symbol, None, broken=True,
                                                reason=f"step {up + 1} produced no value"))
                else:
                    flow.edges.append(FlowEdge(up, step.position, b.symbol, by_position[up].output_value))
    return flow


def step_caption(flow: ChainFlow, position: int) -> str:
    """A sentence describing what step `position` received and produced."""
    step = next(s for s in flow.steps if s.position == position)
    parts = []
    for e in flow.edges:
        if e.target == position:
            if e.broken:
                parts.append(f"{e.symbol} could not be carried ({e.reason})")
            else:
                parts.append(f"{e.symbol} = {e.carried:.4g} from step {e.source + 1}")
    parts += [f"{lit.symbol} = {lit.value:.4g} typed in" for lit in flow.literals if lit.step == position]
    received = "; ".join(parts) if parts else "no overridden inputs"
    if step.status == "ok" and step.value is not None:
        result = f"{step.symbol} = {step.value:.6g}"
    elif step.status == "error":
        result = f"failed: {step.error}"
    else:
        result = "not solved yet"
    return f"Step {position + 1} receives {received}; result: {result}."
