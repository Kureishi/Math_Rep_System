"""
Shared pieces for the quantum-mechanics modules (quantum_1d, quantum_dynamics, quantum_operators,
quantum_qubits, quantum_perturbation) -- Streamlit-free.

TWO IDEAS RUN THROUGH ALL OF THEM, and both live here:

  * CHECKS. Every numerical result comes with `QuantumCheck`s: statements that must be true if the
    computation is right and that were actually evaluated (orthonormality, a known analytic spectrum,
    norm conservation, the uncertainty relation, a decay rate with a closed form). A check has a verdict
    AND the numbers behind it, so a failure says by how much, not just that. The app never shows a result
    as verified because nothing contradicted it; it shows what was tested.

  * SAFE EXPRESSIONS. Potentials and operator expressions are typed by the user. SymPy's parser evaluates
    what it is given, so text is first walked as a Python syntax tree and refused unless it consists only of
    numbers, whitelisted names and functions, and arithmetic -- no attribute access, no subscripts, no
    lambdas, no strings. Only then does SymPy see it.

Units: the modules take hbar and the mass as plain numbers (default 1, i.e. natural units); where a model
fixes its own units (hydrogen, in atomic units) it says so.
"""
import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

MAX_EXPRESSION_LENGTH = 400


@dataclass
class QuantumCheck:
    label: str
    passed: bool
    detail: str


@dataclass
class CheckList:
    """An ordered list of checks with a one-line verdict."""
    checks: list[QuantumCheck] = field(default_factory=list)

    def add(self, label: str, passed: bool, detail: str) -> bool:
        self.checks.append(QuantumCheck(label, bool(passed), detail))
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def summary(self) -> str:
        return f"{self.n_passed}/{len(self.checks)} checks passed" if self.checks else "no checks ran"


class ExpressionError(ValueError):
    """The text is not an acceptable expression (the message says why, in terms the user can act on)."""


# names usable inside a typed expression, besides the symbols the caller allows
_FUNCTIONS: dict[str, Callable] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "sech": lambda z: 1 / sp.cosh(z), "abs": sp.Abs,
    "Abs": sp.Abs, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sign": sp.sign, "erf": sp.erf,
    "Heaviside": lambda z: sp.Heaviside(z, 0), "heaviside": lambda z: sp.Heaviside(z, 0),
    "Max": sp.Max, "Min": sp.Min,
}
_CONSTANTS: dict[str, sp.Expr] = {"pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv)


def check_expression_syntax(text: str, allowed_names: Iterable[str]) -> ast.Expression:
    """The parsed tree of `text`, or ExpressionError. Only numbers, `allowed_names`, the constants and
    function names above, calls of those functions, and + - * / ** are accepted."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("The expression is empty.")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"The expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    source = text.strip().replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not read '{text}' as an expression ({e.msg}). Write products "
                              "explicitly, e.g. 2*x rather than 2x.") from None
    names = set(allowed_names) | set(_CONSTANTS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"'{text}' uses something that is not allowed in an expression "
                                  f"({type(node).__name__}); only numbers, names, + - * / ** and functions.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)) or \
                isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise ExpressionError(f"'{text}' contains a value that is not a plain number.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
                raise ExpressionError(f"'{text}' calls something that is not an available function "
                                      f"({', '.join(sorted(set(_FUNCTIONS) - {'ln', 'abs', 'heaviside'}))}).")
        elif isinstance(node, ast.Name):
            parent_is_call_target = any(isinstance(p, ast.Call) and p.func is node for p in ast.walk(tree))
            if parent_is_call_target:
                continue
            if node.id not in names:
                raise ExpressionError(f"Unknown name '{node.id}'. Available: "
                                      f"{', '.join(sorted(set(allowed_names)))} and pi, e.")
    return tree


def safe_expression(text: str, allowed_names: Iterable[str]) -> sp.Expr:
    """`text` as a SymPy expression in the given symbol names (see the module docstring for what is accepted)."""
    allowed = list(allowed_names)
    check_expression_syntax(text, allowed)
    local = {name: sp.Symbol(name) for name in allowed}
    local.update(_CONSTANTS)
    local.update(_FUNCTIONS)
    try:
        return parse_expr(text.strip().replace("^", "**"), local_dict=local, transformations=standard_transformations)
    except Exception as e:  # noqa: BLE001
        raise ExpressionError(f"Could not interpret '{text}': {e}") from None


def vectorized(expr: sp.Expr, variable: str, parameters: dict[str, float]) -> Callable[[np.ndarray], np.ndarray]:
    """A numpy function of `variable` for `expr`, with every other symbol replaced by its number in
    `parameters`. A symbol with no value is an error, not a silent zero. The result always has the shape
    of its input (a constant expression is broadcast)."""
    x = sp.Symbol(variable)
    substituted = expr.subs({sp.Symbol(k): v for k, v in parameters.items()})
    unresolved = sorted(str(s) for s in substituted.free_symbols if s != x)
    if unresolved:
        raise ExpressionError(f"No value for: {', '.join(unresolved)}.")
    fn = sp.lambdify(x, substituted, modules=["numpy", {"Heaviside": lambda a, b=0.0: np.heaviside(a, b),
                                                          "sign": np.sign, "erf": _erf}])

    def evaluate(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.asarray(fn(arr))
        return np.broadcast_to(out.real if np.iscomplexobj(out) else out, arr.shape).astype(float)
    return evaluate


def _erf(z):
    from scipy.special import erf
    return erf(z)


def trapezoid_norm(values: np.ndarray, spacing: float) -> float:
    """The integral of |values|^2 on a uniform grid (rectangle rule: exact for the grids used here, whose
    wavefunctions vanish at both ends)."""
    return float(np.sum(np.abs(values) ** 2) * spacing)
