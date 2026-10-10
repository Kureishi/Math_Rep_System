"""
Operator algebra for finite-dimensional quantum systems -- exact where it can be, and checked -- Streamlit-free.

Operators are SymPy matrices with exact entries (spin matrices, truncated oscillator operators, or your
own), in units with hbar = 1.

WHAT IT DOES
  * evaluates typed expressions such as `comm(Sx, Sy) - I*Sz`, `dag(A)*A`, `exp(-I*t*Sz)` with a small
    expression evaluator of its own (a syntax tree walked by hand: only matrices, numbers, + - * **, and the
    functions below -- never Python's eval);
  * checks identities (`lhs = rhs`) exactly and shows the difference when they fail;
  * characterises an operator: Hermitian / unitary / normal, trace, determinant, exact eigenvalues and
    eigenvectors, each verified by substitution;
  * tests the uncertainty relations on a state: the Heisenberg-Robertson bound
    ΔA ΔB ≥ ½|<[A, B]>| and the stronger Schrödinger-Robertson bound that adds the covariance term;
  * expands log(e^X e^Y) in nested commutators (Baker–Campbell–Hausdorff) to fifth order and VERIFIES the
    expansion numerically: for matrices scaled by ε the error of the series truncated at order n must shrink
    like ε^(n+1), and the fitted exponent is checked.

THE TRUNCATION CAVEAT. The oscillator's operators are infinite matrices, cut off at some dimension d. The
cut-off breaks the commutator [a, a†] = 1 in the LAST diagonal entry (it comes out as 1 - d there), and
products of several operators carry the artefact a few rows inward. An identity whose only failures sit in
the last rows and columns is reported as consistent with truncation, not as a failure -- and the failing
entries are shown either way.
"""
import ast
import math
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.linalg import expm, logm

from modules.quantum_common import CheckList, ExpressionError, safe_expression
from modules.timeout_utils import ComputationTimeoutError, run_with_timeout

MAX_DIM = 8
MAX_EXACT_EXP_DIM = 4
MAX_BCH_ORDER = 5


# ------------------------------------------------------------------------------------------ the operator library

@dataclass(frozen=True)
class OperatorFamily:
    key: str
    label: str
    description: str
    names: tuple[str, ...]                # the operator names this family provides
    truncated: bool = False


def _spin_matrices(two_j: int) -> dict[str, sp.Matrix]:
    """Jx, Jy, Jz, J+, J-, J2 for spin j = two_j / 2 (hbar = 1), with exact entries, in the basis m = j ... -j."""
    j = sp.Rational(two_j, 2)
    dim = two_j + 1
    ms = [j - k for k in range(dim)]
    jp = sp.zeros(dim)
    for col in range(1, dim):
        m = ms[col]
        jp[col - 1, col] = sp.sqrt(j * (j + 1) - m * (m + 1))
    jm = jp.H
    jx, jy = (jp + jm) / 2, (jp - jm) / (2 * sp.I)
    jz = sp.diag(*ms)
    j2 = sp.simplify(jx * jx + jy * jy + jz * jz)
    return {"Jx": jx, "Jy": jy, "Jz": jz, "Jp": jp, "Jm": jm, "J2": j2}


def _oscillator(dim: int) -> dict[str, sp.Matrix]:
    a = sp.zeros(dim)
    for n in range(1, dim):
        a[n - 1, n] = sp.sqrt(n)
    adag = a.H
    number = sp.diag(*range(dim))
    x = (a + adag) / sp.sqrt(2)
    p = (a - adag) / (sp.sqrt(2) * sp.I)
    return {"a": a, "adag": adag, "N": number, "X": x, "P": p}


def _pauli() -> dict[str, sp.Matrix]:
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    return {"sx": sx, "sy": sy, "sz": sz, "sp": (sx + sp.I * sy) / 2, "sm": (sx - sp.I * sy) / 2}


FAMILIES: dict[str, OperatorFamily] = {f.key: f for f in [
    OperatorFamily("pauli", "Pauli matrices (qubit)", "σx, σy, σz and the raising/lowering σ+, σ-.",
                   ("sx", "sy", "sz", "sp", "sm")),
    OperatorFamily("spin", "Spin j (angular momentum)", "Jx, Jy, Jz, J+, J-, J² for spin j (set 2j).",
                   ("Jx", "Jy", "Jz", "Jp", "Jm", "J2")),
    OperatorFamily("oscillator", "Harmonic oscillator (truncated)",
                   "a, a†, N, X = (a + a†)/√2, P = (a - a†)/(i√2), cut off at a finite dimension.",
                   ("a", "adag", "N", "X", "P"), truncated=True),
]}


def build_family(key: str, size: int = 2) -> dict[str, sp.Matrix]:
    """The named operators of a family. `size` is 2j for spin and the dimension for the oscillator."""
    if key == "pauli":
        return _pauli()
    if key == "spin":
        if not 1 <= size <= MAX_DIM - 1:
            raise ValueError(f"2j must be between 1 and {MAX_DIM - 1}.")
        return _spin_matrices(size)
    if key == "oscillator":
        if not 2 <= size <= MAX_DIM:
            raise ValueError(f"The oscillator dimension must be between 2 and {MAX_DIM}.")
        return _oscillator(size)
    raise ValueError(f"Unknown operator family {key!r}; choose from {', '.join(FAMILIES)}.")


def parse_matrix(rows: Sequence[Sequence[str]]) -> sp.Matrix:
    """A matrix from rows of expression strings (numbers, I, pi, sqrt(...), ...). Square, at most MAX_DIM."""
    if not rows or not rows[0]:
        raise ExpressionError("The matrix is empty.")
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ExpressionError("Operators must be square matrices.")
    if n > MAX_DIM:
        raise ExpressionError(f"Matrices larger than {MAX_DIM} x {MAX_DIM} are not supported.")
    return sp.Matrix([[safe_expression(str(c), []) for c in row] for row in rows])


# ------------------------------------------------------------------------------------------ expression evaluation

Value = Any                               # a SymPy Matrix or a SymPy scalar expression (SymPy is not typed)


def commutator(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return a * b - b * a


def anticommutator(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return a * b + b * a


def _matrix_exp(a: sp.Matrix) -> sp.Matrix:
    if a.shape[0] <= MAX_EXACT_EXP_DIM:
        try:
            return run_with_timeout(lambda: a.exp(), label="matrix exponential", timeout=6.0)
        except ComputationTimeoutError:
            pass
    return sp.Matrix(expm(np.array(a.evalf(), dtype=complex)))


_SCALAR_FUNCS = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "sinh": sp.sinh,
                 "cosh": sp.cosh, "tan": sp.tan}


class _Evaluator:
    def __init__(self, env: dict[str, sp.Matrix], scalars: dict[str, sp.Expr]):
        self.env = env
        self.scalars = {"I": sp.I, "pi": sp.pi, "e": sp.E, **scalars}
        any_matrix = next(iter(env.values()), None)
        self.dim = any_matrix.shape[0] if any_matrix is not None else 1

    def identity(self) -> sp.Matrix:
        return sp.eye(self.dim)

    def eval(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Expression):
            return self.eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("Only plain numbers are allowed.")
            return sp.Rational(str(node.value)) if isinstance(node.value, float) else sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id == "Id":
                return self.identity()
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self.scalars:
                return self.scalars[node.id]
            raise ExpressionError(f"Unknown name '{node.id}'. Available operators: "
                                  f"{', '.join(sorted(self.env))}, Id; constants: I, pi, e.")
        if isinstance(node, ast.UnaryOp):
            value = self.eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ExpressionError("Unsupported unary operator.")
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise ExpressionError(f"Unsupported syntax: {type(node).__name__}.")

    def _as_matrix(self, v: Value) -> sp.Matrix:
        return v if isinstance(v, sp.MatrixBase) else v * self.identity()      # a bare number means number * Id

    def _binop(self, node: ast.BinOp) -> Value:
        left, right = self.eval(node.left), self.eval(node.right)
        lm, rm = isinstance(left, sp.MatrixBase), isinstance(right, sp.MatrixBase)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if lm or rm:
                left, right = self._as_matrix(left), self._as_matrix(right)
            self._same_shape(left, right)
            return left + right if isinstance(node.op, ast.Add) else left - right
        if isinstance(node.op, ast.Mult):
            if lm and rm:
                self._same_shape(left, right)
            return left * right
        if isinstance(node.op, ast.Div):
            if rm:
                raise ExpressionError("Cannot divide by an operator; use inv(A) * B instead.")
            return left / right
        if isinstance(node.op, ast.Pow):
            if rm:
                raise ExpressionError("The exponent must be a number; use exp(A) for an operator exponential.")
            if lm:
                if not (right.is_Integer and 0 <= right <= 12):
                    raise ExpressionError("Operator powers must be integers from 0 to 12.")
                return left ** int(right)
            return left ** right
        raise ExpressionError("Unsupported operator.")

    @staticmethod
    def _same_shape(a: sp.Matrix, b: sp.Matrix) -> None:
        if a.shape != b.shape:
            raise ExpressionError(f"The operators have different sizes ({a.shape[0]} and {b.shape[0]}).")

    def _call(self, node: ast.Call) -> Value:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ExpressionError("Unsupported function call.")
        name, args = node.func.id, [self.eval(a) for a in node.args]
        matrix_funcs = {"comm": 2, "acomm": 2, "dag": 1, "adj": 1, "exp": 1, "tr": 1, "det": 1, "inv": 1, "T": 1}
        if name in matrix_funcs:
            if len(args) != matrix_funcs[name]:
                raise ExpressionError(f"{name} takes {matrix_funcs[name]} argument(s).")
            if name == "exp" and not isinstance(args[0], sp.MatrixBase):
                return sp.exp(args[0])
            ms = [self._as_matrix(a) for a in args]
            if name == "comm":
                self._same_shape(*ms)
                return commutator(*ms)
            if name == "acomm":
                self._same_shape(*ms)
                return anticommutator(*ms)
            if name in ("dag", "adj"):
                return ms[0].H
            if name == "T":
                return ms[0].T
            if name == "exp":
                return _matrix_exp(ms[0])
            if name == "tr":
                return ms[0].trace()
            if name == "det":
                return ms[0].det()
            if ms[0].det() == 0:
                raise ExpressionError("That operator is singular, so it has no inverse.")
            return ms[0].inv()
        if name in _SCALAR_FUNCS and len(args) == 1 and not isinstance(args[0], sp.MatrixBase):
            return _SCALAR_FUNCS[name](args[0])
        raise ExpressionError(f"'{name}' is not available. Use comm, acomm, dag, exp, tr, det, inv, or sin/cos/exp/"
                              "sqrt/log of a number.")


def evaluate(text: str, env: dict[str, sp.Matrix], scalars: dict[str, sp.Expr] | None = None) -> Value:
    """The value of an operator expression. ExpressionError (with a message about what to fix) otherwise."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("The expression is empty.")
    if len(text) > 300:
        raise ExpressionError("The expression is too long.")
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not read '{text}' ({e.msg}). Write products explicitly, e.g. 2*A.") from None
    return _Evaluator(env, scalars or {}).eval(tree)


def _simplified(m: sp.Matrix) -> sp.Matrix:
    return m.applyfunc(lambda e: sp.simplify(sp.expand(e)))


# ------------------------------------------------------------------------------------------ identities

@dataclass
class IdentityResult:
    lhs_text: str
    rhs_text: str
    holds: bool
    difference: sp.Matrix | sp.Expr
    artefact_of_truncation: bool = False
    note: str = ""


def _only_in_last_rows(diff: sp.Matrix, depth: int = 2) -> bool:
    n = diff.shape[0]
    return all(diff[i, j] == 0 or (i >= n - depth and j >= n - depth) for i in range(n) for j in range(n))


def check_identity(lhs: str, rhs: str, env: dict[str, sp.Matrix], truncated: bool = False,
                   scalars: dict[str, sp.Expr] | None = None) -> IdentityResult:
    """Whether `lhs = rhs` holds exactly, for the operators in `env`. A bare number on one side means that
    number times the identity. With `truncated`, a failure confined to the last rows/columns is flagged as an
    artefact of the cut-off rather than a real failure."""
    left, right = evaluate(lhs, env, scalars), evaluate(rhs, env, scalars)
    if isinstance(left, sp.MatrixBase) or isinstance(right, sp.MatrixBase):
        dim = next(iter(env.values())).shape[0]
        left = left if isinstance(left, sp.MatrixBase) else left * sp.eye(dim)
        right = right if isinstance(right, sp.MatrixBase) else right * sp.eye(dim)
        if left.shape != right.shape:
            raise ExpressionError("The two sides are operators of different sizes.")
        diff = _simplified(left - right)
        holds = diff.is_zero_matrix
        artefact = bool(truncated and not holds and _only_in_last_rows(diff))
        note = ("The mismatch sits only in the last basis states: this is the cut-off of an infinite-dimensional "
                "operator, not a failure of the identity." if artefact else "")
        return IdentityResult(lhs, rhs, bool(holds), diff, artefact, note)
    diff_scalar = sp.simplify(left - right)
    return IdentityResult(lhs, rhs, diff_scalar == 0, diff_scalar)


STANDARD_IDENTITIES: dict[str, list[tuple[str, str, str]]] = {
    "pauli": [
        ("[σx, σy] = 2iσz", "comm(sx, sy)", "2*I*sz"),
        ("[σy, σz] = 2iσx", "comm(sy, sz)", "2*I*sx"),
        ("[σz, σx] = 2iσy", "comm(sz, sx)", "2*I*sy"),
        ("{σx, σy} = 0", "acomm(sx, sy)", "0"),
        ("σx σy = iσz", "sx*sy", "I*sz"),
        ("σx² = 1", "sx**2", "Id"),
        ("σy² = 1", "sy**2", "Id"),
        ("σz² = 1", "sz**2", "Id"),
        ("σx σy σz = i", "sx*sy*sz", "I*Id"),
        ("σ+ σ- + σ- σ+ = 1", "sp*sm + sm*sp", "Id"),
        ("[σ+, σ-] = σz", "comm(sp, sm)", "sz"),
        ("e^{iπσx/2} = iσx", "exp(I*pi/2*sx)", "I*sx"),
    ],
    "spin": [
        ("[Jx, Jy] = iJz", "comm(Jx, Jy)", "I*Jz"),
        ("[Jy, Jz] = iJx", "comm(Jy, Jz)", "I*Jx"),
        ("[Jz, Jx] = iJy", "comm(Jz, Jx)", "I*Jy"),
        ("[J², Jx] = 0", "comm(J2, Jx)", "0"),
        ("[J², Jz] = 0", "comm(J2, Jz)", "0"),
        ("[Jz, J+] = J+", "comm(Jz, Jp)", "Jp"),
        ("[Jz, J-] = -J-", "comm(Jz, Jm)", "-Jm"),
        ("[J+, J-] = 2Jz", "comm(Jp, Jm)", "2*Jz"),
        ("J+ J- = J² - Jz² + Jz", "Jp*Jm", "J2 - Jz**2 + Jz"),
        ("J- J+ = J² - Jz² - Jz", "Jm*Jp", "J2 - Jz**2 - Jz"),
        ("J+ = Jx + iJy", "Jp", "Jx + I*Jy"),
    ],
    "oscillator": [
        ("[a, a†] = 1", "comm(a, adag)", "Id"),
        ("[N, a] = -a", "comm(N, a)", "-a"),
        ("[N, a†] = a†", "comm(N, adag)", "adag"),
        ("a† a = N", "adag*a", "N"),
        ("[X, P] = i", "comm(X, P)", "I*Id"),
        ("X² + P² = 2N + 1", "X**2 + P**2", "2*N + 1"),
        ("[N, X²] = [N, P²]·(-1)", "comm(N, X**2)", "-comm(N, P**2)"),
    ],
}


def check_standard_identities(family: str, env: dict[str, sp.Matrix]) -> list[tuple[str, IdentityResult]]:
    """Run the family's textbook identities on the operators in `env`."""
    fam = FAMILIES[family]
    return [(label, check_identity(l, r, env, truncated=fam.truncated)) for label, l, r in STANDARD_IDENTITIES[family]]


# ------------------------------------------------------------------------------------------ analysing one operator

@dataclass
class OperatorReport:
    matrix: sp.Matrix
    hermitian: bool
    unitary: bool
    normal: bool
    trace: sp.Expr
    determinant: sp.Expr
    eigenvalues: list[tuple[sp.Expr, int]]            # (value, multiplicity)
    eigenvectors: list[tuple[sp.Expr, list[sp.Matrix]]] | None
    exact: bool
    checks: CheckList
    notes: list[str] = field(default_factory=list)


def analyze_operator(a: sp.Matrix) -> OperatorReport:
    """Properties of one operator, with eigen-decomposition verified by substitution."""
    n = a.shape[0]
    if a.shape != (n, n) or n > MAX_DIM:
        raise ValueError(f"Operators must be square and at most {MAX_DIM} x {MAX_DIM}.")
    adj = a.H
    hermitian = _simplified(a - adj).is_zero_matrix
    unitary = _simplified(a * adj - sp.eye(n)).is_zero_matrix
    normal = _simplified(a * adj - adj * a).is_zero_matrix
    trace, det = sp.simplify(a.trace()), sp.simplify(a.det())

    checks = CheckList()
    notes: list[str] = []
    eigenvectors = None
    exact = True
    try:
        eigenvectors = run_with_timeout(lambda: a.eigenvects(), label="eigenvectors", timeout=6.0)
        eigenvalues = [(sp.simplify(val), int(mult)) for val, mult, _ in eigenvectors]
        eigenvectors = [(sp.simplify(val), vecs) for val, _, vecs in eigenvectors]
    except ComputationTimeoutError:
        exact = False
        numeric = np.linalg.eigvals(np.array(a.evalf(), dtype=complex))
        eigenvalues = [(sp.Float(complex(v).real) if abs(complex(v).imag) < 1e-12 else sp.Float(complex(v).real) + sp.I * sp.Float(complex(v).imag), 1)
                       for v in numeric]
        notes.append("The exact eigen-decomposition timed out; eigenvalues are numerical and eigenvectors are omitted.")

    if exact and eigenvectors is not None:
        worst = 0.0
        for val, vecs in eigenvectors:
            for v in vecs:
                residual = (a * v - val * v).applyfunc(sp.simplify)
                worst = max(worst, float(max(abs(complex(sp.N(r))) for r in residual)))
        checks.add("A v = λ v for every eigenvector", worst < 1e-9, f"largest residual = {worst:.1e}")
    total = sum(m for _, m in eigenvalues)
    checks.add("Eigenvalue multiplicities add up to the dimension", total == n, f"{total} of {n}")
    ev_sum = sp.simplify(sum(v * m for v, m in eigenvalues))
    checks.add("Σ eigenvalues = trace", abs(complex(sp.N(ev_sum - trace))) < 1e-9, f"Σλ = {sp.N(ev_sum, 6)}, tr A = {sp.N(trace, 6)}")
    ev_prod = sp.simplify(sp.Mul(*[v ** m for v, m in eigenvalues]))
    checks.add("Π eigenvalues = determinant", abs(complex(sp.N(ev_prod - det))) < 1e-8 * max(1.0, abs(complex(sp.N(det)))),
               f"Πλ = {sp.N(ev_prod, 6)}, det A = {sp.N(det, 6)}")
    if hermitian:
        real = all(abs(complex(sp.N(v)).imag) < 1e-10 for v, _ in eigenvalues)
        checks.add("Hermitian ⇒ real eigenvalues", real, "all eigenvalues are real" if real else "complex eigenvalue found")
        if exact and eigenvectors is not None:
            vecs_all = [v for _, vs in eigenvectors for v in vs]
            if len(vecs_all) == n:
                gram = sp.Matrix(n, n, lambda i, j: (vecs_all[i].H * vecs_all[j])[0, 0])
                # eigenvectors of distinct eigenvalues of a Hermitian matrix are orthogonal
                orth_ok = True
                idx = 0
                groups = [len(vs) for _, vs in eigenvectors]
                starts = np.cumsum([0] + groups)
                for gi in range(len(groups)):
                    for gj in range(len(groups)):
                        if gi == gj:
                            continue
                        block = gram[starts[gi]:starts[gi + 1], starts[gj]:starts[gj + 1]]
                        if any(abs(complex(sp.N(v))) > 1e-9 for v in block):
                            orth_ok = False
                del idx
                checks.add("Eigenvectors of different eigenvalues are orthogonal", orth_ok, "Hermitian operator")
    if unitary:
        on_circle = all(abs(abs(complex(sp.N(v))) - 1) < 1e-10 for v, _ in eigenvalues)
        checks.add("Unitary ⇒ eigenvalues on the unit circle", on_circle, "all |λ| = 1" if on_circle else "some |λ| ≠ 1")
    return OperatorReport(a, bool(hermitian), bool(unitary), bool(normal), trace, det, eigenvalues, eigenvectors,
                          exact, checks, notes)


# ------------------------------------------------------------------------------------------ uncertainty relations

@dataclass
class UncertaintyResult:
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    product: float
    heisenberg_bound: float
    schrodinger_bound: float
    commutator_mean: complex
    covariance: float
    checks: CheckList


def parse_state(entries: Sequence[str], dim: int) -> sp.Matrix:
    """A normalised ket from component expressions."""
    if len(entries) != dim:
        raise ExpressionError(f"The state needs {dim} components.")
    vec = sp.Matrix([safe_expression(str(c), []) for c in entries])
    norm = sp.sqrt(sp.simplify((vec.H * vec)[0, 0]))
    if norm == 0:
        raise ExpressionError("The state is the zero vector.")
    return (vec / norm).applyfunc(sp.simplify)


def uncertainty_relations(a: sp.Matrix, b: sp.Matrix, state: sp.Matrix) -> UncertaintyResult:
    """ΔA ΔB against the Heisenberg-Robertson and Schrödinger-Robertson bounds for `state`."""
    n = a.shape[0]
    if a.shape != b.shape or state.shape != (n, 1):
        raise ValueError("The operators and the state must have matching dimensions.")
    for name, op in (("A", a), ("B", b)):
        if not _simplified(op - op.H).is_zero_matrix:
            raise ValueError(f"Operator {name} is not Hermitian, so it is not an observable.")
    an, bn, psi = (np.array(m.evalf(), dtype=complex) for m in (a, b, state))
    psi = psi[:, 0]
    psi = psi / np.linalg.norm(psi)

    def mean(op: np.ndarray) -> complex:
        return complex(np.vdot(psi, op @ psi))

    ma, mb = mean(an).real, mean(bn).real
    var_a = mean(an @ an).real - ma ** 2
    var_b = mean(bn @ bn).real - mb ** 2
    std_a, std_b = math.sqrt(max(var_a, 0.0)), math.sqrt(max(var_b, 0.0))
    comm_mean = mean(an @ bn - bn @ an)
    anti_mean = mean(an @ bn + bn @ an).real
    covariance = 0.5 * anti_mean - ma * mb
    heisenberg = 0.5 * abs(comm_mean)
    schrodinger = math.sqrt(covariance ** 2 + (0.5 * abs(comm_mean)) ** 2)
    product = std_a * std_b

    checks = CheckList()
    eps = 1e-9
    checks.add("Heisenberg–Robertson: ΔA ΔB ≥ ½|⟨[A, B]⟩|", product >= heisenberg - eps,
               f"ΔA ΔB = {product:.6g}, bound = {heisenberg:.6g}")
    checks.add("Schrödinger–Robertson (adds the covariance term)", product >= schrodinger - eps,
               f"ΔA ΔB = {product:.6g}, bound = {schrodinger:.6g}")
    checks.add("The stronger bound is at least as strong", schrodinger >= heisenberg - eps,
               f"{schrodinger:.6g} ≥ {heisenberg:.6g}")
    return UncertaintyResult(ma, mb, std_a, std_b, product, heisenberg, schrodinger, comm_mean, covariance, checks)


# ------------------------------------------------------------------------------------------ Baker–Campbell–Hausdorff

def _nested(*word: str) -> str:
    """[w1, [w2, [... , wn]]] as LaTeX, right-nested."""
    out = word[-1]
    for w in reversed(word[:-1]):
        out = rf"\left[{w},\,{out}\right]"
    return out


def bch_latex_terms(order: int) -> list[tuple[int, str]]:
    """The terms of log(e^X e^Y) up to `order`, as (order, LaTeX), in nested commutators."""
    if not 1 <= order <= MAX_BCH_ORDER:
        raise ValueError(f"The expansion is available to order {MAX_BCH_ORDER}.")
    x, y = "X", "Y"
    terms = [(1, f"{x} + {y}"),
             (2, rf"\tfrac{{1}}{{2}}{_nested(x, y)}"),
             (3, rf"\tfrac{{1}}{{12}}\left({_nested(x, x, y)} + {_nested(y, y, x)}\right)"),
             (4, rf"-\tfrac{{1}}{{24}}{_nested(y, x, x, y)}"),
             (5, rf"-\tfrac{{1}}{{720}}\left({_nested(y, y, y, y, x)} + {_nested(x, x, x, x, y)}\right)"
                 rf"+\tfrac{{1}}{{360}}\left({_nested(x, y, y, y, x)} + {_nested(y, x, x, x, y)}\right)"
                 rf"+\tfrac{{1}}{{120}}\left({_nested(y, x, y, x, y)} + {_nested(x, y, x, y, x)}\right)")]
    return [t for t in terms if t[0] <= order]


def _c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def bch_series(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """log(e^X e^Y) truncated after the terms of the given order (numerical matrices)."""
    if not 1 <= order <= MAX_BCH_ORDER:
        raise ValueError(f"The expansion is available to order {MAX_BCH_ORDER}.")
    z = x + y
    if order >= 2:
        z = z + 0.5 * _c(x, y)
    if order >= 3:
        z = z + (1 / 12) * (_c(x, _c(x, y)) + _c(y, _c(y, x)))
    if order >= 4:
        z = z - (1 / 24) * _c(y, _c(x, _c(x, y)))
    if order >= 5:
        z = (z - (1 / 720) * (_c(y, _c(y, _c(y, _c(y, x)))) + _c(x, _c(x, _c(x, _c(x, y)))))
             + (1 / 360) * (_c(x, _c(y, _c(y, _c(y, x)))) + _c(y, _c(x, _c(x, _c(x, y)))))
             + (1 / 120) * (_c(y, _c(x, _c(y, _c(x, y)))) + _c(x, _c(y, _c(x, _c(y, x))))))
    return z


@dataclass
class BCHResult:
    order: int
    latex_terms: list[tuple[int, str]]
    scales: np.ndarray
    errors: np.ndarray                    # ||log(e^eX e^eY) - series_order(eX, eY)|| for each scale
    fitted_exponent: float
    expected_exponent: int
    checks: CheckList


def verify_bch(x: sp.Matrix, y: sp.Matrix, order: int, scales: Sequence[float] | None = None) -> BCHResult:
    """The BCH series through `order`, verified numerically: the truncation error must shrink as ε^(order+1)
    when X and Y are scaled by ε, so the fitted slope on a log-log plot must be about order + 1."""
    if x.shape != y.shape:
        raise ValueError("X and Y must be operators of the same size.")
    xn, yn = (np.array(m.evalf(), dtype=complex) for m in (x, y))
    scales_arr = np.array(scales if scales is not None else [0.04, 0.06, 0.09, 0.13, 0.19], dtype=float)
    errors = []
    for eps in scales_arr:
        exact = logm(expm(eps * xn) @ expm(eps * yn))
        errors.append(float(np.linalg.norm(exact - bch_series(eps * xn, eps * yn, order), 2)))
    errors_arr = np.array(errors)

    checks = CheckList()
    commuting = np.allclose(_c(xn, yn), 0, atol=1e-12)
    if commuting:
        checks.add("X and Y commute, so the series is exact", float(np.max(errors_arr)) < 1e-10,
                   f"largest error = {np.max(errors_arr):.1e}")
        fitted = float("nan")
    else:
        positive = errors_arr > 1e-14
        if positive.sum() >= 3:
            fitted = float(np.polyfit(np.log(scales_arr[positive]), np.log(errors_arr[positive]), 1)[0])
            checks.add(f"Truncation error scales as ε^{order + 1}", abs(fitted - (order + 1)) < 0.45,
                       f"fitted exponent {fitted:.2f}, expected {order + 1}")
        else:
            fitted = float("nan")
            checks.add(f"Truncation error scales as ε^{order + 1}", True,
                       "the error is already at rounding level (the series closes on a finite algebra)")
    return BCHResult(order, bch_latex_terms(order), scales_arr, errors_arr, fitted, order + 1, checks)
