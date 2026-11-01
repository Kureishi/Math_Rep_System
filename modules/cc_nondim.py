"""
Dimensionless-form helper: non-dimensionalise an equation automatically -- Streamlit-free.

Extends modules/dimensional_analysis.py (which finds combinations of quantities with a given dimension) from "what could this formula be"
to "what is the scale-free form of THIS equation". Input: an equation in a plain notation, the units of every quantity, optionally numbers.

NOTATION. The dependent variable is typed plainly (x), the independent variables are single letters (t, x, ...), and derivatives use
subscripts: x_t, x_tt for dx/dt, d2x/dt2 and u_xx, u_xt for partial derivatives. Every other name is a PARAMETER. Example:
        m*x_tt + c*x_t + k*x = F*sin(w*t)          (units: x -> m, t -> s, m -> kg, c -> kg/s, k -> kg/s^2, F -> N, w -> 1/s)

METHOD.
 1. The equation is checked for DIMENSIONAL HOMOGENEITY (every additive term has the same dimension; the arguments of sin, exp, ... are
    dimensionless) -- an inhomogeneous equation is refused with the term dimensions listed.
 2. Each variable v is replaced by S_v * v_hat, with the scale S_v a product of parameters raised to RATIONAL powers whose dimension
    is that of v (found by linear algebra on the dimension vectors of dim_analysis._dimension_vector). The remaining freedom is used
    to make coefficients equal ("dominant balance"): every subset of the equation's terms (and of its function arguments) is tried as
    the set to balance, the linear system for the exponents is solved, and the choice that leaves the FEWEST independent dimensionless
    groups (then the simplest exponents) wins; the alternatives are listed.
 3. What is left are dimensionless GROUPS multiplying the hatted terms; the count is compared with the BUCKINGHAM-PI prediction
    (#parameters - rank of the dimension matrix) and the groups are named Pi1, Pi2, ...
CHECKS (`NondimResult.checks`): homogeneity of the original; every scale has the right dimension; every group is dimensionless
(dimension vector zero); the SYMBOLIC round trip (rescaling the dimensionless equation back reproduces the original exactly); the group
count does not exceed Buckingham's bound; and, for an ODE whose highest derivative can be isolated and with initial values supplied, a
NUMERICAL EQUIVALENCE: the original equation (with numbers) and its dimensionless form are both integrated and u(t)/U must equal u_hat(t/T).
Numbers are taken in the coherent units of the unit strings (SI base units), which is stated wherever it matters.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from modules.cc_common import CrossCuttingError, names_in, parse_with
from modules.dimensional_analysis import _dimension_vector
from modules.quantum_common import CheckList
from modules.units_checker import UnitParseError

_FUNCS = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sinh, sp.cosh, sp.tanh, sp.atan)


@dataclass
class Candidate:
    scales: dict[str, sp.Expr]
    balanced: list[str]
    n_groups: int
    cost: float


@dataclass
class NondimResult:
    equation: str
    dependent: str
    independent: list[str]
    parameters: list[str]
    scales: dict[str, sp.Expr]                 # variable -> scale (a monomial in the parameters)
    hat: dict[str, sp.Symbol]
    dimensionless: sp.Expr                     # LHS - RHS of the dimensionless equation (== 0), in hat variables and Pi groups
    groups: dict[str, sp.Expr]                 # Pi name -> expression in the parameters
    n_parameters: int
    rank: int
    pi_bound: int
    balanced: list[str]
    alternatives: list[Candidate]
    checks: CheckList = field(default_factory=CheckList)
    numerical_note: str = ""
    free_dependent_scale: bool = False
    normaliser: sp.Expr = sp.Integer(1)          # the coefficient the whole equation was divided by
    reference_scales: list[str] = field(default_factory=list)   # auxiliary reference scales the user must choose

    @property
    def dimensionless_equation(self) -> sp.Eq:
        return sp.Eq(self.dimensionless, 0)


def _dim(unit: str) -> sp.Matrix:
    try:
        return sp.Matrix(list(_dimension_vector(unit)))
    except UnitParseError as exc:
        raise CrossCuttingError(f"Unit '{unit}' is not recognised: {exc}") from None
    except Exception as exc:  # noqa: BLE001
        raise CrossCuttingError(f"Could not interpret the unit '{unit}': {exc}") from None


def _deriv_counts(name: str, dep: str, ind: list[str]) -> dict[str, int] | None:
    if not name.startswith(dep + "_"):
        return None
    letters = name[len(dep) + 1:]
    if not letters or any(ch not in ind for ch in letters):
        return None
    return {v: letters.count(v) for v in ind if v in letters}


def nondimensionalize(equation: str, dependent: str, independent: list[str], units: dict[str, str], param_values: dict[str, float] | None = None,
                      initial: dict[str, float] | None = None, dep_scale: str | None = None) -> NondimResult:
    """See the module docstring. `initial` maps the dependent variable and its derivative symbols (x, x_t, ...) to their values at 0."""
    if "=" not in equation or equation.count("=") != 1:
        raise CrossCuttingError("Write the equation with exactly one '=', e.g. 'm*x_tt + k*x = F*sin(w*t)'.")
    lhs_t, rhs_t = (s.strip() for s in equation.split("="))
    ind = [i.strip() for i in independent if i.strip()]
    dep = dependent.strip()
    if not dep.isidentifier() or not ind or len(set(ind)) != len(ind) or any(len(i) != 1 or not i.isalpha() for i in ind) or dep in ind:
        raise CrossCuttingError("Give one dependent variable and single-letter independent variables (distinct).")
    names = names_in(lhs_t) | names_in(rhs_t)
    deriv_names = sorted(n for n in names if _deriv_counts(n, dep, ind))
    params = sorted(names - set(ind) - {dep} - set(deriv_names))
    missing = [n for n in [dep] + ind + params if n not in units]
    if missing:
        raise CrossCuttingError("Give a unit for: " + ", ".join(missing) + " (use '1' for a dimensionless quantity).")
    dims: dict[str, sp.Matrix] = {n: _dim(units[n]) for n in [dep] + ind + params}
    # a variable whose dimension no product of the parameters has (the length of a domain with no length parameter, the amplitude of a
    # homogeneous linear equation) needs a REFERENCE SCALE: add one as an auxiliary parameter with that dimension, and say so
    reference_scales: list[str] = []
    for v in ind + [dep]:
        Dm0 = sp.Matrix([[dims[q][i] for q in params] for i in range(7)]) if params else sp.zeros(7, 0)
        try:
            ok_dim = bool(params) and bool(sp.linsolve((Dm0, dims[v])))
        except Exception:  # noqa: BLE001
            ok_dim = False
        if not ok_dim and dims[v] != sp.zeros(7, 1):
            aux = f"{v}_ref"
            while aux in params or aux in units:
                aux += "_"
            params.append(aux)
            dims[aux] = dims[v]
            reference_scales.append(aux)
    params = sorted(params)
    sym = {n: sp.Symbol(n, positive=True) for n in [dep] + ind + params + deriv_names}
    expr = sp.expand(parse_with(lhs_t, sym) - parse_with(rhs_t, sym))
    if expr == 0:
        raise CrossCuttingError("The equation is identically 0 = 0.")
    for n in deriv_names:
        cnt = _deriv_counts(n, dep, ind) or {}
        dims[n] = dims[dep] - sum((dims[v] * c for v, c in cnt.items()), sp.zeros(7, 1))
    zero = sp.zeros(7, 1)

    def dim_of(e: sp.Expr) -> sp.Matrix:
        if e.is_Number:
            return zero
        if e.is_Symbol:
            return dims[str(e)]
        if e.is_Pow:
            return dim_of(e.base) * e.exp if e.exp.is_number else _bad(e)
        if e.is_Mul:
            return sum((dim_of(a) for a in e.args), zero)
        if e.is_Add:
            ds = [dim_of(a) for a in e.args]
            if any(d != ds[0] for d in ds[1:]):
                _bad(e)
            return ds[0]
        if isinstance(e, _FUNCS):
            if dim_of(e.args[0]) != zero:
                raise CrossCuttingError(f"The argument of {e.func.__name__} in '{e}' is not dimensionless.")
            return zero
        return _bad(e)

    def _bad(e: sp.Expr) -> sp.Matrix:
        raise CrossCuttingError(f"Cannot assign a dimension to '{e}'.")

    terms = list(sp.Add.make_args(expr))
    tdims = [dim_of(t) for t in terms]
    ck = CheckList()
    homogeneous = all(d == tdims[0] for d in tdims[1:])
    if not homogeneous:
        detail = "; ".join(f"{t}: {list(d)}" for t, d in zip(terms, tdims))
        raise CrossCuttingError("The equation is not dimensionally homogeneous (base-dimension vectors L, M, T, I, Θ, N, J per term): " + detail)
    ck.add("original equation is dimensionally homogeneous", True, f"all {len(terms)} terms have dimension vector {list(tdims[0])}")

    # --- exponent bookkeeping: scale unknowns S_v = prod params^e
    scale_vars = ind + [dep]
    U = {v: sp.Symbol(f"S_{v}", positive=True) for v in scale_vars}
    hat = {n: sp.Symbol(f"{n}_hat", positive=True) for n in [dep] + ind + deriv_names}
    sub: dict[sp.Symbol, sp.Expr] = {sym[dep]: U[dep] * hat[dep]}
    for v in ind:
        sub[sym[v]] = U[v] * hat[v]
    for n in deriv_names:
        cnt = _deriv_counts(n, dep, ind) or {}
        sub[sym[n]] = U[dep] * hat[n] / sp.Mul(*[U[v] ** c for v, c in cnt.items()])
    scaled_terms = [t.xreplace(sub) for t in terms]

    hat_syms = set(hat.values())

    def split(t: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """(coefficient without hat symbols and functions, hat-part)."""
        coeff, part = sp.Integer(1), sp.Integer(1)
        for f in sp.Mul.make_args(t):
            if f.free_symbols & hat_syms:
                part *= f
            else:
                coeff *= f
        return coeff, part
    split_terms = [split(t) for t in scaled_terms]
    # function arguments: G * hat -> condition G == 1
    arg_conditions: list[tuple[str, sp.Expr]] = []
    for _, part in split_terms:
        for fapp in part.atoms(*_FUNCS):
            a = sp.expand(fapp.args[0])
            hs = [h for h in hat_syms if a.has(h)]
            if len(hs) == 1 and a.is_Mul or (len(hs) == 1 and a.as_independent(hs[0])[1] == hs[0]):
                g = a.as_independent(hs[0])[0]
                if g != 1 and (str(fapp.func.__name__), g) not in arg_conditions:
                    arg_conditions.append((f"argument of {fapp.func.__name__} = {a.xreplace({h: 1 for h in hat_syms})}", g))
    p_syms = [sym[p] for p in params]
    all_exps: list[sp.Symbol] = []
    scale_exp: dict[str, list[sp.Symbol]] = {}
    for v in scale_vars:
        scale_exp[v] = [sp.Symbol(f"e_{v}_{p}") for p in params]
        all_exps += scale_exp[v]

    def exps_of(e: sp.Expr) -> dict[sp.Symbol, sp.Expr]:
        return {b: ex for b, ex in sp.powsimp(e).as_powers_dict().items() if isinstance(b, sp.Symbol)}

    def condition_row(ratio: sp.Expr) -> list[sp.Expr]:
        """Linear equations (one per parameter) saying the monomial `ratio` becomes 1 once S_v = prod p^e."""
        pw = exps_of(ratio)
        rows = []
        for k, p in enumerate(p_syms):
            row = pw.get(p, 0) + sum(pw.get(U[v], 0) * scale_exp[v][k] for v in scale_vars)
            rows.append(row)
        return rows

    def dim_rows(v: str) -> list[sp.Expr]:
        return [sum(scale_exp[v][k] * dims[p][i] for k, p in enumerate(params)) - dims[v][i] for i in range(7)]

    K = [c for c, _ in split_terms]
    # --- enumerate balance sets
    n_t = len(terms)
    cands: list[Candidate] = []
    seen: set[tuple] = set()
    free_dep = False

    def build(subset: tuple[int, ...], argset: tuple[int, ...], dep_free: bool) -> Candidate | None:
        eqs: list[sp.Expr] = []
        base = K[subset[0]]
        for j in subset[1:]:
            eqs += condition_row(sp.simplify(K[j] / base))
        for a in argset:
            eqs += condition_row(arg_conditions[a][1])
        for v in ind + ([] if dep_free else [dep]):
            eqs += dim_rows(v)
        if dep_free:
            for e in scale_exp[dep]:
                eqs.append(e)
        sol = sp.linsolve(eqs, all_exps)
        if not sol:
            return None
        vals = next(iter(sol))
        free = set().union(*[v.free_symbols for v in vals]) & set(all_exps)
        zero_sub = {f: 0 for f in free}
        vals = [sp.nsimplify(v.subs(zero_sub)) for v in vals]
        exp_map = dict(zip(all_exps, vals))
        scales = {v: sp.Mul(*[p ** exp_map[scale_exp[v][k]] for k, p in enumerate(p_syms)]) for v in scale_vars}
        if dep_free:
            scales[dep] = sp.Symbol(f"{dep}_c", positive=True)
        # leftover groups: coefficients of the (normalised) terms
        sc_sub = {U[v]: scales[v] for v in scale_vars}
        coeffs = [sp.simplify((K[i] / K[subset[0]]).xreplace(sc_sub)) for i in range(n_t)]
        argc = [sp.simplify(g.xreplace(sc_sub)) for _, g in arg_conditions]
        groups_vecs = []
        for c in coeffs + argc:
            if c == 1:
                continue
            pw = exps_of(c)
            if any(b not in p_syms and b != scales.get(dep) for b in pw):
                continue
            groups_vecs.append([pw.get(p, 0) for p in p_syms])
        rank = int(sp.Matrix(groups_vecs).rank()) if groups_vecs else 0
        cost = float(sum(abs(float(e)) for e in vals))
        return Candidate({v: scales[v] for v in scale_vars}, [str(terms[i]) for i in subset] + [arg_conditions[a][0] for a in argset], rank, cost)

    for dep_free in (False,):
        for r in range(2, n_t + 1):
            for subset in itertools.combinations(range(n_t), r):
                for ar in range(0, len(arg_conditions) + 1):
                    for argset in itertools.combinations(range(len(arg_conditions)), ar):
                        key = (subset, argset, dep_free)
                        if key in seen:
                            continue
                        seen.add(key)
                        try:
                            c = build(subset, argset, dep_free)
                        except Exception:  # noqa: BLE001
                            c = None
                        if c is not None:
                            cands.append(c)
        if cands:
            free_dep = dep_free
            break
    if not cands:
        raise CrossCuttingError("No consistent set of scales could be built from the parameters: some variable has a dimension that no product of "
                                "the parameters has. Add a parameter with that dimension (a reference length, time, ...).")
    if dep_scale:
        # an explicit scale for the dependent variable overrides the automatic one
        extra = {n: sp.Symbol(n, positive=True) for n in names_in(dep_scale) if n in units and n not in params and n not in (dep, *ind)}
        for n in extra:                                          # a reference value named in the units table (e.g. the initial amplitude x0)
            dims[n] = _dim(units[n])
        ds = parse_with(dep_scale, {**{p: sym[p] for p in params}, **extra})
        if dim_of(ds) != dims[dep]:
            raise CrossCuttingError(f"The scale '{dep_scale}' does not have the dimensions of {dep}.")
        free_dep = False
        cands = [c for c in cands]
        for c in cands:
            c.scales[dep] = ds
    cands.sort(key=lambda c: (c.n_groups, c.cost, len(c.balanced)))
    best = cands[0]
    scales = best.scales
    sc_sub = {U[v]: scales[v] for v in scale_vars}

    # --- the dimensionless equation
    base_idx = next(i for i, t in enumerate(terms) if str(t) in best.balanced[0] or True)
    del base_idx
    norm = K[[i for i, t in enumerate(terms) if str(t) == best.balanced[0]][0]].xreplace(sc_sub)
    norm = sp.simplify(norm)
    new = sp.Integer(0)
    for (c, part), t in zip(split_terms, scaled_terms):
        new += sp.simplify(c.xreplace(sc_sub) / norm) * part
    # hat the function arguments too: S_t * t_hat inside sin(...) -> already substituted through scaled_terms
    new = new.xreplace({U[v]: scales[v] for v in scale_vars})
    new = sp.expand(sp.powsimp(new))

    # name the groups
    group_exprs: dict[str, sp.Expr] = {}
    renamed = new
    coeffs_in = []
    for f in sp.Add.make_args(new):
        c, part = f.as_independent(*hat_syms, *[a for a in f.atoms(*_FUNCS)], as_Add=False)
        coeffs_in.append(c)
    # replace each distinct non-numeric coefficient by a Pi symbol (identical monomials share a symbol)
    pi_of: dict[sp.Expr, sp.Symbol] = {}
    for c in coeffs_in:
        mono, num = c.as_independent(*p_syms, as_Add=False) if False else (c, sp.Integer(1))
        const, rest = c.as_coeff_Mul()
        if rest != 1 and rest not in pi_of:
            pi_of[rest] = sp.Symbol(f"Pi{len(pi_of) + 1}")
        del mono, num, const
    terms_new = []
    for f in sp.Add.make_args(new):
        c, part = f.as_independent(*hat_syms, *[a for a in f.atoms(*_FUNCS)], as_Add=False)
        const, rest = c.as_coeff_Mul()
        terms_new.append(const * (pi_of[rest] if rest != 1 else 1) * part)
    renamed = sp.Add(*terms_new)
    for rest, s in pi_of.items():
        group_exprs[str(s)] = rest
    # arguments of functions that still contain parameters (wT) get their own group
    for fapp in list(renamed.atoms(*_FUNCS)):
        a = fapp.args[0]
        c, h = a.as_independent(*hat_syms, as_Add=False)
        if c != 1 and not c.is_number:
            nm = f"Pi{len(group_exprs) + 1}"
            s = sp.Symbol(nm)
            group_exprs[nm] = c
            renamed = renamed.xreplace({fapp: fapp.func(s * h)})
    # merge duplicate group definitions
    inv: dict[sp.Expr, str] = {}
    for nm, e in list(group_exprs.items()):
        if e in inv:
            renamed = renamed.xreplace({sp.Symbol(nm): sp.Symbol(inv[e])})
            del group_exprs[nm]
        else:
            inv[e] = nm
    renamed = sp.expand(renamed)

    # --- Buckingham
    Dm = sp.Matrix([[dims[p][i] for p in params] for i in range(7)]) if params else sp.zeros(7, 0)
    rank = int(Dm.rank()) if params else 0
    pi_bound = len(params) - rank
    res = NondimResult(equation, dep, ind, params, scales, hat, renamed, group_exprs, len(params), rank, pi_bound, best.balanced, cands[1:6], ck, "", free_dep, norm, reference_scales)

    # --- checks
    for v in scale_vars:
        sdim = dim_of(scales[v]) if not (free_dep and v == dep) else dims[dep]
        ck.add(f"scale of {v} has the dimensions of {v}", sdim == dims[v], f"S_{v} = {scales[v]}")
    gd_ok = all(dim_of(e) == zero for e in group_exprs.values())
    ck.add("every dimensionless group is dimensionless (dimension vector 0)", gd_ok,
           ", ".join(f"{k} = {sp.simplify(v)}" for k, v in group_exprs.items()) or "no groups left: the equation has no free parameter")
    # round trip: rescale the dimensionless equation back
    back = {sp.Symbol(k): v for k, v in group_exprs.items()}
    inverse: dict[sp.Symbol, sp.Expr] = {hat[dep]: sym[dep] / scales[dep]}
    for v in ind:
        inverse[hat[v]] = sym[v] / scales[v]
    for n in deriv_names:
        cnt = _deriv_counts(n, dep, ind) or {}
        inverse[hat[n]] = sym[n] * sp.Mul(*[scales[v] ** c for v, c in cnt.items()]) / scales[dep]
    restored = renamed.xreplace(back).xreplace(inverse)
    residual = sp.simplify(sp.expand(restored * res.normaliser - expr))
    ck.add("symbolic round trip: rescaling the dimensionless equation back and multiplying by the balanced coefficient gives the original EXACTLY",
           residual == 0, f"original − rescaled = {residual}")
    n_indep = _count_groups(group_exprs, params)
    ck.add("number of independent groups ≤ Buckingham π bound (#parameters − rank of the dimension matrix)", n_indep <= pi_bound + 0,
           f"{n_indep} independent group(s); {len(params)} parameters, rank {rank} → bound {pi_bound}")
    res.numerical_note = _numerical_equivalence(res, sym, expr, deriv_names, dep, ind, param_values, initial)
    if res.numerical_note.startswith("PASS"):
        ck.add("numerical equivalence: original ODE with numbers and its dimensionless form give the same solution", True, res.numerical_note[6:])
    elif res.numerical_note.startswith("FAIL"):
        ck.add("numerical equivalence: original ODE with numbers and its dimensionless form give the same solution", False, res.numerical_note[6:])
    return res


def _count_groups(groups: dict[str, sp.Expr], params: list[str]) -> int:
    vecs = []
    for e in groups.values():
        pw = e.as_powers_dict()
        vecs.append([pw.get(sp.Symbol(p, positive=True), 0) for p in params])
    return int(sp.Matrix(vecs).rank()) if vecs else 0


def _numerical_equivalence(res: NondimResult, sym: dict[str, sp.Symbol], expr: sp.Expr, deriv_names: list[str], dep: str, ind: list[str],
                           param_values: dict[str, float] | None, initial: dict[str, float] | None) -> str:
    """For a single-independent-variable ODE with its highest derivative isolable: integrate both forms and compare. Returns '' if not applicable."""
    if len(ind) != 1 or param_values is None or initial is None:
        return ""
    t = ind[0]
    orders = {n: (_deriv_counts(n, dep, ind) or {}).get(t, 0) for n in deriv_names}
    if not orders:
        return ""
    top_name = max(orders, key=lambda n: orders[n])
    order = orders[top_name]
    chain = [dep] + [dep + "_" + t * k for k in range(1, order + 1)]
    if any(n not in chain for n in deriv_names) or any(n not in initial for n in chain[:-1]):
        return ""
    try:
        pv = {sym[p]: float(param_values[p]) for p in res.parameters}
    except KeyError:
        return ""
    try:
        # the symbols were declared positive (for the scale algebra); a state variable or derivative can be negative, and SymPy would then
        # find no positive solution for it, so solve and integrate with plain real counterparts
        real = lambda e: e.xreplace({x: sp.Symbol(x.name + "_r", real=True) for x in e.free_symbols if x.is_positive})  # noqa: E731
        sol_top = sp.solve(real(expr.xreplace(pv)), real(sym[top_name]))
        if len(sol_top) != 1:
            return ""
        f_orig = sp.lambdify([real(sym[t])] + [real(sym[n]) for n in chain[:-1]], sol_top[0], "numpy")
        hexpr = res.dimensionless.xreplace({sp.Symbol(k): float(sp.N(v.xreplace(pv))) for k, v in res.groups.items()})
        hs = [res.hat[n] for n in chain]
        sol_h = sp.solve(real(hexpr), real(hs[-1]))
        if len(sol_h) != 1:
            return ""
        f_hat = sp.lambdify([real(res.hat[t])] + [real(h) for h in hs[:-1]], sol_h[0], "numpy")
        S = {v: float(sp.N(res.scales[v].xreplace(pv))) for v in [dep, t]}
        y0 = [float(initial[n]) for n in chain[:-1]]
        T_end = 10.0 * S[t]
        n_state = order
        ts = np.linspace(0, T_end, 200)
        so = solve_ivp(lambda tt, y: list(y[1:]) + [float(f_orig(tt, *y))], (0, T_end), y0, t_eval=ts, rtol=1e-11, atol=1e-13, method="DOP853")
        # initial values of the hatted derivatives: x_hat^(k) = x^(k) * S_t^k / S_dep
        yh0 = [y0[k] * S[t] ** k / S[dep] for k in range(n_state)]
        sh = solve_ivp(lambda tt, y: list(y[1:]) + [float(f_hat(tt, *y))], (0, 10.0), yh0, t_eval=ts / S[t], rtol=1e-11, atol=1e-13, method="DOP853")
        if not (so.success and sh.success):
            return "FAIL: an integration failed"
        a, b = so.y[0] / S[dep], sh.y[0]
        err = float(np.max(np.abs(a - b)))
        scale = max(float(np.max(np.abs(b))), 1e-12)
        ok = err < 1e-6 * max(scale, 1.0)
        return ("PASS: " if ok else "FAIL: ") + f"max |{dep}(t)/S_{dep} − {dep}̂(t/S_{t})| = {err:.1e} over 10 time scales (S_{t} = {S[t]:.4g}, S_{dep} = {S[dep]:.4g}; SI units)"
    except Exception as exc:  # noqa: BLE001
        return f"SKIP: {exc}"


def groups_numeric(res: NondimResult, param_values: dict[str, float]) -> dict[str, float]:
    pv = {sp.Symbol(p, positive=True): float(param_values[p]) for p in res.parameters}
    return {k: float(sp.N(v.xreplace(pv))) for k, v in res.groups.items()}


def sweep_points(res: NondimResult, ranges: dict[str, tuple[float, float]], n: int = 5) -> list[dict[str, float]]:
    """Log-spaced grid over the dimensionless groups: since the dimensionless equation depends only on them, this grid covers every
    physically distinct regime, however many dimensional parameters the original problem had."""
    unknown = set(ranges) - set(res.groups)
    if unknown:
        raise CrossCuttingError("Unknown group(s): " + ", ".join(sorted(unknown)))
    axes = [np.geomspace(lo, hi, n) for lo, hi in (ranges[k] for k in res.groups if k in ranges)]
    keys = [k for k in res.groups if k in ranges]
    return [dict(zip(keys, map(float, combo))) for combo in itertools.product(*axes)]
