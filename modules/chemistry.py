"""
Chemistry: balancing, stoichiometry, acid-base and titration, kinetics -- Streamlit-free, no LLM.

Every result here is checked by something other than the method that produced it:
  * BALANCING finds coefficients as the exact (rational) null space of the element-and-charge matrix, then AUDITS them by
    recounting every atom and the charge on both sides from the formulas. The audit, not the null space, is what is reported.
  * STOICHIOMETRY ends with a mass-conservation check: the mass of everything consumed equals the mass of everything produced.
  * pH and TITRATION CURVES solve the exact charge balance numerically at every point (no "x is small" approximations, no
    Henderson-Hasselbalch assumption), then report the charge-balance residual, the stoichiometric equivalence volume, and where
    the textbook shortcuts (pH at half-equivalence = pKa; pH 7 at the strong/strong equivalence point) hold and where they do not.
  * KINETICS fits zero, first and second order by nonlinear least squares and compares them by BIC; when the data cannot tell two
    orders apart it SAYS SO instead of naming a winner, and checks the residuals for systematic structure.

CONVENTIONS. Species are separated by " + " (spaces around the plus). Charges are written Fe^3+, SO4^2-, or Na+ / OH- for +/-1;
the electron is e-. State symbols (s) (l) (g) (aq) are accepted and ignored. Hydrates use a dot: CuSO4.5H2O. 25 C throughout
(Kw = 1e-14). Atomic masses are the IUPAC conventional values rounded to 4-5 significant figures (enough for coursework, not for
metrology).
"""
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd

import numpy as np
import sympy as sp
from scipy.optimize import brentq, least_squares

KW = 1e-14
R_GAS = 8.314462618            # J / (mol K)


class ChemistryError(ValueError):
    """Bad input, with a reason a student can act on."""


# ------------------------------------------------------------------------------------------ atomic masses

ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Nd": 144.24, "Gd": 157.25, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22,
    "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04, "U": 238.03,
}


# ------------------------------------------------------------------------------------------ species and reactions

@dataclass(frozen=True)
class Species:
    text: str                                   # the formula as cleaned (no coefficient, no state symbol)
    atoms: tuple[tuple[str, int], ...]          # sorted (element, count)
    charge: int = 0
    state: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.atoms)

    @property
    def is_electron(self) -> bool:
        return not self.atoms and self.charge == -1

    def label(self) -> str:
        return self.text + (f" ({self.state})" if self.state else "")


_ELEMENT = re.compile(r"[A-Z][a-z]?")
_ARROWS = ("<=>", "<->", "⇌", "⟶", "->", "→", "=>", "=")
_STATE = re.compile(r"\((s|l|g|aq)\)\s*$")
_CHARGE = re.compile(r"(?:\^\{?(\d*)([+-])\}?|(?<=[A-Za-z0-9)\]])([+-]))$")


def parse_formula(text: str) -> dict[str, int]:
    """Atom counts of a formula: nested () and [], coefficients inside, and hydrates (CuSO4.5H2O)."""
    s = text.strip().replace("·", ".").replace("•", ".")
    if not s:
        raise ChemistryError("An empty formula.")
    total: dict[str, int] = {}
    for part in re.split(r"[.*]", s):
        mult = 1
        m = re.match(r"^(\d+)(?=[A-Z(\[])", part)
        if m:
            mult, part = int(m.group(1)), part[m.end():]
        for el, n in _parse_group(part, text).items():
            total[el] = total.get(el, 0) + n * mult
    if not total:
        raise ChemistryError(f"No atoms found in '{text}'.")
    return total


def _parse_group(s: str, original: str) -> dict[str, int]:
    stack: list[dict[str, int]] = [{}]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([":
            stack.append({})
            i += 1
        elif ch in ")]":
            if len(stack) == 1:
                raise ChemistryError(f"Unbalanced parentheses in '{original}'.")
            group = stack.pop()
            i += 1
            m = re.match(r"\d+", s[i:])
            n = int(m.group()) if m else 1
            i += len(m.group()) if m else 0
            for el, c in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + c * n
        else:
            m = _ELEMENT.match(s[i:])
            if not m:
                raise ChemistryError(f"Cannot read '{s[i:]}' in '{original}' (element symbols start with a capital letter).")
            el = m.group()
            if el not in ATOMIC_MASS:
                raise ChemistryError(f"Unknown element '{el}' in '{original}'.")
            i += len(el)
            m2 = re.match(r"\d+", s[i:])
            n = int(m2.group()) if m2 else 1
            i += len(m2.group()) if m2 else 0
            stack[-1][el] = stack[-1].get(el, 0) + n
    if len(stack) != 1:
        raise ChemistryError(f"Unbalanced parentheses in '{original}'.")
    return stack[0]


def parse_species(text: str) -> Species:
    """One species: 'H2O', 'Ca(OH)2 (aq)', 'Fe^3+', 'SO4^2-', 'OH-', 'e-'. A leading coefficient is ignored."""
    s = text.strip()
    if not s:
        raise ChemistryError("An empty species (two '+' in a row, or a side with nothing on it).")
    s = re.sub(r"^\d+\s*(?=[A-Za-z(\[])", "", s)                      # a coefficient the person already wrote
    state = ""
    m = _STATE.search(s)
    if m:
        state, s = m.group(1), s[:m.start()].rstrip()
    if s in ("e-", "e^-", "e^{-}"):
        return Species("e-", (), -1, state)
    charge = 0
    m = _CHARGE.search(s)
    if m:
        if m.group(2):
            charge = int(m.group(1) or 1) * (1 if m.group(2) == "+" else -1)
        else:
            charge = 1 if m.group(3) == "+" else -1
        s = s[:m.start()]
    atoms = parse_formula(s)
    return Species(s, tuple(sorted(atoms.items())), charge, state)


def parse_reaction(text: str) -> tuple[list[Species], list[Species]]:
    """(reactants, products) from 'C3H8 + O2 -> CO2 + H2O'. Species are separated by ' + ' (spaces around the plus)."""
    found = [(text.find(a), a) for a in _ARROWS if a in text]
    if not found:
        raise ChemistryError("No arrow found: write the reaction as 'reactants -> products'.")
    pos, arrow = min(found, key=lambda t: (t[0], -len(t[1])))
    left, right = text[:pos], text[pos + len(arrow):]
    if any(a in right for a in _ARROWS if a not in ("=",)) and arrow != "=":
        raise ChemistryError("More than one arrow: enter one reaction at a time.")

    def side(s: str) -> list[Species]:
        parts = re.split(r"\s+\+\s+", s.strip())
        if len(parts) == 1 and "^" not in s and "-" not in s and re.search(r"[A-Za-z0-9)\]]\+[A-Z(\[0-9]", s):
            parts = re.split(r"(?<=[A-Za-z0-9)\]])\+(?=[A-Z(\[0-9])", s.strip())
        return [parse_species(p) for p in parts]

    reactants, products = side(left), side(right)
    if not reactants or not products:
        raise ChemistryError("Both sides of the arrow need at least one species.")
    return reactants, products


def format_species(sp_: Species) -> str:
    c = sp_.charge
    ch = "" if c == 0 else ("^" + ("" if abs(c) == 1 else str(abs(c))) + ("+" if c > 0 else "-"))
    return sp_.text + ch


def species_latex(sp_: Species) -> str:
    """LaTeX with subscripted counts and a superscript charge: Ca(OH)2 -> \\mathrm{Ca(OH)_{2}}."""
    body = re.sub(r"(?<=[A-Za-z)\]])(\d+)", r"_{\1}", sp_.text).replace(".", r"\cdot ")
    c = sp_.charge
    ch = "" if c == 0 else "^{" + ("" if abs(c) == 1 else str(abs(c))) + ("+" if c > 0 else "-") + "}"
    return r"\mathrm{" + body + ch + "}"


def molar_mass(sp_: Species) -> float:
    """g/mol (electron mass neglected)."""
    return float(sum(ATOMIC_MASS[el] * n for el, n in sp_.atoms))


def percent_composition(sp_: Species) -> dict[str, float]:
    mm = molar_mass(sp_)
    if mm == 0:
        raise ChemistryError("An electron has no composition.")
    return {el: 100.0 * ATOMIC_MASS[el] * n / mm for el, n in sp_.atoms}


# ------------------------------------------------------------------------------------------ balancing

@dataclass
class LedgerRow:
    item: str                    # an element symbol, or "charge"
    left: int
    right: int

    @property
    def ok(self) -> bool:
        return self.left == self.right


@dataclass
class BalanceResult:
    reactants: list[Species]
    products: list[Species]
    status: str                              # balanced | underdetermined | unbalanceable
    message: str
    coefficients: list[int] | None = None    # reactants then products
    basis: list[list[int]] = field(default_factory=list)
    ledger: list[LedgerRow] = field(default_factory=list)
    verified: bool = False

    def _terms(self, species: list[Species], coeffs: list[int]) -> str:
        return " + ".join((f"{c} " if c != 1 else "") + format_species(s) for s, c in zip(species, coeffs))

    def equation(self) -> str:
        if self.coefficients is None:
            return ""
        n = len(self.reactants)
        return f"{self._terms(self.reactants, self.coefficients[:n])} → {self._terms(self.products, self.coefficients[n:])}"

    def latex(self) -> str:
        if self.coefficients is None:
            return ""
        n = len(self.reactants)

        def terms(species, coeffs):
            return " + ".join((f"{c}\\," if c != 1 else "") + species_latex(s) for s, c in zip(species, coeffs))

        return terms(self.reactants, self.coefficients[:n]) + r" \longrightarrow " + terms(self.products, self.coefficients[n:])


def _ledger(reactants: list[Species], products: list[Species], coeffs: Sequence[int]) -> list[LedgerRow]:
    """Recount every atom and the charge from the formulas and coefficients: independent of how the coefficients were found."""
    n = len(reactants)
    elements = sorted({el for s in reactants + products for el, _ in s.atoms})
    rows = []
    for el in elements:
        left = sum(c * s.counts.get(el, 0) for s, c in zip(reactants, coeffs[:n]))
        right = sum(c * s.counts.get(el, 0) for s, c in zip(products, coeffs[n:]))
        rows.append(LedgerRow(el, left, right))
    if any(s.charge for s in reactants + products):
        rows.append(LedgerRow("charge", sum(c * s.charge for s, c in zip(reactants, coeffs[:n])),
                              sum(c * s.charge for s, c in zip(products, coeffs[n:]))))
    return rows


def _to_integers(vec: Sequence[sp.Rational]) -> list[int]:
    fracs = [Fraction(int(v.p), int(v.q)) for v in vec]
    lcm = reduce(lambda a, b: a * b // gcd(a, b), [f.denominator for f in fracs], 1)
    ints = [int(f * lcm) for f in fracs]
    g = reduce(gcd, [abs(i) for i in ints if i], 0) or 1
    return [i // g for i in ints]


def balance(reactants: list[Species], products: list[Species]) -> BalanceResult:
    """Smallest whole-number coefficients that conserve every element and the charge, found as the exact null space of the
    conservation matrix and then verified by recounting."""
    species = reactants + products
    elements = sorted({el for s in species for el, _ in s.atoms})
    has_charge = any(s.charge for s in species)
    rows = [[(1 if i < len(reactants) else -1) * s.counts.get(el, 0) for i, s in enumerate(species)] for el in elements]
    if has_charge:
        rows.append([(1 if i < len(reactants) else -1) * s.charge for i, s in enumerate(species)])
    null = sp.Matrix(rows).nullspace()
    res = BalanceResult(reactants, products, "unbalanceable", "")
    if not null:
        res.message = ("No set of coefficients conserves every element" + (" and the charge" if has_charge else "") +
                       ". Check the formulas, and that each element appears on both sides.")
        return res
    if len(null) > 1:
        basis = [_to_integers(list(v)) for v in null]
        res.basis = basis
        res.status = "underdetermined"
        best = _smallest_positive(basis)
        res.message = (f"These species allow {len(null)} independent balances, so the answer is not unique"
                       + (" (the smallest all-positive one is shown)." if best else ", and none of them has all-positive coefficients."))
        if best is None:
            return res
        coeffs = best
    else:
        coeffs = _to_integers(list(null[0]))
        zero = [format_species(s) for s, c in zip(species, coeffs) if c == 0]
        if zero:
            res.message = ("These species take no part in a balanced reaction (their elements appear nowhere else): " + ", ".join(zero) + ".")
            return res
        if all(c < 0 for c in coeffs):
            coeffs = [-c for c in coeffs]
        if not all(c > 0 for c in coeffs):
            wrong = [format_species(s) for s, c in zip(species, coeffs) if c < 0]
            res.message = ("No all-positive solution: the reaction cannot balance as written. One or more species may be on the wrong "
                           "side of the arrow (the ones whose coefficients came out negative: " + ", ".join(wrong) + ").")
            return res
        res.status = "balanced"
        res.message = "Balanced."
    res.coefficients = coeffs
    res.ledger = _ledger(reactants, products, coeffs)
    res.verified = all(r.ok for r in res.ledger)
    if not res.verified:                                                       # the audit disagrees with the algebra: never report it as balanced
        res.status, res.message = "unbalanceable", "Internal check failed: the coefficients do not recount correctly."
        res.coefficients = None
    return res


def _smallest_positive(basis: list[list[int]], span: int = 6) -> list[int] | None:
    """The all-positive integer combination of the basis vectors with the smallest sum, searched over small weights."""
    if len(basis) > 4:                                                           # the search grows as (2*span+1)^k: not worth it beyond this
        return None
    best: tuple[int, list[int]] | None = None
    for weights in product(range(-span, span + 1), repeat=len(basis)):
        if not any(weights):
            continue
        vec = [sum(w * b[i] for w, b in zip(weights, basis)) for i in range(len(basis[0]))]
        if all(v > 0 for v in vec):
            g = reduce(gcd, vec)
            vec = [v // g for v in vec]
            if best is None or sum(vec) < best[0]:
                best = (sum(vec), vec)
    return best[1] if best else None


def balance_text(text: str) -> BalanceResult:
    return balance(*parse_reaction(text))


# ------------------------------------------------------------------------------------------ stoichiometry

@dataclass
class StoichRow:
    label: str
    role: str                    # reactant | product
    coefficient: int
    molar_mass: float
    initial_mol: float | None
    change_mol: float
    final_mol: float | None
    final_g: float | None


@dataclass
class StoichResult:
    rows: list[StoichRow]
    limiting: list[str]
    extent: float                # mol of reaction
    mass_consumed_g: float
    mass_produced_g: float
    notes: list[str] = field(default_factory=list)

    @property
    def mass_conserved(self) -> bool:
        return abs(self.mass_consumed_g - self.mass_produced_g) <= 1e-9 * max(self.mass_consumed_g, 1e-300)


def stoichiometry(result: BalanceResult, amounts: dict[int, tuple[float, str]]) -> StoichResult:
    """Limiting reagent and yields. `amounts` maps a REACTANT's index to (value, 'mol' or 'g'); a reactant with no amount is taken to
    be in excess (never limiting)."""
    if result.status != "balanced" or result.coefficients is None:
        raise ChemistryError("Balance the reaction first.")
    n = len(result.reactants)
    species = result.reactants + result.products
    moles: dict[int, float] = {}
    for idx, (value, unit) in amounts.items():
        if not 0 <= idx < n:
            raise ChemistryError("Amounts can only be given for reactants.")
        if value < 0 or not math.isfinite(value):
            raise ChemistryError("Amounts must be non-negative numbers.")
        if unit not in ("mol", "g"):
            raise ChemistryError("The unit must be 'mol' or 'g'.")
        mm = molar_mass(species[idx])
        if unit == "g" and mm == 0:
            raise ChemistryError("Cannot convert grams of an electron.")
        moles[idx] = value if unit == "mol" else value / mm
    if not moles:
        raise ChemistryError("Give the amount of at least one reactant.")
    ratios = {i: moles[i] / result.coefficients[i] for i in moles}
    extent = min(ratios.values())
    limiting = [format_species(species[i]) for i, r in ratios.items() if math.isclose(r, extent, rel_tol=1e-9, abs_tol=1e-15)]
    rows, consumed, produced = [], 0.0, 0.0
    for i, s in enumerate(species):
        nu, mm = result.coefficients[i], molar_mass(s)
        if i < n:
            change = -nu * extent
            initial = moles.get(i)
            final = None if initial is None else initial + change
            consumed += nu * extent * mm
            rows.append(StoichRow(format_species(s), "reactant", nu, mm, initial, change, final, None if final is None else final * mm))
        else:
            change = nu * extent
            produced += nu * extent * mm
            rows.append(StoichRow(format_species(s), "product", nu, mm, 0.0, change, change, change * mm))
    notes = []
    if len(limiting) > 1:
        notes.append("Several reactants run out together (stoichiometric proportions): " + ", ".join(limiting) + ".")
    missing = [format_species(species[i]) for i in range(n) if i not in moles]
    if missing:
        notes.append("Assumed in excess (no amount given): " + ", ".join(missing) + ".")
    return StoichResult(rows, limiting, extent, consumed, produced, notes)


def percent_yield(actual: float, theoretical: float) -> float:
    if theoretical <= 0:
        raise ChemistryError("The theoretical yield must be positive.")
    return 100.0 * actual / theoretical


# ------------------------------------------------------------------------------------------ acid-base equilibria

def _ka(pkas: Sequence[float]) -> list[float]:
    return [10.0 ** (-p) for p in pkas]


def alpha_fractions(h: float | np.ndarray, pkas: Sequence[float]) -> list[np.ndarray]:
    """Fractions of an n-protic acid H_nA present as H_n-iA^i- (i = 0..n) at hydrogen-ion concentration h (mol/L)."""
    ka = _ka(pkas)
    n = len(ka)
    h = np.asarray(h, dtype=float)
    terms = [h ** n]
    prod = 1.0
    for i, k in enumerate(ka, start=1):
        prod = prod * k
        terms.append(prod * h ** (n - i))
    den = sum(terms)
    return [t / den for t in terms]


def _mean_charge(h: float, pkas: Sequence[float]) -> float:
    """Average number of protons lost per acid molecule at h: sum i*alpha_i."""
    return float(sum(i * a for i, a in enumerate(alpha_fractions(h, pkas))))


def _solve_log_h(f, lo: float = -2.0, hi: float = 16.0) -> float:
    """The hydrogen-ion concentration h (mol/L) at which f(h) = 0, searched in pH (so it works across 18 orders of magnitude)."""
    ph = brentq(lambda p: f(10.0 ** (-p)), lo, hi, xtol=1e-13, rtol=1e-13, maxiter=500)
    return 10.0 ** (-ph)


@dataclass
class PhResult:
    ph: float
    h: float
    approximation: float | None            # the textbook estimate, if there is one
    residual: float                        # relative charge-balance residual at the solution
    note: str = ""

    @property
    def approximation_error(self) -> float | None:
        return None if self.approximation is None else self.approximation - self.ph


def ph_strong(c: float, kind: str = "acid") -> PhResult:
    """Exact pH of a strong monoprotic acid or base of concentration c (water's own ions included, so 1e-8 M HCl is 6.98, not 8)."""
    if c <= 0:
        raise ChemistryError("The concentration must be positive.")
    if kind not in ("acid", "base"):
        raise ChemistryError("kind must be 'acid' or 'base'.")
    h = (c + math.sqrt(c * c + 4 * KW)) / 2 if kind == "acid" else 2 * KW / (c + math.sqrt(c * c + 4 * KW))
    approx = -math.log10(c) if kind == "acid" else 14.0 + math.log10(c)
    return PhResult(-math.log10(h), h, approx, 0.0)


def ph_weak_acid(c: float, pkas: Sequence[float]) -> PhResult:
    """Exact pH of a weak n-protic acid (n = len(pkas)) at total concentration c: the charge balance
    [H+] = [OH-] + sum i [A^i-] solved numerically."""
    if c <= 0:
        raise ChemistryError("The concentration must be positive.")
    if not pkas or len(pkas) > 4:
        raise ChemistryError("Give between 1 and 4 pKa values.")
    if list(pkas) != sorted(pkas):
        raise ChemistryError("pKa values must increase (pKa1 < pKa2 < ...).")

    def f(h: float) -> float:
        return h - KW / h - c * _mean_charge(h, pkas)

    h = _solve_log_h(f)
    scale = max(h, KW / h, c * _mean_charge(h, pkas), 1e-300)
    approx = -0.5 * math.log10(10.0 ** (-pkas[0]) * c) if len(pkas) >= 1 else None
    return PhResult(-math.log10(h), h, approx, abs(f(h)) / scale,
                    "The shortcut pH = ½(pKa − log c) assumes the acid is barely dissociated and ignores water and later steps.")


def ph_weak_base(c: float, pkb: float) -> PhResult:
    """Exact pH of a weak base B (B + H2O <-> BH+ + OH-) at total concentration c."""
    if c <= 0:
        raise ChemistryError("The concentration must be positive.")
    kb = 10.0 ** (-pkb)

    def g(oh: float) -> float:
        return c * kb / (kb + oh) + KW / oh - oh                                # [BH+] + [H+] - [OH-], with [BH+] = c Kb / (Kb + [OH-])

    poh = brentq(lambda p: g(10.0 ** (-p)), -2.0, 16.0, xtol=1e-13, rtol=1e-13, maxiter=500)
    oh = 10.0 ** (-poh)
    h = KW / oh
    scale = max(c * kb / (kb + oh), KW / oh, oh)
    approx = 14.0 + 0.5 * math.log10(kb * c)
    return PhResult(-math.log10(h), h, approx, abs(g(oh)) / scale,
                    "The shortcut pOH = ½(pKb − log c) assumes the base is barely protonated and ignores water.")


@dataclass
class BufferResult:
    exact_ph: float
    henderson_hasselbalch: float
    residual: float

    @property
    def difference(self) -> float:
        return self.henderson_hasselbalch - self.exact_ph


def buffer_ph(pka: float, c_acid: float, c_base: float) -> BufferResult:
    """A weak acid HA (c_acid) with its conjugate base A- (c_base): exact pH against the Henderson-Hasselbalch estimate."""
    if c_acid <= 0 or c_base <= 0:
        raise ChemistryError("Both concentrations must be positive for a buffer.")
    ct = c_acid + c_base

    def f(h: float) -> float:
        return c_base + h - KW / h - ct * float(alpha_fractions(h, [pka])[1])

    h = _solve_log_h(f)
    return BufferResult(-math.log10(h), pka + math.log10(c_base / c_acid), abs(f(h)) / max(c_base, h, KW / h))


# ------------------------------------------------------------------------------------------ titration

@dataclass
class TitrationCheck:
    label: str
    passed: bool
    detail: str


@dataclass
class TitrationCurve:
    volume_ml: np.ndarray
    ph: np.ndarray
    equivalence: list[tuple[float, float]]            # (volume, pH) of each stoichiometric equivalence point
    half_equivalence: list[tuple[float, float, float]]  # (volume, pH, expected pKa)
    checks: list[TitrationCheck]
    pkas: list[float]
    kind: str
    strong: bool

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def titration_curve(kind: str, pkas: Sequence[float], c_analyte: float, v_analyte_ml: float, c_titrant: float, v_max_ml: float | None = None,
                    strong: bool = False, n_points: int = 241) -> TitrationCurve:
    """pH against titrant volume for an acid titrated with strong base (kind='acid', pKa values for each proton; strong=True for HCl-like)
    or a weak/strong base titrated with strong acid (kind='base', pkas=[pKa of its conjugate acid]).

    At every volume the exact charge balance is solved, for the diluted concentrations, in pH from -2 to 16."""
    if kind not in ("acid", "base"):
        raise ChemistryError("kind must be 'acid' or 'base'.")
    if min(c_analyte, v_analyte_ml, c_titrant) <= 0:
        raise ChemistryError("Concentrations and the analyte volume must be positive.")
    if kind == "base" and len(pkas) != 1 and not strong:
        raise ChemistryError("A weak base needs exactly one value: the pKa of its conjugate acid.")
    pk = [-8.0] if (strong and kind == "acid") else (list(pkas) if not strong else [22.0])
    n_protons = 1 if strong or kind == "base" else len(pk)
    if list(pk) != sorted(pk):
        raise ChemistryError("pKa values must increase.")
    mol_analyte = c_analyte * v_analyte_ml / 1000.0
    v_eq = [k * mol_analyte / c_titrant * 1000.0 for k in range(1, n_protons + 1)]
    v_max = v_max_ml if v_max_ml is not None else v_eq[-1] * 1.5
    if v_max <= 0:
        raise ChemistryError("The maximum titrant volume must be positive.")
    half = [(v_eq[0] / 2.0) if k == 1 else (v_eq[k - 2] + v_eq[k - 1]) / 2.0 for k in range(1, n_protons + 1)]
    volumes = np.unique(np.concatenate([np.linspace(0.0, v_max, n_points), v_eq, half]))
    volumes = volumes[volumes <= v_max + 1e-12]
    ka_conj = 10.0 ** (-pk[0])

    def balance_fn(h: float, v: float) -> tuple[float, float]:
        """(charge-balance residual, scale) at hydrogen-ion concentration h and titrant volume v (mL)."""
        total_v = v_analyte_ml + v
        ct = c_analyte * v_analyte_ml / total_v
        added = c_titrant * v / total_v
        if kind == "acid":
            neg = ct * _mean_charge(h, pk)
            return added + h - KW / h - neg, max(added, h, KW / h, neg, 1e-300)
        frac = h / (h + ka_conj)                                             # protonated fraction of the base
        return h + ct * frac - KW / h - added, max(added, h, KW / h, ct * frac, 1e-300)

    ph = np.empty_like(volumes)
    worst = 0.0
    for i, v in enumerate(volumes):
        root = _solve_log_h(lambda h, v=v: balance_fn(h, v)[0])
        ph[i] = -math.log10(root)
        res, scale = balance_fn(root, v)
        worst = max(worst, abs(res) / scale)

    def ph_at(v: float) -> float:
        return float(np.interp(v, volumes, ph))

    checks = [TitrationCheck("Charge balance", worst < 1e-8, f"the largest relative residual over {len(volumes)} points is {worst:.1e}")]
    diffs = np.diff(ph)
    monotonic = bool(np.all(diffs >= -1e-9)) if kind == "acid" else bool(np.all(diffs <= 1e-9))
    checks.append(TitrationCheck("Physical trend", monotonic, "pH rises steadily with added base" if kind == "acid" else "pH falls steadily with added acid"))
    eq = [(v, ph_at(v)) for v in v_eq if v <= v_max]
    checks.append(TitrationCheck("Equivalence volume", True, "from the amounts: " + ", ".join(f"{v:.4g} mL" for v, _ in eq) +
                                 (f" (moles of {'acid' if kind == 'acid' else 'base'} × protons ÷ titrant concentration)")))
    for v, p in eq[:1]:                                                       # the steepest point should sit at the equivalence volume
        window = (volumes > 0.97 * v) & (volumes < 1.03 * v)
        if window.sum() >= 4:
            jump = abs(float(np.interp(1.03 * v, volumes, ph) - np.interp(0.97 * v, volumes, ph)))
            if jump >= 2.0:
                slope = np.abs(np.gradient(ph, volumes))
                v_steep = float(volumes[window][np.argmax(slope[window])])
                checks.append(TitrationCheck("Steepest point at equivalence", abs(v_steep - v) <= 0.01 * v,
                                             f"the steepest rise is at {v_steep:.4g} mL; the amounts say {v:.4g} mL"))
    if strong and kind == "acid" and eq:
        checks.append(TitrationCheck("Strong acid / strong base", abs(eq[-1][1] - 7.0) < 0.02, f"pH at equivalence is {eq[-1][1]:.3f} (7.000 expected at 25 °C)"))
    half_pts = []
    if not strong and kind == "acid":
        for (v, expected) in zip(half, pk):
            half_pts.append((v, ph_at(v), expected))
    if not strong and kind == "base":
        half_pts.append((v_eq[0] / 2.0, ph_at(v_eq[0] / 2.0), pk[0]))
    return TitrationCurve(volumes, ph, eq, half_pts, checks, list(pk), kind, strong)


# ------------------------------------------------------------------------------------------ kinetics

ORDER_NAMES = {0: "zero order", 1: "first order", 2: "second order"}


def rate_curve(order: int, c0: float, k: float, t: np.ndarray | float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if order == 0:
        return np.maximum(c0 - k * t, 0.0)
    if order == 1:
        return c0 * np.exp(-k * t)
    if order == 2:
        return c0 / (1.0 + c0 * k * t)
    raise ChemistryError("The order must be 0, 1 or 2.")


def half_life(order: int, c0: float, k: float) -> float:
    return {0: c0 / (2 * k), 1: math.log(2) / k, 2: 1.0 / (k * c0)}[order]


def k_units(order: int, conc: str = "M", time: str = "s") -> str:
    return {0: f"{conc}/{time}", 1: f"1/{time}", 2: f"1/({conc}·{time})"}[order]


def posterior_model(order: int) -> tuple[str, list[str], list[float]]:
    """(expression in x = time, parameter names, rough start) for handing a kinetics fit to the posterior sampler."""
    return {0: ("C0 - k*x", ["C0", "k"], [1.0, 0.01]), 1: ("C0*exp(-k*x)", ["C0", "k"], [1.0, 0.1]),
            2: ("C0/(1 + C0*k*x)", ["C0", "k"], [1.0, 0.1])}[order]


@dataclass
class OrderFit:
    order: int
    c0: float
    k: float
    c0_se: float | None
    k_se: float | None
    rss: float
    bic: float
    r2: float
    r2_linearised: float | None
    half_life: float | None
    lag1_z: float                            # residual structure along time (|z| above 3: the model leaves a pattern)


@dataclass
class KineticsReport:
    fits: list[OrderFit]                     # best first
    ambiguous: bool
    warnings: list[str]
    t: np.ndarray
    c: np.ndarray

    @property
    def best(self) -> OrderFit:
        return self.fits[0]


def fit_rate_law(times: Sequence[float] | np.ndarray, conc: Sequence[float] | np.ndarray, orders: Sequence[int] = (0, 1, 2)) -> KineticsReport:
    """Fit each integrated rate law to concentration-time data and compare them. The comparison is by BIC on the nonlinear fits (not by
    R^2 of the linearised plots, which are reported for reference: linearising distorts the noise and favours the wrong order)."""
    t, c = np.asarray(times, dtype=float), np.asarray(conc, dtype=float)
    if t.shape != c.shape or t.ndim != 1:
        raise ChemistryError("Time and concentration must be lists of the same length.")
    if len(t) < 5:
        raise ChemistryError("Need at least 5 points to compare reaction orders.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(c))):
        raise ChemistryError("The data contains missing or non-finite values.")
    if np.any(c < 0):
        raise ChemistryError("Concentrations cannot be negative.")
    if np.any(np.diff(t) <= 0):
        t_order = np.argsort(t)
        t, c = t[t_order], c[t_order]
        if np.any(np.diff(t) <= 0):
            raise ChemistryError("Times must be distinct.")
    n = len(t)
    fits: list[OrderFit] = []
    for order in orders:
        if order not in (0, 1, 2):
            raise ChemistryError("Orders must be 0, 1 or 2.")
        lin_y = {0: c, 1: np.log(np.where(c > 0, c, np.nan)), 2: 1.0 / np.where(c > 0, c, np.nan)}[order]
        ok = np.isfinite(lin_y)
        r2_lin = None
        c0_guess, k_guess = float(c[0]), 0.1 / max(float(t[-1] - t[0]), 1e-12)
        if ok.sum() >= 3:
            slope, intercept = np.polyfit(t[ok], lin_y[ok], 1)
            pred = slope * t[ok] + intercept
            ss_tot = float(np.sum((lin_y[ok] - lin_y[ok].mean()) ** 2))
            r2_lin = 1.0 - float(np.sum((lin_y[ok] - pred) ** 2)) / ss_tot if ss_tot > 0 else None
            if order == 0 and slope < 0:
                c0_guess, k_guess = float(intercept), float(-slope)
            elif order == 1 and slope < 0:
                c0_guess, k_guess = float(math.exp(intercept)), float(-slope)
            elif order == 2 and slope > 0 and intercept > 0:
                c0_guess, k_guess = float(1.0 / intercept), float(slope)            # 1/c = 1/C0 + k t
        def resid(p, order=order):
            return rate_curve(order, p[0], p[1], t) - c
        try:
            sol = least_squares(resid, [max(c0_guess, 1e-12), max(k_guess, 1e-12)], bounds=([1e-15, 1e-15], [np.inf, np.inf]), x_scale="jac")
        except Exception as exc:  # noqa: BLE001
            raise ChemistryError(f"The {ORDER_NAMES[order]} fit failed ({type(exc).__name__}).") from exc
        rss = float(np.sum(sol.fun ** 2))
        dof = max(n - 2, 1)
        try:
            cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / dof)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.array([np.nan, np.nan])
        ss_tot_c = float(np.sum((c - c.mean()) ** 2))
        res = sol.fun
        r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if np.std(res) > 0 and n >= 8 else 0.0
        fits.append(OrderFit(order, float(sol.x[0]), float(sol.x[1]), float(se[0]), float(se[1]), rss,
                             n * math.log(max(rss, 1e-300) / n) + 2 * math.log(n), 1.0 - rss / ss_tot_c if ss_tot_c > 0 else 1.0,
                             r2_lin, half_life(order, float(sol.x[0]), float(sol.x[1])), (r1 if math.isfinite(r1) else 0.0) * math.sqrt(n)))
    fits.sort(key=lambda f: f.bic)
    warnings = []
    ambiguous = len(fits) > 1 and fits[1].bic - fits[0].bic < 2.0
    if ambiguous:
        warnings.append(f"The data cannot tell {ORDER_NAMES[fits[0].order]} from {ORDER_NAMES[fits[1].order]} (BIC difference "
                        f"{fits[1].bic - fits[0].bic:.1f}, below 2): a longer run, to a larger extent of reaction, would separate them.")
    if abs(fits[0].lag1_z) > 3:
        warnings.append(f"Even the best order ({ORDER_NAMES[fits[0].order]}) leaves a pattern in the residuals (lag-1 z = {fits[0].lag1_z:+.1f}): "
                        "the mechanism may be more complex than a single simple order (a reversible or multi-step reaction).")
    if (t[-1] - t[0]) > 0 and fits[0].order in (1, 2) and fits[0].half_life and (t[-1] - t[0]) < 0.5 * fits[0].half_life:
        warnings.append("The data cover less than half of one half-life, so the order is weakly determined.")
    return KineticsReport(fits, ambiguous, warnings, t, c)


def demo_kinetics(order: int = 1, k: float = 0.12, c0: float = 1.0, noise: float = 0.01, n: int = 14, t_max: float | None = None,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    t_max = t_max if t_max is not None else 3.0 * half_life(order, c0, k)
    t = np.linspace(0.0, t_max, n)
    c = rate_curve(order, c0, k, t) + np.random.default_rng(seed).normal(0.0, noise * c0, n)
    return t, np.maximum(c, 1e-6)


@dataclass
class ArrheniusFit:
    ea_kj: float
    ea_se_kj: float
    ln_a: float
    a: float
    r2: float
    temperatures: np.ndarray
    ln_k: np.ndarray

    def predict_k(self, temperature_k: float) -> float:
        return float(math.exp(self.ln_a - self.ea_kj * 1000.0 / (R_GAS * temperature_k)))


def arrhenius_fit(temperatures_k: Sequence[float], ks: Sequence[float]) -> ArrheniusFit:
    """k = A exp(-Ea/RT): a straight line of ln k against 1/T with slope -Ea/R."""
    T, k = np.asarray(temperatures_k, dtype=float), np.asarray(ks, dtype=float)
    if T.shape != k.shape or len(T) < 3:
        raise ChemistryError("Need at least 3 (temperature, k) pairs.")
    if np.any(T <= 0) or np.any(k <= 0):
        raise ChemistryError("Temperatures (in kelvin) and rate constants must be positive.")
    x, y = 1.0 / T, np.log(k)
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return ArrheniusFit(float(-slope * R_GAS / 1000.0), float(math.sqrt(cov[0, 0]) * R_GAS / 1000.0), float(intercept), float(math.exp(intercept)),
                        r2, T, y)
