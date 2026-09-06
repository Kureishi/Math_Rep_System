from fractions import Fraction

from modules.dimensional_analysis import analyze_dimensions


# ---------------------------------------------------------------- uniquely-determined cases

def test_newtons_second_law_uniquely_determined():
    result = analyze_dimensions({"m": "kg", "a": "m/s^2"}, "N")
    assert result.feasible
    assert result.degrees_of_freedom == 0
    assert result.particular_solution == {"m": Fraction(1), "a": Fraction(1)}
    assert any(c.description == "C * m * a" for c in result.candidates)


def test_kinetic_energy_uniquely_determined():
    result = analyze_dimensions({"m": "kg", "v": "m/s"}, "J")
    assert result.feasible
    assert result.degrees_of_freedom == 0
    assert result.particular_solution == {"m": Fraction(1), "v": Fraction(2)}


def test_pendulum_period_from_length_and_gravity():
    result = analyze_dimensions({"L": "m", "g": "m/s^2"}, "s")
    assert result.feasible
    assert result.degrees_of_freedom == 0
    assert result.particular_solution == {"L": Fraction(1, 2), "g": Fraction(-1, 2)}


def test_density_uniquely_determined():
    result = analyze_dimensions({"m": "kg", "V": "m^3"}, "kg/m^3")
    assert result.feasible
    assert result.particular_solution == {"m": Fraction(1), "V": Fraction(-1)}


# ---------------------------------------------------------------- underdetermined / irrelevant inputs

def test_irrelevant_input_forced_to_zero_exponent_not_a_free_dof():
    """A mass-dimensioned input has no bearing on reaching a pure-time
    target when the other inputs already span length+time -- it should
    be forced to exponent 0, not counted as an extra degree of freedom."""
    result = analyze_dimensions({"L": "m", "g": "m/s^2", "m": "kg"}, "s")
    assert result.feasible
    assert result.degrees_of_freedom == 0
    assert result.particular_solution["m"] == Fraction(0)


def test_genuine_degrees_of_freedom_when_inputs_share_dimensions():
    """Two inputs with the SAME dimension (both plain lengths) given a
    length target is genuinely underdetermined -- infinitely many valid
    exponent splits exist (a/b is a dimensionless ratio that can be
    freely raised to any power)."""
    result = analyze_dimensions({"x": "m", "y": "m"}, "m")
    assert result.feasible
    assert result.degrees_of_freedom > 0
    assert len(result.candidates) > 1


# ---------------------------------------------------------------- infeasible cases

def test_infeasible_when_no_combination_can_work():
    result = analyze_dimensions({"x": "m", "y": "m"}, "s")
    assert not result.feasible
    assert "incompatible" in result.message.lower()
    assert result.particular_solution is None


def test_coulombs_law_infeasible_in_si_since_k_is_not_dimensionless():
    """A well-known SI-specific quirk: unlike F=ma, Coulomb's law's
    constant k = 1/(4*pi*eps0) itself carries dimensions in SI, so no
    purely dimensionless combination of q1, q2, r alone can reach a
    force -- correctly reported as infeasible, not silently wrong."""
    result = analyze_dimensions({"q1": "C", "q2": "C", "r": "m"}, "N")
    assert not result.feasible


# ---------------------------------------------------------------- error handling

def test_empty_inputs_reports_infeasible_not_crash():
    result = analyze_dimensions({}, "N")
    assert not result.feasible
    assert "at least one" in result.message.lower()


def test_unparseable_unit_reports_infeasible_not_crash():
    result = analyze_dimensions({"x": "sprockets"}, "N")
    assert not result.feasible
    assert result.message  # some explanatory message, not silent


def test_unparseable_target_unit_reports_infeasible_not_crash():
    result = analyze_dimensions({"x": "kg"}, "not_a_unit_at_all")
    assert not result.feasible


# ---------------------------------------------------------------- candidate formatting

def test_candidate_description_formats_exponent_one_without_caret():
    result = analyze_dimensions({"m": "kg", "a": "m/s^2"}, "N")
    desc = next(c.description for c in result.candidates if set(c.exponents) == {"m", "a"})
    assert "^1" not in desc
    assert "m" in desc and "a" in desc


def test_candidate_description_formats_negative_one_exponent():
    result = analyze_dimensions({"m": "kg", "V": "m^3"}, "kg/m^3")
    desc = next(c.description for c in result.candidates if "V" in c.exponents)
    assert "V^-1" in desc


# ---------------------------------------------------------------- dimensionless target

def test_dimensionless_target_with_matching_ratio_input():
    """Two lengths combining to a dimensionless target (a pure ratio,
    e.g. strain = dL/L) should be recognized as feasible."""
    result = analyze_dimensions({"dL": "m", "L": "m"}, "dimensionless")
    assert result.feasible
