import pytest

from modules.units_checker import parse_unit, dimension_of, dims_equivalent, UnitParseError


def test_simple_unit_parses():
    assert str(parse_unit("m")) == "meter"


def test_compound_unit_parses():
    result = parse_unit("m/s^2")
    assert "second" in str(result)
    assert "meter" in str(result)


def test_whitespace_separated_unit_parses():
    """kg m/s^2 (space-separated) should parse via implicit multiplication
    normalization, not choke on the missing '*'."""
    result = parse_unit("kg m/s^2")
    assert "kilogram" in str(result)
    assert "meter" in str(result)


def test_none_and_empty_are_dimensionless():
    assert parse_unit(None) == 1
    assert parse_unit("") == 1
    assert parse_unit("unitless") == 1


def test_unrecognized_unit_raises_cleanly():
    """This is a real bug that showed up during development: unrecognized
    multi-letter tokens like 'sprockets' were being silently decomposed
    into products of known single-letter units (e.g. the 's' in
    'sprockets' resolving to 'second') by sympy's implicit-multiplication
    parser, producing a bogus dimension instead of failing loudly."""
    with pytest.raises(UnitParseError):
        parse_unit("sprockets")


def test_velocity_and_acceleration_times_time_are_equivalent():
    v_unit = parse_unit("m/s")
    a_unit = parse_unit("m/s^2")
    t_unit = parse_unit("s")
    v_dim = dimension_of(v_unit)
    at_dim = dimension_of(a_unit * t_unit)
    assert dims_equivalent(v_dim, at_dim)


def test_distance_and_velocity_are_not_equivalent():
    d_dim = dimension_of(parse_unit("m"))
    v_dim = dimension_of(parse_unit("m/s"))
    assert not dims_equivalent(d_dim, v_dim)


def test_mph_recognized():
    result = parse_unit("mph")
    assert "mile" in str(result)
    assert "hour" in str(result)
