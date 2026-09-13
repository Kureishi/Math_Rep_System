"""Tests for modules/unit_conversion.py's preferred_conversion() --
the global unit-system preference feature."""
from modules.unit_conversion import preferred_conversion


def test_si_value_gets_imperial_suggestion():
    result = preferred_conversion(10.0, "m", "imperial")
    assert result is not None
    alt_unit, alt_val = result
    assert alt_unit == "ft"
    assert abs(alt_val - 32.8084) < 0.01


def test_imperial_value_gets_si_suggestion():
    result = preferred_conversion(10.0, "ft", "SI")
    assert result is not None
    alt_unit, alt_val = result
    assert alt_unit == "m"
    assert abs(alt_val - 3.048) < 0.01


def test_value_already_in_preferred_system_returns_none():
    assert preferred_conversion(10.0, "m", "SI") is None
    assert preferred_conversion(10.0, "ft", "imperial") is None


def test_system_neutral_unit_returns_none():
    assert preferred_conversion(10.0, "s", "imperial") is None
    assert preferred_conversion(10.0, "s", "SI") is None


def test_dimension_with_no_alternate_in_that_system_returns_none():
    assert preferred_conversion(10.0, "W", "imperial") is None


def test_invalid_system_returns_none():
    assert preferred_conversion(10.0, "m", "metric") is None


def test_none_unit_returns_none():
    assert preferred_conversion(10.0, None, "SI") is None
