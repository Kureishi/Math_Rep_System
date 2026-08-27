import pytest

from modules.unit_conversion import convert_value, sweep_conversions


# ---------------------------------------------------------------- convert_value


def test_convert_meters_to_feet():
    assert convert_value(5.0, "m", "ft") == pytest.approx(16.4041994751, rel=1e-6)


def test_convert_kg_to_lb():
    assert convert_value(100.0, "kg", "lb") == pytest.approx(220.462262, rel=1e-5)


def test_convert_mps_to_kmh():
    assert convert_value(10.0, "m/s", "km/hr") == pytest.approx(36.0, rel=1e-6)


def test_convert_mismatched_dimensions_returns_none():
    assert convert_value(5.0, "m", "kg") is None


def test_convert_unparseable_unit_returns_none():
    assert convert_value(5.0, "m", "sprockets") is None


def test_convert_dimensionless_returns_none():
    assert convert_value(5.0, "USD", "USD") is None


def test_convert_compound_units():
    # 1000 N should equal 1000 kg*m/s^2
    assert convert_value(1000.0, "N", "kg*m/s^2") == pytest.approx(1000.0, rel=1e-6)


def test_convert_round_trip_is_consistent():
    original = 42.0
    converted = convert_value(original, "m", "ft")
    back = convert_value(converted, "ft", "m")
    assert back == pytest.approx(original, rel=1e-9)


# ---------------------------------------------------------------- sweep_conversions


def test_sweep_velocity():
    results = sweep_conversions(10.0, "m/s")
    units = dict(results)
    assert units["km/hr"] == pytest.approx(36.0, rel=1e-6)
    assert "mph" in units
    assert "ft/s" in units
    assert "m/s" not in units  # excludes the original unit


def test_sweep_length():
    results = sweep_conversions(5.0, "m")
    units = dict(results)
    assert "ft" in units and "km" in units and "mi" in units
    assert "m" not in units


def test_sweep_mass():
    results = sweep_conversions(100.0, "kg")
    units = dict(results)
    assert units["lb"] == pytest.approx(220.462262, rel=1e-5)


def test_sweep_time():
    results = sweep_conversions(3600.0, "s")
    units = dict(results)
    assert units["hr"] == pytest.approx(1.0, rel=1e-6)
    assert units["min"] == pytest.approx(60.0, rel=1e-6)


def test_sweep_dimensionless_returns_empty():
    assert sweep_conversions(50.0, "USD") == []
    assert sweep_conversions(50.0, "%") == []


def test_sweep_none_unit_returns_empty():
    assert sweep_conversions(50.0, None) == []


def test_sweep_unknown_unit_returns_empty():
    assert sweep_conversions(1.0, "sprockets") == []


def test_sweep_unrecognized_dimension_not_in_any_profile_returns_empty():
    # moles isn't covered by any _ALTERNATE_PROFILES entry -- should
    # gracefully return nothing rather than error
    assert sweep_conversions(2.0, "mol") == []
