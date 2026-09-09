import pytest

import modules.settings_profiles as sp_module
from modules.settings_profiles import save_profile, list_profiles, load_profile, delete_profile, apply_profile
from config import Settings


@pytest.fixture(autouse=True)
def _redirect_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sp_module, "DB_PATH", tmp_path / "test_settings_profiles.db")


def test_save_and_load_profile_round_trip():
    settings = Settings()
    settings.temperature_extraction = 0.05
    settings.max_verification_retries = 5
    settings.computation_timeout_seconds = 30.0
    save_profile("strict", settings)

    loaded = load_profile("strict")
    assert loaded.name == "strict"
    assert loaded.temperature_extraction == pytest.approx(0.05)
    assert loaded.max_verification_retries == 5
    assert loaded.computation_timeout_seconds == pytest.approx(30.0)


def test_list_profiles_returns_saved_names_sorted():
    settings = Settings()
    save_profile("zeta", settings)
    save_profile("alpha", settings)
    assert list_profiles() == ["alpha", "zeta"]


def test_load_nonexistent_profile_returns_none():
    assert load_profile("does not exist") is None


def test_save_overwrites_existing_profile_with_same_name():
    settings = Settings()
    settings.temperature_extraction = 0.1
    save_profile("myprofile", settings)

    settings.temperature_extraction = 0.9
    save_profile("myprofile", settings)  # same name -- should overwrite, not duplicate

    assert list_profiles() == ["myprofile"]
    loaded = load_profile("myprofile")
    assert loaded.temperature_extraction == pytest.approx(0.9)


def test_delete_profile():
    settings = Settings()
    save_profile("temp", settings)
    assert "temp" in list_profiles()
    delete_profile("temp")
    assert "temp" not in list_profiles()


def test_delete_nonexistent_profile_does_not_raise():
    delete_profile("never existed")  # should not raise


def test_save_profile_rejects_empty_name():
    settings = Settings()
    with pytest.raises(ValueError):
        save_profile("", settings)
    with pytest.raises(ValueError):
        save_profile("   ", settings)


def test_save_profile_strips_whitespace_from_name():
    settings = Settings()
    save_profile("  padded  ", settings)
    assert "padded" in list_profiles()


# ---------------------------------------------------------------- apply_profile

def test_apply_profile_mutates_settings_in_place():
    source = Settings()
    source.temperature_extraction = 0.02
    source.temperature_narration = 0.9
    source.max_verification_retries = 7
    source.numeric_tolerance = 1e-9
    source.cross_check_tolerance = 0.5
    source.computation_timeout_seconds = 99.0
    save_profile("custom", source)
    profile = load_profile("custom")

    target = Settings()  # a fresh instance, at defaults
    apply_profile(profile, target)

    assert target.temperature_extraction == pytest.approx(0.02)
    assert target.temperature_narration == pytest.approx(0.9)
    assert target.max_verification_retries == 7
    assert target.numeric_tolerance == pytest.approx(1e-9)
    assert target.cross_check_tolerance == pytest.approx(0.5)
    assert target.computation_timeout_seconds == pytest.approx(99.0)


def test_apply_profile_never_touches_connection_fields():
    """Profiles cover verification/generation tuning only -- applying
    one must never change lm_studio_base_url, model names, etc."""
    target = Settings()
    original_url = target.lm_studio_base_url
    original_model = target.reasoning_model

    source = Settings()
    save_profile("p", source)
    apply_profile(load_profile("p"), target)

    assert target.lm_studio_base_url == original_url
    assert target.reasoning_model == original_model


def test_multiple_profiles_are_independent():
    fast = Settings()
    fast.computation_timeout_seconds = 3.0
    fast.max_verification_retries = 0
    save_profile("fast", fast)

    strict = Settings()
    strict.computation_timeout_seconds = 60.0
    strict.max_verification_retries = 5
    save_profile("strict", strict)

    loaded_fast = load_profile("fast")
    loaded_strict = load_profile("strict")
    assert loaded_fast.computation_timeout_seconds == pytest.approx(3.0)
    assert loaded_strict.computation_timeout_seconds == pytest.approx(60.0)
    assert loaded_fast.max_verification_retries == 0
    assert loaded_strict.max_verification_retries == 5
