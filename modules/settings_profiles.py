"""
Settings profiles: named, savable/loadable presets for the Advanced
Settings tuning knobs (verification tolerances, retry counts, timeouts,
generation temperatures) -- distinct from config.py's Settings object,
which is a single live-tweaked instance that resets to its hardcoded
defaults every session. A researcher switching between "run this fast
and loose while exploring" and "verify this strictly before I cite it"
currently has to remember and re-type each slider's value by hand every
time; this lets that be a one-click named profile switch instead.

Persisted to a small local SQLite table (in the same data/ directory
history.db and chains.db already live in) so profiles survive across
sessions -- unlike the live Settings object itself, this module's whole
purpose is to NOT reset when the browser tab closes.
"""
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "settings_profiles.db"

# the specific config.py Settings fields a profile actually covers --
# deliberately excludes connection config (lm_studio_base_url, the api
# key, model names), which is per-machine setup, not a "how strict/fast
# should verification be" tuning choice a profile is meant for
PROFILE_FIELDS = (
    "temperature_extraction", "temperature_narration", "max_verification_retries",
    "numeric_tolerance", "cross_check_tolerance", "computation_timeout_seconds",
)


@dataclass
class SettingsProfile:
    name: str
    temperature_extraction: float
    temperature_narration: float
    max_verification_retries: int
    numeric_tolerance: float
    cross_check_tolerance: float
    computation_timeout_seconds: float


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings_profiles (
            name TEXT PRIMARY KEY,
            temperature_extraction REAL NOT NULL,
            temperature_narration REAL NOT NULL,
            max_verification_retries INTEGER NOT NULL,
            numeric_tolerance REAL NOT NULL,
            cross_check_tolerance REAL NOT NULL,
            computation_timeout_seconds REAL NOT NULL
        )
    """)
    return conn


def save_profile(name: str, settings_obj) -> None:
    """Saves (or overwrites, if `name` already exists) a named profile
    from the given config.py Settings instance's CURRENT values."""
    name = name.strip()
    if not name:
        raise ValueError("Profile name can't be empty.")
    values = {f: getattr(settings_obj, f) for f in PROFILE_FIELDS}
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings_profiles (name, temperature_extraction, temperature_narration, "
            "max_verification_retries, numeric_tolerance, cross_check_tolerance, "
            "computation_timeout_seconds) VALUES (:name, :temperature_extraction, "
            ":temperature_narration, :max_verification_retries, :numeric_tolerance, "
            ":cross_check_tolerance, :computation_timeout_seconds) "
            "ON CONFLICT(name) DO UPDATE SET "
            "temperature_extraction=excluded.temperature_extraction, "
            "temperature_narration=excluded.temperature_narration, "
            "max_verification_retries=excluded.max_verification_retries, "
            "numeric_tolerance=excluded.numeric_tolerance, "
            "cross_check_tolerance=excluded.cross_check_tolerance, "
            "computation_timeout_seconds=excluded.computation_timeout_seconds",
            {"name": name, **values},
        )


def list_profiles() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT name FROM settings_profiles ORDER BY name").fetchall()
    return [r[0] for r in rows]


def load_profile(name: str):
    with _connect() as conn:
        row = conn.execute(
            "SELECT name, temperature_extraction, temperature_narration, max_verification_retries, "
            "numeric_tolerance, cross_check_tolerance, computation_timeout_seconds "
            "FROM settings_profiles WHERE name = ?", (name,),
        ).fetchone()
    if row is None:
        return None
    return SettingsProfile(*row)


def delete_profile(name: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM settings_profiles WHERE name = ?", (name,))


def apply_profile(profile: SettingsProfile, settings_obj) -> None:
    """Applies a profile's values onto the given live Settings instance
    IN PLACE (mutates `settings_obj`) -- the same pattern app.py's own
    "Reset to defaults" button already uses for config.py's own
    hardcoded Settings() defaults."""
    for f in PROFILE_FIELDS:
        setattr(settings_obj, f, getattr(profile, f))
