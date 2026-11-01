"""Tests for modules/cc_notebook.py: exported notebooks execute and reproduce their recorded values."""
import json

import pytest

from modules import cc_notebook as nb

FAST = {"lattice_scan": {"temps": [2.0, 2.6], "n_therm": 100, "n_meas": 600, "L": 6}, "einstein": {"n_paths": 1500, "t_max": 6.0},
        "random_matrix": {"N": 60, "n_samples": 8}, "geodesic_precession": {"orbits": 2}, "integrators": {"t_end": 100.0}}


def seed_for(key):
    return 7 if nb.RECIPES[key].uses_seed else None


@pytest.mark.parametrize("key", list(nb.RECIPES))
def test_every_recipe_exports_a_notebook_that_reproduces_itself(key):
    rec = nb.make_record(key, FAST.get(key), seed_for(key))
    ok, msg = nb.verify_notebook(nb.build_research_notebook(rec))
    assert ok, (key, msg)
    assert len(rec.summary) >= 2


def test_the_same_seed_gives_identical_numbers_and_a_different_seed_does_not():
    a = nb.make_record("lattice_scan", FAST["lattice_scan"], 3)
    b = nb.make_record("lattice_scan", FAST["lattice_scan"], 3)
    c = nb.make_record("lattice_scan", FAST["lattice_scan"], 4)
    assert a.summary == b.summary and a.summary != c.summary and a.seed == 3


def test_a_fresh_seed_is_drawn_stored_and_reproducible():
    rec = nb.make_record("einstein", FAST["einstein"], None)
    assert isinstance(rec.seed, int)
    assert nb.verify_notebook(nb.build_research_notebook(rec))[0]
    assert nb.make_record("partition").seed is None                                                # deterministic recipes carry no seed


def _edit_expected(js: str, mutate) -> str:
    book = json.loads(js)
    for cell in book["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.startswith("EXPECTED"):
            body = json.loads(src[len("EXPECTED = "): src.index("\nTOLERANCE")])
            mutate(body)
            cell["source"] = [f"EXPECTED = {json.dumps(body)}\n" + src[src.index("TOLERANCE"):]]
    return json.dumps(book)


def test_a_tampered_recorded_value_makes_the_notebook_fail_loudly():
    rec = nb.make_record("partition")
    js = nb.build_research_notebook(rec)
    key = next(k for k, v in rec.summary.items() if k.startswith("U@"))
    bad = _edit_expected(js, lambda d: d.__setitem__(key, d[key] * 1.01))
    ok, msg = nb.verify_notebook(bad)
    assert not ok and "NOT REPRODUCED" in msg and key in msg
    bad_text = _edit_expected(js, lambda d: d.__setitem__("checks_passed", "no"))
    assert not nb.verify_notebook(bad_text)[0]                                                    # a recorded string that differs is caught too


def test_changing_a_parameter_without_updating_the_record_is_caught():
    rec = nb.make_record("partition")
    js = nb.build_research_notebook(rec).replace('\\"gap\\": 1.0', '\\"gap\\": 1.5')
    assert js != nb.build_research_notebook(rec)
    assert not nb.verify_notebook(js)[0]


def test_generated_code_carries_data_only_and_valid_json():
    rec = nb.make_record("nondim")
    book = json.loads(nb.build_research_notebook(rec))
    assert book["nbformat"] == 4 and book["metadata"]["math_rep_system"]["recipe"] == "nondim"
    sources = ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"]
    assert any(nb.literal_params_ok(s) for s in sources)
    md = "".join(book["cells"][0]["source"])
    assert "numpy" in md and "seed" in md and "tolerance" in md
    script = nb.build_script(rec)
    assert "PARAMS = " in script and compile(script, "<script>", "exec")


@pytest.mark.parametrize("params,msg", [({"gap": [object()]}, "plain number"), ({"nope": 1}, "Unknown parameter"), ({"gap": "wide"}, "must be a number"),
                                        ({"gap": float("nan")}, "NaN")])
def test_unsafe_or_unknown_parameters_are_refused(params, msg):
    with pytest.raises(nb.CrossCuttingError, match=msg):
        nb.make_record("partition", params)


def test_unknown_recipe_and_broken_notebooks():
    with pytest.raises(nb.CrossCuttingError, match="Unknown recipe"):
        nb.make_record("nope")
    with pytest.raises(nb.CrossCuttingError, match="Not a valid notebook"):
        nb.notebook_code("{not json")
    ok, msg = nb.verify_notebook(json.dumps({"cells": [{"cell_type": "code", "source": ["raise RuntimeError('boom')"]}]}))
    assert not ok and "failed to run" in msg and "boom" in msg
    assert nb.literal_params_ok("PARAMS = {'a': 1}") and not nb.literal_params_ok("PARAMS = {'a': __import__('os')}") and not nb.literal_params_ok("x = 1")
