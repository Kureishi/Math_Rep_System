"""'Have I solved this before?': wording, meaning and shape, each optional, each explained; cached embeddings; graceful failure."""
import json

import numpy as np
import pytest

from modules import history, recall
from modules.equation_engine import build_model
from modules.similarity import problem_shape
from modules.solve_compare import deterministic_report
from tests.conftest import KINEMATICS_TWO_TARGET_JSON

CAR = "A car accelerates from 8 m/s to 20 m/s in 6 seconds. How far does it travel?"
CAR_NEW_NUMBERS = "A car accelerates from 12 m/s to 30 m/s in 9 seconds. How far does it travel?"
ROCKET = "A rocket burns fuel at 3 kg/s with an exhaust speed of 2500 m/s. What thrust does it produce?"
CIRCUIT = "A 12 V battery drives current through a 4 ohm resistor. What current flows?"


def corpus(*texts):
    return [{"id": i + 1, "problem_text": t, "domain": "physics", "timestamp": "2026-11-01T10:00:00", "shape": None}
            for i, t in enumerate(texts)]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(recall, "DB_PATH", tmp_path / "embeddings.db")
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")


# A stand-in embedding model: words that mean the same thing share a coordinate, so "automobile" lands next to "car".
_CONCEPTS = {"car": 0, "automobile": 0, "vehicle": 0, "accelerate": 1, "speeds": 1, "rocket": 2, "battery": 3, "resistor": 3}


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(8)
        for w in recall.tokens(t):
            v[_CONCEPTS.get(w, 4)] += 1.0
        out.append(v.tolist())
    return out


# ------------------------------------------------------------------------------------------ wording

def test_numbers_are_set_aside_so_the_same_problem_with_new_numbers_matches():
    assert recall.tokens(CAR) == recall.tokens("A car accelerates from 8 m/s to 20 m/s in 6 seconds. How far does it travel?")
    assert recall.normalised(CAR) == recall.normalised(CAR_NEW_NUMBERS)
    assert "NUM" in recall.tokens("it took 3.5e2 s and -4,5 m")


def test_stop_words_go_and_plurals_fold():
    assert recall.tokens("The cars are going") == ["car", "going"]
    assert recall.tokens("mass") == ["mass"] and recall.tokens("cars") == ["car"]


def test_tfidf_ranks_the_overlapping_text_first_and_unrelated_text_near_zero():
    sims = recall.tfidf_cosine(CAR, [CAR_NEW_NUMBERS, ROCKET, CIRCUIT])
    assert sims[0] > 0.9 and sims[0] > sims[1] and sims[2] < 0.2


def test_tfidf_handles_empty_input():
    assert recall.tfidf_cosine("", ["a car"]) == [0.0]
    assert recall.tfidf_cosine("a car", []) == []


# ------------------------------------------------------------------------------------------ the search

def test_the_same_problem_with_different_numbers_is_called_the_same_and_says_why():
    report = recall.search(CAR_NEW_NUMBERS, corpus(CAR, ROCKET, CIRCUIT))
    best = report.best
    assert best.id == 1 and best.verdict == "same" and best.same_problem_new_numbers
    assert "different numbers" in best.why and report.signals == ["wording"]


def test_an_identical_problem_is_the_same_but_not_flagged_as_new_numbers():
    best = recall.search(CAR, corpus(CAR, ROCKET)).best
    assert best.verdict == "same" and not best.same_problem_new_numbers


def test_unrelated_problems_are_not_returned():
    assert recall.search("Predict the half-life of a radioactive sample", corpus(CAR, ROCKET, CIRCUIT)).matches == []


def test_the_exclude_id_keeps_a_problem_from_matching_itself():
    assert recall.search(CAR, corpus(CAR, ROCKET), exclude_id=1).matches == []


def test_meaning_finds_what_the_words_miss():
    query = "An automobile speeds up from rest"
    items = corpus("A car accelerates uniformly from rest", ROCKET)
    wording_only = recall.search(query, items, min_score=0.2)
    with_meaning = recall.search(query, items, embed=fake_embed, embedding_model="fake", min_score=0.2)
    assert with_meaning.signals[0] == "meaning" and with_meaning.best.id == 1
    assert with_meaning.best.parts["meaning"] > with_meaning.best.parts["wording"]
    assert "similar in meaning" in with_meaning.best.why


def test_the_equation_shape_signal_joins_when_the_question_has_a_model():
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    shape = problem_shape(model)
    items = [{"id": 1, "problem_text": "Work out how a vehicle changes speed", "domain": "physics", "timestamp": "t", "shape": shape},
             {"id": 2, "problem_text": "Work out how a vehicle changes speed", "domain": "physics", "timestamp": "t", "shape": frozenset({"equation:x"})}]
    report = recall.search("how a vehicle changes speed", items, query_shape=shape)
    assert "shape" in report.signals
    assert report.matches[0].id == 1 and report.matches[0].parts["shape"] == 1.0 and "equation shape" in report.matches[0].why


def test_the_score_renormalises_over_the_signals_that_exist():
    a = recall.search(CAR_NEW_NUMBERS, corpus(CAR))
    b = recall.search(CAR_NEW_NUMBERS, corpus(CAR), embed=fake_embed, embedding_model="fake")
    assert 0 < a.best.score <= 1 and 0 < b.best.score <= 1


def test_a_failing_embedding_model_degrades_to_the_other_signals_and_says_so():
    def broken(texts):
        raise ConnectionError("LM Studio went away")
    report = recall.search(CAR_NEW_NUMBERS, corpus(CAR), embed=broken, embedding_model="fake")
    assert report.best.id == 1 and "meaning" not in report.signals
    assert any("Meaning-based matching was skipped" in n for n in report.notes)


def test_no_embedding_model_is_explained_not_hidden():
    assert any("No embedding model" in n for n in recall.search(CAR, corpus(CAR)).notes)


def test_empty_query_or_corpus_gives_an_empty_report():
    assert recall.search("", corpus(CAR)).matches == [] and recall.search(CAR, []).matches == []
    assert recall.search(CAR, corpus("", "   ")).searched == 0


def test_the_limit_and_ordering_are_respected():
    items = corpus(*[f"A car accelerates from {i} m/s to 20 m/s in 6 seconds. How far?" for i in range(1, 9)])
    report = recall.search(CAR, items, limit=3)
    assert len(report.matches) == 3 and [m.score for m in report.matches] == sorted([m.score for m in report.matches], reverse=True)


@pytest.mark.parametrize("score, verdict", [(0.95, "same"), (0.7, "very similar"), (0.5, "related"), (0.2, None)])
def test_verdict_thresholds(score, verdict):
    assert recall.verdict_for(score) == verdict


# ------------------------------------------------------------------------------------------ the embedding cache

def test_each_text_is_embedded_once_across_searches():
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    items = corpus(CAR, ROCKET, CIRCUIT)
    recall.search("a vehicle speeds up", items, embed=counting, embedding_model="m1")
    first = sum(len(c) for c in calls)
    recall.search("a vehicle speeds up", items, embed=counting, embedding_model="m1")
    assert first == 4 and sum(len(c) for c in calls) == 4                            # nothing re-embedded
    recall.search("a rocket burns", items, embed=counting, embedding_model="m1")
    assert sum(len(c) for c in calls) == 5                                           # only the new query


def test_a_different_embedding_model_has_its_own_cache():
    calls = []
    items = corpus(CAR)
    for m in ("m1", "m2"):
        recall.search("a car", items, embed=lambda t: (calls.append(len(t)), fake_embed(t))[1], embedding_model=m)
    assert calls == [2, 2]


def test_a_new_problem_added_later_is_the_only_one_embedded():
    seen = []
    recall.search("a car", corpus(CAR), embed=lambda t: (seen.extend(t), fake_embed(t))[1], embedding_model="m")
    seen.clear()
    recall.search("a car", corpus(CAR, ROCKET), embed=lambda t: (seen.extend(t), fake_embed(t))[1], embedding_model="m")
    assert seen == [ROCKET]


def test_vectors_round_trip_through_the_database():
    v = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    recall.store_vectors("m", {recall.text_hash("hello"): v})
    got = recall.cached_vectors("m", ["hello", "other"])
    assert list(got) == [recall.text_hash("hello")] and np.allclose(got[recall.text_hash("hello")], v)


def test_a_model_returning_the_wrong_number_of_vectors_is_an_error_that_falls_back():
    report = recall.search(CAR_NEW_NUMBERS, corpus(CAR), embed=lambda t: [[1.0]], embedding_model="bad")
    assert "meaning" not in report.signals and report.best is not None and report.notes


def test_progress_is_reported_while_embedding():
    seen = []
    recall.search("q", corpus(*[f"problem number {i} about a car" for i in range(70)]), embed=fake_embed, embedding_model="m",
                  on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1] and len(seen) >= 2


# ------------------------------------------------------------------------------------------ against the real history

def _save(text):
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    return history.save(text, model, deterministic_report(model), {}, [])


def test_search_history_finds_a_saved_problem_by_wording_and_shape():
    first = _save(CAR)
    _save(ROCKET)
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = recall.search_history(CAR_NEW_NUMBERS, query_shape=problem_shape(model))
    assert report.best.id == first and report.best.verdict == "same" and "shape" in report.signals


def test_list_corpus_carries_the_stored_shape():
    _save(CAR)
    (entry,) = history.list_corpus()
    assert entry["problem_text"] == CAR and entry["shape"]


class _Client:
    def __init__(self, models, vectors=fake_embed):
        self.models, self.vectors = models, vectors

    def list_models(self):
        return self.models

    def embed(self, texts, model):
        return self.vectors(texts)


def test_search_history_uses_an_embedding_model_when_one_is_loaded_and_skips_it_otherwise(monkeypatch):
    monkeypatch.delenv("LM_EMBEDDING_MODEL", raising=False)
    _save(CAR)
    assert "meaning" in recall.search_history("an automobile speeds up", _Client(["qwen", "nomic-embed-text"])).signals
    assert "meaning" not in recall.search_history("an automobile speeds up", _Client(["qwen"])).signals


def test_an_unreachable_server_just_means_no_embeddings():
    class Down:
        def list_models(self):
            raise ConnectionError("down")
    _save(CAR)
    report = recall.search_history(CAR, Down())
    assert report.best is not None and "meaning" not in report.signals
