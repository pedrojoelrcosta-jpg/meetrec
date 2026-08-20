"""Voiceprint matching tests with synthetic embedding vectors."""

import numpy as np
import pytest

from meetrec.voiceprints import VoiceprintDB, cosine_similarity


@pytest.fixture
def db(tmp_path):
    database = VoiceprintDB(tmp_path / "voices.db")
    yield database
    database.close()


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cosine_similarity_basics():
    assert cosine_similarity(unit([1, 0, 0]), unit([1, 0, 0])) == pytest.approx(1.0)
    assert cosine_similarity(unit([1, 0, 0]), unit([0, 1, 0])) == pytest.approx(0.0)
    assert cosine_similarity(np.zeros(3), unit([1, 0, 0])) == 0.0


def test_match_above_threshold(db):
    db.add("Ana", unit([1.0, 0.1, 0.0]))
    result = db.match(unit([1.0, 0.15, 0.02]), threshold=0.75)
    assert result is not None
    name, score = result
    assert name == "Ana"
    assert score > 0.99


def test_no_match_below_threshold(db):
    db.add("Ana", unit([1.0, 0.0, 0.0]))
    assert db.match(unit([0.0, 1.0, 0.0]), threshold=0.75) is None


def test_best_of_multiple_speakers(db):
    db.add("Ana", unit([1.0, 0.0, 0.0]))
    db.add("Bruno", unit([0.9, 0.4, 0.0]))
    name, _ = db.match(unit([0.92, 0.39, 0.0]), threshold=0.75)
    assert name == "Bruno"


def test_add_same_name_folds_running_mean(db):
    db.add("Ana", np.array([1.0, 0.0], dtype=np.float32))
    db.add("Ana", np.array([0.0, 1.0], dtype=np.float32))
    speakers = db.list_speakers()
    assert len(speakers) == 1
    assert speakers[0]["num_samples"] == 2
    name, score = db.match(unit([1.0, 1.0]), threshold=0.5)
    assert name == "Ana"
    assert score == pytest.approx(1.0)  # mean is [0.5, 0.5], same direction


def test_rename_and_delete(db):
    db.add("SPEAKER_00", unit([1.0, 2.0, 3.0]))
    assert db.rename("SPEAKER_00", "Carla")
    assert [s["name"] for s in db.list_speakers()] == ["Carla"]
    assert db.delete("Carla")
    assert db.list_speakers() == []
    assert not db.delete("Carla")


def test_threshold_is_tunable(db):
    db.add("Ana", unit([1.0, 0.3, 0.0]))
    probe = unit([1.0, -0.25, 0.1])
    strict = db.match(probe, threshold=0.99)
    loose = db.match(probe, threshold=0.5)
    assert strict is None
    assert loose is not None and loose[0] == "Ana"
