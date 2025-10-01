import datetime as dt
from pathlib import Path

import pytest

from src import utils


def test_parse_date_iso():
    timestamp = utils.parse_date("2024-04-01T12:30:00Z")
    assert timestamp is not None
    assert timestamp.year == 2024
    assert timestamp.tzinfo is not None


def test_within_days():
    now = dt.datetime(2024, 4, 10, tzinfo=dt.timezone.utc)
    recent = dt.datetime(2024, 4, 5, tzinfo=dt.timezone.utc)
    assert utils.within_days(recent, 7, now=now)
    old = dt.datetime(2024, 3, 20, tzinfo=dt.timezone.utc)
    assert not utils.within_days(old, 7, now=now)


def test_hash_dict_stable():
    data = {"a": 1, "b": 2}
    assert utils.hash_dict(data) == utils.hash_dict({"b": 2, "a": 1})


def test_compute_score_weights():
    weights = {"source_trust": 0.3, "novelty": 0.2, "sector_impact": 0.2, "tr_relevance": 0.2, "diversity": 0.1}
    score = utils.compute_score(1.0, 0.5, 0.6, 0.4, 0.8, weights)
    expected = 1.0 * 0.3 + 0.5 * 0.2 + 0.6 * 0.2 + 0.4 * 0.2 + 0.8 * 0.1
    assert abs(score - expected) < 1e-9


def test_cache_roundtrip(tmp_path: Path):
    key = "abc"
    cache_file = tmp_path / f"{key}.json"
    data = {"value": 42}
    utils.save_cached_json(cache_file, data)
    loaded = utils.load_cached_json(cache_file)
    assert loaded == data


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First sentence. Second sentence!", ["First sentence", "Second sentence", "Gelişme hakkında ayrıntı bekleniyor."]),
        ("", ["Gelişme hakkında ayrıntı bekleniyor.", "Gelişme hakkında ayrıntı bekleniyor.", "Gelişme hakkında ayrıntı bekleniyor."]),
    ],
)
def test_bullets_from_text(text, expected):
    assert utils.bullets_from_text(text) == expected
