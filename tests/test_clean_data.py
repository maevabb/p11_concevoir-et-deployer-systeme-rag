import json
import pandas as pd
import pytest
from pathlib import Path

from scripts.clean_data import (
    load_data,
    drop_duplicates,
    drop_empty_columns,
    drop_missing_title_or_desc,
    save_data,
)

# === Fixtures ===

@pytest.fixture
def sample_records(tmp_path):
    """
    Crée un petit fichier JSON d’événements pour les tests.
    - uid en double
    - colonnes entièrement vides
    - quelques titres ou descriptions manquants
    """
    data = [
        {"uid": "1", "title_fr": "T1", "description_fr": "D1", "empty_col": None},
        {"uid": "1", "title_fr": "T1", "description_fr": "D1", "empty_col": None},  # duplicate
        {"uid": "2", "title_fr": None, "description_fr": "D2", "empty_col": None},  # missing title
        {"uid": "3", "title_fr": "T3", "description_fr": None, "empty_col": None},  # missing desc
        {"uid": "4", "title_fr": "T4", "description_fr": "D4", "empty_col": None},
    ]
    path = tmp_path / "data" / "events_raw.json"
    path.parent.mkdir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path, data

# === Tests ===

def test_load_data(sample_records):
    path, original = sample_records
    df = load_data(path)
    # on retrouve toutes les lignes
    assert isinstance(df, pd.DataFrame)
    # on compare en strings pour tenir compte du typage automatique de read_json
    uids = list(df["uid"].astype(str))
    expected = [r["uid"] for r in original]
    assert uids == expected

def test_drop_duplicates():
    df = pd.DataFrame([
        {"uid": "a"}, {"uid": "a"}, {"uid": "b"}
    ])
    df2 = drop_duplicates(df)
    assert list(df2["uid"]) == ["a", "b"]  # garde la première de chaque

def test_drop_empty_columns():
    df = pd.DataFrame({
        "keep": [1, 2],
        "all_null": [None, None],
        "some_val": [None, 3]
    })
    df2 = drop_empty_columns(df)
    assert "all_null" not in df2.columns
    assert "keep" in df2.columns and "some_val" in df2.columns

def test_drop_missing_title_or_desc():
    df = pd.DataFrame([
        {"title_fr": "T", "description_fr": "D"},
        {"title_fr": None, "description_fr": "D"},
        {"title_fr": "T2", "description_fr": None},
    ])
    df2 = drop_missing_title_or_desc(df)
    # ne garde que la première ligne
    assert len(df2) == 1
    assert df2.iloc[0]["title_fr"] == "T"

def test_save_data(tmp_path):
    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    out = tmp_path / "foo" / "bar.json"
    save_data(df, out)
    # le fichier existe et contient le bon JSON
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == [{"a": 1}, {"a": 2}]
