import json
import pytest
from pathlib import Path
import requests
from scripts.fetch_openagenda import (
    fetch_openagenda_events,
    save_raw_events,
    BASE_URL,
    EXPORT_FORMAT,
)

# === Fixtures ===

@pytest.fixture
def sample_events():
    """
    Charge un petit échantillon d'événements JSON stocké dans tests/fixtures/events.json.
    Le fichier fixtures/events.json doit contenir à la fois :
      - un objet {"results": […], "total_count": N}
      - une liste brute […]
    """
    p = Path(__file__).parent / "fixtures" / "events.json"
    return json.loads(p.read_text(encoding="utf-8"))

# === Tests ===

def test_fetch_http_error(requests_mock):
    """
    Vérifie que fetch_openagenda_events lève une HTTPError
    lorsque l'API renvoie un statut 500.
    """
    url = f"{BASE_URL}/exports/{EXPORT_FORMAT}"
    # on simule un 500
    requests_mock.get(url, status_code=500)
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_openagenda_events(BASE_URL, EXPORT_FORMAT)


def test_where_clause_is_built(mocker):
    """
    Vérifie que la clause SQL 'where' est correctement construite
    en fonction des paramètres start_date, end_date et city.
    """
    # on patch requests.get pour intercepter l'appel
    mocked = mocker.patch("scripts.fetch_openagenda.requests.get")
    mocked.return_value.status_code = 200
    mocked.return_value.json.return_value = []
    fetch_openagenda_events(
        BASE_URL,
        EXPORT_FORMAT,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-12-31T23:59:59Z",
        city="Nantes",
    )
    # on récupère les kwargs passés à requests.get
    _, kwargs = mocked.call_args
    params = kwargs.get("params", {})
    assert "where" in params
    where = params["where"]
    assert "firstdate_begin >= '2024-01-01T00:00:00Z'" in where
    assert "firstdate_begin <= '2024-12-31T23:59:59Z'" in where
    assert "location_city = 'Nantes'" in where


def test_parsing_list_and_dict(requests_mock, sample_events):
    """
    Vérifie que la fonction retourne bien la liste d'événements
    et le nombre d'éléments correspondant lorsque l'API renvoie un JSON array.
    """
    url = f"{BASE_URL}/exports/{EXPORT_FORMAT}"
    list_events = sample_events.get("list", sample_events.get("results", []))

    # Cas liste brute
    requests_mock.get(url, json=list_events)
    evts, tot = fetch_openagenda_events(BASE_URL, EXPORT_FORMAT)
    assert isinstance(evts, list)
    assert tot == len(evts)


def test_save_raw_events(tmp_path, sample_events):
    """
    Vérifie que save_raw_events écrit correctement la liste d'événements
    dans un fichier JSON au chemin spécifié.
    """
    list_events = sample_events["results"]
    out = tmp_path / "data" / "events_raw.json"
    save_raw_events(list_events, output_path=str(out))
    content = json.loads(out.read_text(encoding="utf-8"))
    assert content == list_events