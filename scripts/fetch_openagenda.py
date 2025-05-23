import requests
from datetime import datetime, timedelta
import json
from pathlib import Path

# === Paramètres ===
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda"
OUTPUT_PATH = "data/events_raw.json"

CITY = "Paris"

START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
END_DATE = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

EXPORT_FORMAT = "json"

# === Requête API ===
def fetch_openagenda_events(base_url: str, 
                            export_format: str, 
                            limit: int = -1, 
                            offset: int = 0, 
                            start_date: str | None = None, 
                            end_date: str | None = None, 
                            city: str | None = None):
    """
    Récupère des événements depuis l'API OpenAgenda (/exports/{format}),
    avec filtres optionnels sur la plage de dates et la ville.

    Args:
        base_url (str): URL de base de l'API OpenAgenda.
        export_format (str): Format d'export ('json', 'csv', 'parquet').
        limit (int): Nombre max d'enregistrements à obtenir (-1 = pas de limite).
        offset (int): Nombre d'enregistrements à ignorer avant le début.
        start_date (str | None): Date ISO de début (inclus) pour firstdate_begin.
        end_date (str | None): Date ISO de fin (inclus) pour firstdate_begin.
        city (str | None): Nom de la ville pour filtrer (requiert start_date et end_date).

    Returns:
        tuple[list[dict], int]: 
            - Liste des événements (chaque événement est un dict).  
            - Nombre total d'événements récupérés.
    """
    # Définition des paramètres
    params = {
        "limit": limit,
        "offset": offset,
    }
    conditions: list[str] = []
    if start_date:
        conditions.append(f"firstdate_begin >= '{start_date}'")
    if end_date:
        conditions.append(f"firstdate_begin <= '{end_date}'")
    if city:
        conditions.append(f"location_city = '{city}'")

    if conditions:
        params["where"] = " AND ".join(conditions)

    # Appel à l’export
    url = f"{base_url}/exports/{export_format}"
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        events = data
        total = len(data)
    else:
        events = data.get("results", [])
        total = data.get("total_count", len(events))

    return events, total

# === Enregistrement dans un fichier JSON ===
def save_raw_events(events: list, output_path: str = "data/events_raw.json") -> None:
    """
    Sauvegarde la liste brute des événements dans un fichier JSON.

    Args:
        events (list): Liste d'événements.
        output_path (str): Chemin du fichier de sortie.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"📁 Données brutes sauvegardées dans {output_path}")

# === Exécution principale ===
if __name__ == "__main__":
    print(f"➡️ Chargement des événements pour {CITY} entre {START_DATE} et {END_DATE}")
    events, total = fetch_openagenda_events(BASE_URL, EXPORT_FORMAT, -1, 0, START_DATE, END_DATE, CITY)  
    save_raw_events(events, OUTPUT_PATH)

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        loaded = json.load(f)
    count = len(loaded) if isinstance(loaded, list) else 0

    print(f"📊 Nombre d'événements dans le fichier : {count} (total attendu : {total})")