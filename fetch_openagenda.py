import requests
from datetime import datetime, timedelta
import json

# === Paramètres ===
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda"

CITY = "Nantes"

START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
END_DATE = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

EXPORT_FORMAT = "json"

# Paramètres de filtrage
params = {
    "where": f"firstdate_begin >= '{START_DATE}' AND firstdate_begin <= '{END_DATE}' AND location_city = '{CITY}'"
}

# Appel à l’export
url = f"{BASE_URL}/exports/{EXPORT_FORMAT}"
response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()  # liste d'événements JSON
count = len(data if isinstance(data, list) else data.get("results", []))
print(f"Nombre de lignes exportées : {count}")

# Export JSON vers fichier
output_file = "openagenda_export.json"
with open(output_file, "w", encoding="utf-8") as f:
    # Si data est un dict avec 'results', on exporte data['results']
    payload = data if isinstance(data, list) else data.get("results", [])
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"✅ Export JSON terminé : {output_file}")

