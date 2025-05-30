import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import html

# === Configuration du logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Paramètres ===

INPUT_PATH  = Path("data/events_raw.json")
OUTPUT_PATH = Path("data/events_clean.json")
SELECT_COLUMNS = ["uid","title_fr","description","firstdate_begin","firstdate_end","location_address" ,"location_city"]

# === Fonctions ===

def load_data(path: Path) -> pd.DataFrame:
    """
    Charge le fichier JSON brut en DataFrame Pandas.
    """
    logging.info(f"Chargement des données brutes depuis {path}…")
    df = pd.read_json(path)
    logging.info(f"{len(df)} enregistrements chargés.")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes dupliquées sur la colonne 'uid', en gardant la première occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset="uid", keep="first")
    removed = before - len(df)
    logging.info(f"Suppression des doublons : {removed} lignes supprimées.")
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes entièrement vides (100% NaN).
    """
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    removed = before - df.shape[1]
    logging.info(f"Colonnes supprimées (100% NaN) : {removed}.")
    return df


def drop_missing_title_or_desc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les entrées dont 'title_fr' OU 'description_fr' est manquant(e).
    """
    before = len(df)
    df = df.dropna(subset=["title_fr", "description_fr"], how="any")
    removed = before - len(df)
    logging.info(f"Lignes supprimées (titre ou description manquants) : {removed}.")
    return df

def clean_html(raw: str) -> str:
    """
    Supprime les balises HTML et décode les entités,
    puis contracte les espaces multiples.
    """
    # 1) Extraire le texte brut
    text = BeautifulSoup(raw or "", "html.parser").get_text(separator=" ")
    # 2) Décoder les entités HTML (&amp; → &)
    text = html.unescape(text)
    # 3) Écraser les multiples espaces / retours-ligne
    return " ".join(text.split())

def combine_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatène title_fr, description_fr et longdescription_fr en un seul champ 'description',
    puis nettoie le HTML.
    """
    def make_desc(row):
        parts = []
        # Titre
        if pd.notna(row["title_fr"]):
            parts.append(f"Titre : {row['title_fr']}")
        # Résumé
        if pd.notna(row["description_fr"]):
            parts.append(f"Résumé : {row['description_fr']}")
        # Détails
        if pd.notna(row["longdescription_fr"]):
            parts.append(f"Détails : {clean_html(row['longdescription_fr'])}")
        # Concatène et nettoie le HTML dans chaque bloc
        return "\n\n".join(parts)

    df["description"] = df.apply(make_desc, axis=1)
    logging.info(f"Champ 'description' créé pour {len(df)} événements.")
    return df

def select_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Ne garde que les colonnes listées.
    """
    if not isinstance(cols, (list, tuple)):
        raise TypeError("SELECT_COLUMNS doit être une liste ou un tuple")
    df = df[list(cols)]
    logging.info(f"Colonnes conservées : {cols}")
    return df

def format_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """
    Formate les colonnes datetime pour afficher 'YYYY-MM-DD HH:MM'.
    """
    for col in date_cols:
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], utc=True, errors="coerce")
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M")
            )
            logging.info(f"Colonne '{col}' reformatée.")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """
    Sauvegarde le DataFrame en JSON indenté.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", force_ascii=False, indent=2)
    logging.info(f"✔ Données nettoyées sauvegardées dans {path}")

# === Exécution principale ===

def main():
    # 1. Chargement
    print(f"Chargement des données brutes depuis {INPUT_PATH} …")
    df = load_data(INPUT_PATH)

    # 2. Nettoyage
    df = drop_duplicates(df)
    df = drop_empty_columns(df)
    df = drop_missing_title_or_desc(df)
    df = combine_descriptions(df)
    
    # 3. Sélection de colonnes (optionnelle)
    if SELECT_COLUMNS:
        df = select_columns(df, SELECT_COLUMNS)
        
        # 3.bis. Formatage des dates si présentes
        date_cols = [c for c in ["firstdate_begin", "firstdate_end"] if c in SELECT_COLUMNS]
        if date_cols:
            df = format_dates(df, date_cols)

    # 4. Sauvegarde
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
