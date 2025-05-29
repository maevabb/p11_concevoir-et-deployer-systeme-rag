import pandas as pd
from pathlib import Path

# === Paramètres ===

INPUT_PATH  = Path("data/events_raw.json")
OUTPUT_PATH = Path("data/events_clean.json")
SELECT_COLUMNS = ["uid","title_fr","description_fr","longdescription_fr","firstdate_begin","firstdate_end"]

# === Fonctions ===

def load_data(path: Path) -> pd.DataFrame:
    """
    Charge le fichier JSON brut en DataFrame Pandas.
    """
    return pd.read_json(path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes dupliquées sur la colonne 'uid', en gardant la première occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset="uid", keep="first")
    after = len(df)
    print(f"→ Suppression des doublons : {before - after} lignes supprimées")
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes entièrement vides (100% NaN).
    """
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after = df.shape[1]
    print(f"→ Colonnes supprimées (100% NaN) : {before - after}")
    return df


def drop_missing_title_or_desc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les entrées dont 'title_fr' OU 'description_fr' est manquant(e).
    """
    before = len(df)
    df = df.dropna(subset=["title_fr", "description_fr"], how="any")
    after = len(df)
    print(f"→ Lignes supprimées (titre ou description manquants) : {before - after}")
    return df

def select_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Ne garde que les colonnes listées.

    Args:
        df (DataFrame): DataFrame en entrée.
        cols (list[str]): 
            - liste de noms de colonnes à conserver
    """
    if isinstance(cols, (list, tuple)):
        df = df[list(cols)]
        print(f"→ Colonnes filtrées : {cols}")
    else:
        raise TypeError("SELECT_COLUMNS doit être une liste")
    return df

def save_data(df: pd.DataFrame, path: Path) -> None:
    """
    Sauvegarde le DataFrame en JSON indenté.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", force_ascii=False, indent=2)
    print(f"✔ Données nettoyées sauvegardées dans {path}")

# === Exécution principale ===

def main():
    # 1. Chargement
    print(f"Chargement des données brutes depuis {INPUT_PATH} …")
    df = load_data(INPUT_PATH)

    # 2. Nettoyage
    df = drop_duplicates(df)
    df = drop_empty_columns(df)
    df = drop_missing_title_or_desc(df)
    
    # 3. Sélection de colonnes (optionnelle)
    if SELECT_COLUMNS:
        df = select_columns(df, SELECT_COLUMNS)
    
    # 4. Sauvegarde
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
