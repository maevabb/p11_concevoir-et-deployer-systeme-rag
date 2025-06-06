import os
import json
import datetime
import random
import warnings
import logging

import pytest
import pandas as pd
import numpy as np
import faiss
from mistralai import Mistral

from pathlib import Path
from dateutil.relativedelta import relativedelta

from scripts.vectorize import chunk_events, embed_batch
from puls_events_chatbot import retrieve_top_k_chunks

# === PARAMETRES ===

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "events_clean.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.idx"
FAISS_METADATA_PATH = PROJECT_ROOT / "data" / "faiss_metadata.json"

# === TESTS ===

def test_events_within_one_year():
    """
    Vérifie que tous les événements dans events_clean.json 
    sont datés à ±1 an de la date du jour.
    """
    logging.info("Démarrage du test : test_events_within_one_year")
    df = pd.read_json(CLEAN_PATH)
    assert not df.empty, "Le fichier events_clean.json est vide."
    logging.info("Nombre d'événements chargés pour vérification temporelle : %d", len(df))

    today = datetime.datetime.now().date()
    one_year_ago = today - relativedelta(years=1)
    one_year_future = today + relativedelta(years=1)

    def parse_date(s: str) -> datetime.date:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").date()

    for idx, row in df.iterrows():
        date_begin = parse_date(row["firstdate_begin"])
        assert one_year_ago <= date_begin <= one_year_future, (
            f"Événement hors période ±1 an détecté: UID={row['uid']}, date={date_begin}"
        )
    logging.info("test_events_within_one_year terminé avec succès.")


def test_events_in_idf_region():
    """
    Vérifie que tous les événements proviennent bien de la région Île-de-France
    en contrôlant le champ 'location_region'.
    """
    logging.info("Démarrage du test : test_events_in_idf_region")
    df = pd.read_json(CLEAN_PATH)
    assert not df.empty, "Le fichier events_clean.json est vide."
    logging.info("Nombre d'événements chargés pour vérification de la région : %d", len(df))

    for idx, row in df.iterrows():
        region = row.get("location_region", "")
        assert region == "Île-de-France", (
            f"Événement hors Île-de-France détecté : UID={row['uid']}, location_region={region}"
        )
    logging.info("test_events_in_idf_region terminé avec succès.")


def test_chunks_count_matches_metadata():
    """
    Vérifie que le nombre total de chunks générés correspond à la taille de la liste metadata (JSON).
    """
    logging.info("Démarrage du test : test_chunks_count_matches_metadata")
    # Charge events_clean
    with CLEAN_PATH.open(encoding="utf-8") as f:
        events = json.load(f)
    logging.info("Nombre d'événements à chunker : %d", len(events))

    # Génère les chunks (texte + metadata) à partir de events
    chunks, metadata = chunk_events(events)
    logging.info("Chunks générés : %d", len(chunks))
    logging.info("Metadata interne générée : %d", len(metadata))

    # Charge le metadata JSON qui a été sauvegardé pour Faiss
    with FAISS_METADATA_PATH.open(encoding="utf-8") as f:
        saved_metadata = json.load(f)
    logging.info("Metadata sauvegardée sur disque : %d", len(saved_metadata))

    assert len(chunks) == len(metadata) == len(saved_metadata), (
        "Incohérence entre les chunks générés et les métadonnées sauvegardées : "
        f"{len(chunks)} chunks contre {len(saved_metadata)} entrées de métadonnées sauvegardées"
    )
    logging.info("test_chunks_count_matches_metadata terminé avec succès.")


def test_faiss_index_vector_count():
    """
    Vérifie que l'index Faiss contient autant de vecteurs que de chunks générés.
    """
    logging.info("Démarrage du test : test_faiss_index_vector_count")
    # Charge events_clean et génère chunks
    with CLEAN_PATH.open(encoding="utf-8") as f:
        events = json.load(f)
    chunks, _ = chunk_events(events)
    logging.info("Nombre de chunks générés pour vérification Faiss : %d", len(chunks))

    # Charge l'index Faiss
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    ntotal = index.ntotal
    logging.info("Nombre de vecteurs dans Faiss index : %d", ntotal)

    assert ntotal == len(chunks), (
        f"Le nombre de vecteurs dans Faiss ({ntotal}) ne correspond pas au nombre de chunks générés ({len(chunks)})"
    )
    logging.info("test_faiss_index_vector_count terminé avec succès.")


def test_faiss_search_returns_correct_metadata():
    """
    Vérifie qu'une recherche Faiss sur le texte d'un chunk renvoie l'ID et la metadata correspondante
    pour ce même chunk.
    """
    logging.info("Démarrage du test : test_faiss_search_returns_correct_metadata")
    # Charge events_clean
    with CLEAN_PATH.open(encoding="utf-8") as f:
        events = json.load(f)
    logging.info("Nombre d'événements pour test de recherche Faiss : %d", len(events))

    # Génère chunks et metadata
    chunks, chunk_meta = chunk_events(events)
    total_chunks = len(chunks)
    logging.info("Nombre total de chunks générés : %d", total_chunks)

    # Sélectionne aléatoirement un chunk
    random_idx = random.randrange(total_chunks)
    chunk_text = chunks[random_idx]
    logging.info("Chunk aléatoire choisi → index=%d, début du texte : '%s...'", random_idx, chunk_text[:50])

    # Embed le chunk_text
    emb = embed_batch(Mistral(api_key=os.getenv("MISTRAL_API_KEY")), chunk_text)
    faiss.normalize_L2(emb.reshape(1, -1))
    logging.info("Embedding du chunk réalisé (forme : %s)", emb.shape)

    # Charge le même index Faiss
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    distances, indices = index.search(emb.reshape(1, -1), 1)
    found_idx = indices[0][0]
    logging.info("Résultat de la recherche Faiss → found_idx=%d, score=%.4f", found_idx, float(distances[0][0]))

    assert found_idx == random_idx, (
        f"Faiss a renvoyé l'index {found_idx} pour la recherche, "
        f"alors que le chunk original avait l'index {random_idx}"
    )

    # Vérifie que la metadata correspond au chunk recherché
    meta = chunk_meta[found_idx]
    logging.info("Metadata associée au chunk trouvé: UID=%s, texte début : '%s...'", meta["uid"], meta["text"][:50])
    assert meta["text"] == chunk_text, (
        "Le texte stocké dans la metadata ne correspond pas au texte du chunk recherché"
    )
    assert "firstdate_begin" in meta and "location_city" in meta, (
        "La metadata ne contient pas les champs attendus"
    )
    logging.info("test_faiss_search_returns_correct_metadata terminé avec succès.")


if __name__ == "__main__":
    pytest.main([__file__])