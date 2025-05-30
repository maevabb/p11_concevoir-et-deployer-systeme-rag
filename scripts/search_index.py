import os
import logging
import json
import time
from pathlib import Path

import numpy as np
import faiss
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
import argparse

# === Configuration du logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s")

# === Paramètres ===

API_KEY        = os.getenv("MISTRAL_API_KEY")
MODEL_NAME     = "mistral-embed"
FAISS_PATH     = Path("data/faiss_index.idx")
METADATA_PATH  = Path("data/faiss_metadata.json")

# Requête par défaut
DEFAULT_QUERY = "Musique Ukraine"
DEFAULT_K     = 5

# === Fonctions ===

def init_mistral_client(api_key: str) -> Mistral:
    """
    Initialise le client Mistral pour la génération d'embeddings.
    """
    if not api_key:
        logging.error("La variable d'environnement MISTRAL_API_KEY est manquante")
        raise RuntimeError("MISTRAL_API_KEY is missing")
    logging.info("Initialisation du client Mistral")
    return Mistral(api_key=api_key)

def load_index(path: Path) -> faiss.Index:
    """
    Charge un index Faiss depuis un fichier.
    """
    logging.info("Chargement de l'index Faiss depuis %s", path)
    if not path.exists():
        logging.error("Fichier d'index introuvable : %s", path)
        raise FileNotFoundError(f"{path} not found")
    index = faiss.read_index(str(path))
    logging.info("Index chargé (%d vecteurs)", index.ntotal)
    return index

def load_metadata(path: Path) -> list[dict]:
    """
    Charge les métadonnées correspondantes aux vecteurs depuis un JSON.
    """
    logging.info("Chargement des métadonnées depuis %s", path)
    if not path.exists():
        logging.error("Fichier de métadonnées introuvable : %s", path)
        raise FileNotFoundError(f"{path} not found")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    logging.info("%d entrées de métadonnées chargées", len(metadata))
    return metadata

def embed_query(client: Mistral, query: str) -> np.ndarray:
    """
    Génère l'embedding d'une requête textuelle, en gérant la mise en forme et le retry.
    """
    logging.info("Génération de l'embedding pour la requête : « %s »", query)
    for attempt in range(1, 4):
        try:
            resp = client.embeddings.create(model=MODEL_NAME, inputs=[query])
            emb = np.array(resp.data[0].embedding, dtype="float32")
            faiss.normalize_L2(emb.reshape(1, -1))
            logging.info("Embedding généré (dimension=%d)", emb.shape[0])
            return emb
        except SDKError as e:
            if e.status_code == 429:
                wait = 2 ** attempt
                logging.warning("Rate limit (429), nouvelle tentative #%d dans %ds", attempt, wait)
                time.sleep(wait)
                continue
            logging.error("Erreur Mistral lors de l'embedding de la requête: %s", e)
            raise
    logging.critical("Échec de la génération d'embedding après 3 tentatives")
    raise RuntimeError("Failed to embed query")

def search(index: faiss.Index, metadata: list[dict], query: str, k: int = 5):
    """
    Interroge l'index Faiss avec la requête fournie, et affiche les k résultats les plus similaires.
    """
    client = init_mistral_client(API_KEY)
    q_emb = embed_query(client, query)

    logging.info("Recherche des %d voisins les plus proches", k)
    distances, indices = index.search(q_emb.reshape(1, -1), k)

    for rank, idx in enumerate(indices[0]):
        m = metadata[idx]
        snippet = m["text"][:200].replace("\n", " ")
        logging.info(
            "%d. uid=%s chunk=%d score=%.4f",
            rank+1, m['uid'], m['chunk_id'], distances[0][rank]
        )
        logging.info("    Titre  : %s", m.get('title_fr'))
        logging.info("    Dates  : %s → %s", m.get('firstdate_begin'), m.get('firstdate_end'))
        logging.info("    Lieu   : %s, %s", m.get('location_address'), m.get('location_city'))
        logging.info("    Extrait: %s…", snippet)

# === Exécution principale ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrogation de l'index vectoriel Faiss")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Requête de recherche")
    parser.add_argument("-k",      type=int, default=DEFAULT_K,  help="Nombre de résultats à retourner")
    args = parser.parse_args()

    idx      = load_index(FAISS_PATH)
    metadata = load_metadata(METADATA_PATH)
    search(idx, metadata, args.query, k=args.k)

