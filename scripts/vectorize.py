import os
import logging
import json
import time
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import faiss
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# === Configuration du logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s")

# === Paramètres ===

API_KEY        = os.getenv("MISTRAL_API_KEY")
MODEL_NAME     = "mistral-embed"
BATCH_SIZE     = 50
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
CLEAN_PATH     = Path("data/events_clean.json")
FAISS_PATH     = Path("data/faiss_index.idx")
METADATA_PATH  = Path("data/faiss_metadata.json")

# === Fonctions ===

def load_cleaned_events(path: Path) -> list[dict]:
    """
    Charge les événements nettoyés depuis un fichier JSON.
    """
    logging.info("Chargement des événements nettoyés depuis %s", path)
    if not path.exists():
        logging.error("Fichier introuvable : %s", path)
        raise FileNotFoundError(f"{path} not found")
    with path.open(encoding="utf-8") as f:
        events = json.load(f)
    logging.info("%d événements chargés", len(events))
    return events

def init_mistral_client(api_key: str) -> Mistral:
    """
    Initialise le client Mistral pour la génération d'embeddings.
    """
    if not api_key:
        logging.error("La variable d'environnement MISTRAL_API_KEY est manquante")
        raise RuntimeError("MISTRAL_API_KEY is missing")
    logging.info("Initialisation du client Mistral")
    return Mistral(api_key=api_key)


def embed_batch(client: Mistral, texts: list[str], max_retries: int = 5) -> np.ndarray:
    """
    Génère les embeddings pour un lot de textes via l'API Mistral, avec retry en cas de rate limit.
    Args:
        client (Mistral): Client Mistral configuré.
        texts (list[str]): Liste de textes à vectoriser.
        max_retries (int): Nombre maximal de tentatives en cas de 429.
    Returns:
        np.ndarray: Tableau d'embeddings de forme (len(texts), dim).
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(model=MODEL_NAME, inputs=texts)
            embeddings = np.array([d.embedding for d in resp.data], dtype="float32")
            logging.info("Embeddings générés pour un lot de %d textes", len(texts))
            return embeddings
        except SDKError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt
                logging.warning("Limite de débit atteinte, nouvelle tentative #%d dans %ds", attempt, wait)
                time.sleep(wait)
                continue
            logging.error("Échec de la génération des embeddings : %s", e)
            raise
    logging.critical("Nombre maximal de tentatives épuisé pour l'embedding")
    raise RuntimeError("Failed to embed batch after retries")

def chunk_events(events: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Découpe chaque description d'événement en chunks de taille fixe avec chevauchement.
    Args:
        events (list[dict]): Liste d'événements nettoyés.
    Returns:
        tuple:
            - all_chunks (list[str]): Tous les morceaux de texte.
            - chunk_meta (list[dict]): Métadonnées associées à chaque chunk
               (uid, chunk_id, texte, titre, dates, lieu, etc.).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    all_chunks, chunk_meta = [], []
    for ev in events:
        text = ev.get("description", "")
        # split_text renvoie une liste de strings
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            # on conserve uid + contexte d'origine
            chunk_meta.append({
                "uid":               ev["uid"],
                "chunk_id":          i,
                "text":              chunk,
                "title_fr":          ev.get("title_fr"),
                "firstdate_begin":   ev.get("firstdate_begin"),
                "firstdate_end":     ev.get("firstdate_end"),
                "location_address":  ev.get("location_address"),
                "location_city":     ev.get("location_city"),
            })
    logging.info("%d segments créés à partir de %d événements", len(all_chunks), len(events))
    return all_chunks, chunk_meta

def build_embeddings_array(client: Mistral, chunks: list[str]) -> np.ndarray:
    """
    Génère les embeddings pour tous les chunks par lots.
    Args:
        client (Mistral): Client Mistral configuré.
        chunks (list[str]): Liste de textes chunkés.
    Returns:
        np.ndarray: Matrice d'embeddings de forme (n_chunks, dim).
    """
    logging.info("Vectorisation de %d segments en lots de %d", len(chunks), BATCH_SIZE)
    all_embs = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embs = embed_batch(client, batch)
        logging.info("Lot %d/%d vectorisé (shape=%s)", i // BATCH_SIZE + 1, total_batches, embs.shape)
        all_embs.append(embs)
    embeddings = np.vstack(all_embs)
    logging.info("Tous les embeddings empilés : %s", embeddings.shape)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Construit un index Faiss pour la recherche par similarité cosinus.
    Args:
        embeddings (np.ndarray): Matrice d'embeddings normalisés ou non.
    Returns:
        faiss.Index: Index FlatIP prêt pour la recherche.
    """
    logging.info("Normalisation des embeddings pour la similarité cosinus")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logging.info("Index Faiss créé avec %d vecteurs (dimension=%d)", index.ntotal, dim)
    return index

def save_index_and_metadata(index: faiss.Index, metadata: list[dict]) -> None:
    """
    Sauvegarde l'index Faiss et les métadonnées associées sur le disque.
    Args:
        index (faiss.Index): Index Faiss à sauvegarder.
        metadata (list[dict]): Liste des métadonnées alignées avec l'index.
    """
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    logging.info("Index Faiss enregistré dans %s", FAISS_PATH)
    METADATA_PATH.parent.mkdir(exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logging.info("Métadonnées enregistrées dans %s", METADATA_PATH)

# === Exécution principale ===

def main():
    """
    Pipeline complet : chargement des événements, chunking, embeddings, indexation et sauvegarde.
    """
    logging.info("=== DÉMARRAGE DU PIPELINE DE VECTORIZATION ===")
    events = load_cleaned_events(CLEAN_PATH)
    client = init_mistral_client(API_KEY)
    # 1) chunking
    chunks, chunk_meta = chunk_events(events)
    # 2) embeddings
    embeddings = build_embeddings_array(client, chunks)
    # 3) index Faiss
    index = build_faiss_index(embeddings)
    # 4) save
    save_index_and_metadata(index, chunk_meta)
    logging.info("=== PIPELINE DE VECTORIZATION TERMINÉ ===")

if __name__ == "__main__":
    main()
