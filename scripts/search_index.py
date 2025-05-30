import os
import json
import time
from pathlib import Path
import numpy as np
import faiss
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
import argparse

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
    Args:
        api_key (str): Clé API Mistral.
    Returns:
        Mistral: Instance du client Mistral.
    """
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")
    return Mistral(api_key=api_key)

def load_index(path: Path) -> faiss.Index:
    """
    Charge un index Faiss depuis un fichier.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return faiss.read_index(str(path))

def load_metadata(path: Path) -> list[dict]:
    """
    Charge les métadonnées correspondantes aux vecteurs depuis un JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))

def embed_query(client: Mistral, query: str) -> np.ndarray:
    """
    Génère l'embedding d'une requête textuelle, en gérant la mise en forme et le retry.
    Args:
        client (Mistral): Client Mistral configuré.
        query (str): Texte de la requête à embedder.
    Returns:
        np.ndarray: Vecteur d'embedding normalisé de forme (dim,).

    """
    for _ in range(3):
        try:
            resp = client.embeddings.create(model=MODEL_NAME, inputs=[query])
            emb = np.array(resp.data[0].embedding, dtype="float32")
            faiss.normalize_L2(emb.reshape(1, -1))
            return emb
        except SDKError as e:
            if e.status_code == 429:
                time.sleep(2)
                continue
            raise
    raise RuntimeError("Failed to embed query")

def search(index: faiss.Index, metadata: list[dict], query: str, k: int = 5):
    """
    Interroge l'index Faiss avec la requête fournie, et affiche les k résultats les plus similaires.
    Args:
        index (faiss.Index): Index Faiss préalablement chargé.
        metadata (list[dict]): Métadonnées alignées avec l'index (uid, chunk_id, etc.).
        query (str): Texte de la requête.
        k (int): Nombre de résultats à afficher.
    """
    client = init_mistral_client(API_KEY)
    q_emb = embed_query(client, query)
    D, I = index.search(q_emb.reshape(1, -1), k)
    for rank, idx in enumerate(I[0]):
        m = metadata[idx]
        snippet = m["text"][:200].replace("\n", " ")
        print(f"{rank+1}. uid={m['uid']} chunk={m['chunk_id']} score={D[0][rank]:.4f}")
        print(f"    Title: {m['title_fr']}")
        print(f"    Dates: {m['firstdate_begin']} → {m['firstdate_end']}")
        print(f"    Location: {m['location_address']}, {m['location_city']}")
        print(f"    Snippet: {snippet}…\n")

# === Exécution principale ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query")
    parser.add_argument("-k",      type=int, default=DEFAULT_K,  help="Number of results")
    args = parser.parse_args()

    idx      = load_index(FAISS_PATH)
    metadata = load_metadata(METADATA_PATH)
    search(idx, metadata, args.query, k=args.k)

