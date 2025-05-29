import os
import json
import time
from pathlib import Path
import numpy as np
import faiss
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

# === Paramètres ===
API_KEY       = os.getenv("MISTRAL_API_KEY")
MODEL_NAME    = "mistral-embed"
BATCH_SIZE    = 50
CLEAN_PATH    = Path("data/events_clean.json")
FAISS_PATH    = Path("data/faiss_index.idx")
METADATA_PATH = Path("data/faiss_metadata.json")

def load_cleaned_events(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def init_mistral_client(api_key: str) -> Mistral:
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")
    return Mistral(api_key=api_key)

def embed_batch(client: Mistral, texts: list[str], max_retries: int = 5) -> np.ndarray:
    for attempt in range(1, max_retries+1):
        try:
            resp = client.embeddings.create(model=MODEL_NAME, inputs=texts)
            return np.array([d.embedding for d in resp.data], dtype="float32")
        except SDKError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt
                print(f"⚠️ Rate limit, retry #{attempt} after {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Failed to embed batch after retries")

def build_embeddings_array(client: Mistral, descriptions: list[str]) -> np.ndarray:
    all_embs = []
    for i in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[i: i+BATCH_SIZE]
        embs = embed_batch(client, batch)
        print(f" → Batch {i//BATCH_SIZE+1}: {embs.shape}")
        all_embs.append(embs)
    return np.vstack(all_embs)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_index_and_metadata(index: faiss.Index, metadata: list[dict]) -> None:
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"💾 Index saved to {FAISS_PATH}")
    METADATA_PATH.parent.mkdir(exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved to {METADATA_PATH}")

def test_search(client: Mistral, index: faiss.Index, metadata: list[dict], query: str, k: int = 5):
    print(f"\n🔎 Test search for: “{query}”")
    q_emb = embed_batch(client, [query])[0].astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))
    distances, indices = index.search(q_emb.reshape(1, -1), k)
    for rank, idx in enumerate(indices[0]):
        info = metadata[idx]
        print(f"{rank+1}. uid={info['uid']}  score={distances[0][rank]:.4f}")
        print(f"    Titre   : {info.get('title_fr')}")
        print(f"    Dates   : {info.get('firstdate_begin')} → {info.get('firstdate_end')}")
        print(f"    Adresse : {info.get('location_address')}")
        print(f"    Ville   : {info.get('location_city')}\n")

def main():
    events      = load_cleaned_events(CLEAN_PATH)
    descriptions = [e["description"] for e in events]
    metadata    = [
        {
            "uid":              e["uid"],
            "title_fr":         e.get("title_fr"),
            "firstdate_begin":  e.get("firstdate_begin"),
            "firstdate_end":    e.get("firstdate_end"),
            "location_address": e.get("location_address"),
            "location_city":    e.get("location_city"),
        } 
        for e in events
    ]

    client     = init_mistral_client(API_KEY)
    embeddings = build_embeddings_array(client, descriptions)
    print(f"✅ All embeddings ready: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    print(f"✔ Faiss index created with {index.ntotal} vectors")

    save_index_and_metadata(index, metadata)
    test_search(client, index, metadata, query="Musique Ukraine", k=5)

if __name__ == "__main__":
    main()
