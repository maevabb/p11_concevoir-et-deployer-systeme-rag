import os
import json
import numpy as np
import faiss
from pathlib import Path
from mistralai import Mistral
import time
from mistralai.models.sdkerror import SDKError

# === Paramètres ===
API_KEY = "oNWTwHqZt2UjVWsnVUyLwJevo6OWZfXR"
MODEL_NAME = "mistral-embed"
BATCH_SIZE = 50              # nombre de textes par appel API
CLEAN_PATH = Path("data/events_clean.json")
FAISS_PATH = Path("data/faiss_index.idx")
METADATA_PATH = Path("data/faiss_metadata.json")

# 1) Vérifications
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY non défini dans l'environnement")

# 2) Charge les données nettoyées
with CLEAN_PATH.open(encoding="utf-8") as f:
    events = json.load(f)

texts = [e["description_fr"] or "" for e in events]
uids  = [e["uid"] for e in events]

print(f"🔍 {len(texts)} descriptions à vectoriser (modèle={MODEL_NAME})")

# 3) Instancie le client
client = Mistral(api_key=API_KEY)

def embed_batch(batch: list[str], max_retries: int = 5) -> np.ndarray:
    """
    Renvoie un array float32 (batch_size, dim) pour la liste de batch,
    avec retry en cas de 429 (rate limit).
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(
                model=MODEL_NAME,
                inputs=batch,
            )
            embeddings = [item.embedding for item in resp.data]
            return np.array(embeddings, dtype="float32")

        except SDKError as e:
            # 429 rate limit
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt  # 2, 4, 8, 16, ...
                print(f"⚠️  Rate limit (429), retry #{attempt} après {wait}s…")
                time.sleep(wait)
                continue
            # on ne connaît pas le code, ou on a épuisé les retries
            print(f"❌ Erreur Mistral (status={e.status_code}): {e}. Abandon.")
            raise

        except Exception as e:
            print(f"❌ Erreur inattendue lors de l’embed (lot): {e}")
            raise

    # Si jamais on boucle sans avoir retourné :
    raise RuntimeError("Échec de la génération d'embeddings après retries.")

# 4 Générer tous les embeddings en batch
all_embs = []
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i : i + BATCH_SIZE]
    embs = embed_batch(batch_texts)
    all_embs.append(embs)
    print(f"  → Batch {i//BATCH_SIZE+1} : {embs.shape}")

embeddings = np.vstack(all_embs).astype("float32")  # shape = (2074, 1024)
print(f"✅ Tous les embeddings sont prêts : {embeddings.shape}")

# 5 Normaliser pour similarité cosinus
faiss.normalize_L2(embeddings)

# 6 Créer et remplir l’index Faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"✔ Index Faiss créé avec {index.ntotal} vecteurs.")

# 7 Sauvegarde de l’index et des métadonnées
FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(FAISS_PATH))
print(f"💾 Index sauvegardé dans {FAISS_PATH}")

# On sauve aussi les UIDs pour retrouver l’événement d’origine
meta = [{"uid": u} for u in uids]
with METADATA_PATH.open("w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"💾 Métadonnées sauvegardées dans {METADATA_PATH}")