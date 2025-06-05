import streamlit as st
import os
import json
from pathlib import Path
import faiss
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage
from mistralai.models.sdkerror import SDKError
import logging
import time
import datetime
import numpy as np

load_dotenv()

# === Configuration du logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# === PARAMÈTRES GLOBAUX ===
API_KEY             = os.getenv("MISTRAL_API_KEY")
MODEL_NAME          = "mistral-large-latest"
EMBEDDING_MODEL     = "mistral-embed"
FAISS_INDEX_PATH    = Path("data/faiss_index.idx")
FAISS_METADATA_PATH = Path("data/faiss_metadata.json")

# Nombre de chunks à récupérer pour construire le contexte RAG
TOP_K_CHUNKS    = 5

# === INITIALISATION CLIENT MISTRAL ===
def init_mistral_client(api_key: str) -> Mistral:
    """
    Initialise le client Mistral pour la génération d'embeddings et de chat.
    """
    if not api_key:
        logging.error("La variable d'environnement MISTRAL_API_KEY est manquante.")
        raise RuntimeError("MISTRAL_API_KEY is missing")
    logging.info("Initialisation du client Mistral")
    return Mistral(api_key=api_key)

client = init_mistral_client(API_KEY)


# === CHARGEMENT DE L'INDEX FAISS ET DES MÉTADONNÉES ===
def load_index(path: Path) -> faiss.Index:
    """
    Charge un index Faiss depuis un fichier sur disque.
    """
    logging.info("Chargement de l'index Faiss depuis %s …", path)
    if not path.exists():
        logging.error("Fichier d'index introuvable : %s", path)
        raise FileNotFoundError(f"{path} not found")
    index = faiss.read_index(str(path))
    logging.info("Index Faiss chargé (%d vecteurs).", index.ntotal)
    return index

def load_metadata(path: Path) -> list[dict]:
    """
    Charge les métadonnées alignées avec les vecteurs (UID, texte du chunk, etc.).
    """
    logging.info("Chargement des métadonnées depuis %s …", path)
    if not path.exists():
        logging.error("Fichier de métadonnées introuvable : %s", path)
        raise FileNotFoundError(f"{path} not found")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    logging.info("%d entrées de métadonnées chargées.", len(metadata))
    return metadata

# On charge l’index et les métadonnées une seule fois au démarrage
faiss_index   = load_index(FAISS_INDEX_PATH)
faiss_metadata = load_metadata(FAISS_METADATA_PATH)

# === EMBEDDING DE LA REQUÊTE UTILISATEUR ===
def embed_query(client: Mistral, query: str, max_retries: int = 3) -> np.ndarray:
    """
    Génère l'embedding Mistral pour une requête textuelle, en gérant la mise en forme et les retries en cas de rate limit.
    Renvoie un vecteur numpy float32 normalisé.
    """
    logging.info("Génération d'embedding pour la requête : «%s»", query)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, inputs=[query])
            emb = np.array(resp.data[0].embedding, dtype="float32")
            # Normalisation L2 pour similarité cosinus
            faiss.normalize_L2(emb.reshape(1, -1))
            logging.info("Embedding généré (dimension=%d).", emb.shape[0])
            return emb
        except SDKError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt
                logging.warning("Rate limit (429). Réessai #%d dans %d s …", attempt, wait)
                time.sleep(wait)
                continue
            logging.error("Erreur Mistral lors de l'embedding de la requête : %s", e)
            raise
    logging.critical("Échec de la génération d'embedding après %d tentatives.", max_retries)
    raise RuntimeError("Failed to embed query")

# === FONCTION DE RECHERCHE RAG AVEC FAISS ===
def retrieve_top_k_chunks(query: str, k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    1. Embed la requête utilisateur.
    2. Recherche dans l'index Faiss les k vecteurs les plus similaires.
    3. Retourne la liste des métadonnées correspondantes (top k) pour construire le contexte.
    """
    q_emb = embed_query(client, query)
    distances, indices = faiss_index.search(q_emb.reshape(1, -1), k)
    logging.info("Recherche FAISS effectuée (top %d).", k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(faiss_metadata):
            continue
        meta = faiss_metadata[idx]
        results.append({
            "uid": idx,  # ou meta["uid"] si l’index sur UID diffère
            "chunk_id": meta.get("chunk_id"),
            "text": meta.get("text", ""),
            "title_fr": meta.get("title_fr"),
            "firstdate_begin": meta.get("firstdate_begin"),
            "firstdate_end": meta.get("firstdate_end"),
            "location_address": meta.get("location_address"),
            "location_city": meta.get("location_city"),
            "distance": float(distances[0][rank])
        })
    return results


# === INITIALISATION DE L’HISTORIQUE DE CONVERSATION (SESSION STATE) ===
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Bonjour et bienvenue sur Puls-Events ! Je suis votre assistant virtuel dédié à la découverte et à la recommandation "
                "d'événements culturels. Vous pouvez me demander des suggestions d'activités par lieu, date ou type d'événement."
            )
        }
    ]

def construire_prompt_session(max_messages: int = 10) -> list[dict]:
    """
    Extrait les derniers messages de l'historique (session_state["messages"]) 
    et renvoie une liste de dicts au format {'role': ..., 'content': ...}.
    """
    history = st.session_state["messages"]
    recent = history[-max_messages:] if len(history) > max_messages else history
    return [{"role": msg["role"], "content": msg["content"]} for msg in recent]

# === GÉNÉRATION DE LA RÉPONSE AVEC CONTEXTE RAG ===
def generer_reponse_rag(user_query: str) -> str:
    """
    1. Récupère les top K chunks pertinents depuis FAISS.
    2. Construit un « system prompt » contenant les passages récupérés.
    3. Rajoute l'historique des messages ainsi que la requête utilisateur.
    4. Appelle Mistral en mode chat pour générer la réponse.
    """
    # 1) Récupération des chunks FAISS
    top_chunks = retrieve_top_k_chunks(user_query, k=TOP_K_CHUNKS)
    if not top_chunks:
        context_text = "Aucun contexte pertinent trouvé dans la base d'événements."
        logging.warning("Aucun chunk FAISS pertinent pour la requête : «%s»", user_query)
    else:
        # On combine simplement le texte de chaque chunk, en introduisant des séparateurs
        context_pieces = []
        for i, chunk in enumerate(top_chunks, start=1):
            piece = (
                f"--- Contexte {i} (score={chunk['distance']:.3f}) ---\n"
                f"{chunk['text']}"
            )
            context_pieces.append(piece)
        context_text = "\n\n".join(context_pieces)

    # 2) On construit le message système RAG
    system_content = (
        "Vous êtes un assistant spécialisé dans la recommandation d'événements culturels. "
        "Utilisez STRICTEMENT les extraits ci-dessous (issus de la base de données d'événements) "
        "pour répondre à la question suivante : « " + user_query + " »\n\n"
        "INFORMATIONS RETROUVÉES :\n"
        f"{context_text}\n\n"
        "Si l'information demandée n'est pas présente dans ces extraits, répondez honnêtement que vous ne la connaissez pas."
    )

    system_message = {"role": "system", "content": system_content}
    logging.info("Message système RAG construit (longueur=%d).", len(system_content))

    # 3) On assemble la conversation complète pour l’appel chat
    prompt_history = construire_prompt_session()
    # On place le message système en tête, puis l’historique
    full_prompt = [system_message] + prompt_history

    # 4) On appelle Mistral pour générer la réponse
    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=full_prompt
        )
        answer = response.choices[0].message.content
        logging.info("Réponse générée par Mistral (longueur=%d).", len(answer))
        return answer

    except Exception as e:
        logging.error("Erreur lors de l’appel à Mistral Chat : %s", e)
        return "Désolé, une erreur interne est survenue. Veuillez réessayer."
    

# === INTERFACE STREAMLIT ===

# 1) Afficher tout l’historique (chat bubbles)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2) Gérer la saisie de l’utilisateur
if user_input := st.chat_input("Comment puis-je vous aider ?"):
    # a) On mémorise d’abord la question de l’utilisateur
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # b) On affiche la bulle « assistant réfléchit… »
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.text("Recherche dans la base de données et réflexion…")

        # c) On génère la réponse RAG
        answer = generer_reponse_rag(user_input)

        # d) On remplace l’indicateur par la réponse finale
        placeholder.write(answer)

    # e) On ajoute la réponse à l’historique
    st.session_state["messages"].append({"role": "assistant", "content": answer})
