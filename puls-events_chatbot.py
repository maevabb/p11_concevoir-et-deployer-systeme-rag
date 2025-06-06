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

# On charge l'index et les métadonnées une seule fois au démarrage
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
    3. Retourne la liste des métadonnées correspondantes (top k) pour construire le contexte,
       en loggant chaque résultat pour vérification.
    """
    q_emb = embed_query(client, query)
    distances, indices = faiss_index.search(q_emb.reshape(1, -1), k)
    logging.info("Recherche FAISS effectuée (top %d).", k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(faiss_metadata):
            continue
        m = faiss_metadata[idx]

        # On logge ici les informations de chaque chunk retourné
        snippet = m.get("text", "").replace("\n", " ")[:100]  # On limite à 100 caractères pour le log
        logging.info(
            "%d. uid=%s chunk=%d score=%.4f",
            rank + 1, m["uid"], m["chunk_id"], float(distances[0][rank])
        )
        logging.info("    Titre  : %s", m.get("title_fr"))
        logging.info("    Dates  : %s → %s", m.get("firstdate_begin"), m.get("firstdate_end"))
        logging.info("    Lieu   : %s, %s", m.get("location_address"), m.get("location_city"))
        logging.info("    Extrait: %s…", snippet)

        results.append({
            "uid":              m["uid"],
            "chunk_id":         m["chunk_id"],
            "text":             m.get("text", ""),
            "title_fr":         m.get("title_fr"),
            "firstdate_begin":  m.get("firstdate_begin"),
            "firstdate_end":    m.get("firstdate_end"),
            "location_address": m.get("location_address"),
            "location_city":    m.get("location_city"),
            "distance":         float(distances[0][rank])
        })
    return results


# === INITIALISATION DE L'HISTORIQUE DE CONVERSATION (SESSION STATE) ===
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
def construire_system_message_rag(user_query: str, top_chunks: list[dict]) -> dict:
    """
    Construit le 'system_message' complet en respectant les recommandations Puls-EventsBot.
    - user_query : la question de l'utilisateur.
    - top_chunks : liste des k meilleurs chunks renvoyés par FAISS, chacun contenant 'text', 'distance', 'title_fr', 'firstdate_begin', 'firstdate_end', 'location_address', 'location_city', etc.
    """

    # 1) On prépare le contexte issu de FAISS (les passages/chunks)
    if not top_chunks:
        context_text = "Aucun contexte pertinent trouvé dans la base d'événements."
    else:
        fragments = []
        for i, chunk in enumerate(top_chunks, start=1):
            # On lit les métadonnées qui ont été chargées dans faiss_metadata
            titre     = chunk.get("title_fr", "Titre inconnu")
            date      = chunk.get("firstdate_begin", "Date inconnue")
            lieu      = chunk.get("location_city", "Lieu inconnu")
            descriptif = chunk.get("text", "")

            fragments.append(
                f"--- Contexte {i} (score FAISS={chunk['distance']:.3f}) ---\n"
                f"Titre      : {titre}\n"
                f"Date       : {date}\n"
                f"Lieu       : {lieu}\n"
                f"Descriptif : {descriptif}"
            )

        context_text = "\n\n".join(fragments)

    # 2) On rédige le prompt système en respectant toutes les consignes
    system_content = f"""
Vous êtes Puls-EventsBot, un assistant virtuel spécialisé dans la recommandation d'événements culturels en temps réel.

RÔLE DE L'ASSISTANT
• Identité : Puls-EventsBot, expert en événements culturels (concerts, expositions, festivals, spectacles, etc.) collectés via Open Agenda.
• Autorité : Vous êtes habilité à fournir uniquement des renseignements factuels sur les événements indexés (titre, date, lieu, bref descriptif),
  et à proposer des suggestions personnalisées en fonction des critères de l'utilisateur (ville, période, type d'événement).
• Périmètre :
  - Recommandations d'événements situés dans la zone géographique couverte (par exemple : Paris et sa région), datant de moins d'un an.
  - Consultation de la base Faiss pour récupérer des « passages » (chunks) extraits des descriptions d'événements.
  - Génération de réponses exclusivement à partir des passages retournés par l'index Faiss.

SOURCES D'INFORMATION AUTORISÉES
1. Base vectorielle FAISS :
   - Chaque requête utilisateur génère un embedding via Mistral, taggué pour similarité.
   - Les « chunks » obtenus sont des segments textuels extraits de la colonne description (titre + résumé + détails nettoyés) des événements Open Agenda.
   - Les métadonnées associées (uid, titre, date, lieu) sont stockées au format JSON.
2. Modèle Mistral (mistral-embed + mistral-chat) :
   - Utilisé pour transformer les textes en vecteurs et pour formuler les réponses finales.
   - Vous devez impérativement vous appuyer uniquement sur le contenu des passages fournis par Faiss.

Vous ne devez PAS :
• Inventer d'événements, de dates ou de lieux qui ne figurent pas dans la base.
• Citer une source non indexée ou lier vers un site extérieur.
• Fournir des informations promotionnelles sur des événements non présents dans notre base.

COMPORTEMENTS OBLIGATOIRES
1. Mode de réponse :
   - Réponse factuelle et concise : chaque événement présenté doit comporter au minimum son titre, sa date, son lieu, et un court descriptif (extrait du chunk).
   - Si plusieurs événements correspondent, classez-les par ordre de pertinence (score FAISS décroissant).
   - Proposez au moins 3 à 5 suggestions lorsque la requête est suffisamment précise.
   - Si l'utilisateur précise un filtre (ex. « concert jazz à Lyon »), ne renvoyez que des événements répondant exactement à ces critères.
2. Gestion des ambiguïtés :
   - Si la requête est trop vague (ex. « Je veux aller à un spectacle »), demandez une précision (type d'événement, localisation, période).
     Exemple : « Pouvez-vous préciser si vous cherchez un concert, une pièce de théâtre ou une exposition, et dans quelle ville ou période ? »
3. En cas d'absence de résultat :
   - Si aucune entrée pertinente n'existe dans la base vectorielle, informez clairement l'utilisateur :
     « Désolé, je n'ai pas trouvé d'événement correspondant à cette recherche. Voulez-vous élargir la période ou changer de type d'événement ? »
   - Proposez toujours une alternative (ex. « Vous pourriez consulter notre page d'accueil pour découvrir tous les événements du mois » ou
     « Élargissez votre recherche à une autre ville »).
4. Structure de la réponse :
   - Introduction courte et polie (ex. « Voici ce que j'ai trouvé pour … »).
   - Liste numérotée ou à puces des événements :
       1. Titre : …
       2. Date : …
       3. Lieu : …
       4. Descriptif : …
   - Conclusion avec invite ou suggestion (ex. « Si vous souhaitez d'autres recommandations, n'hésitez pas à préciser vos critères »).
5. Ton et style :
   - Chaleureux mais professionnel : langue accessible à tous, sans jargon technique.
   - Courtois et aidant : montrez-vous patient·e et encourageant·e.

COMPORTEMENTS INTERDITS
• Ne jamais halluciner :
  - N'ajoutez pas de détails (prix, réservation, playlist, etc.) qui ne proviennent pas explicitement du chunk retourné.
  - Si vous n'êtes pas certain d'un élément, mentionnez « information non disponible dans nos données actuelles ».
• Ne pas dépasser le périmètre des événements culturels :
  - Interrogez uniquement la base Faiss d'événements indexés.
  - Évitez toute recommandation de services extérieurs (restaurants, parkings, etc.), sauf si l'utilisateur le demande explicitement.
• Ne pas communiquer de données privées :
  - N'incluez jamais d'informations personnelles ou sensibles (numéros de téléphone, adresse mail, données personnelles des organisateurs).

INFORMATIONS RETROUVÉES :
{context_text}

"""
    return {"role": "system", "content": system_content}


def generer_reponse_rag(user_query: str) -> str:
    """
    1. Récupère les top K chunks pertinents depuis FAISS.
    2. Construit le system_message complet contenant les passages récupérés.
    3. Rajoute l'historique des messages ainsi que la requête utilisateur.
    4. Appelle Mistral en mode chat pour générer la réponse.
    """
    # 1) Récupération des chunks FAISS
    top_chunks = retrieve_top_k_chunks(user_query, k=TOP_K_CHUNKS)

    # 2) Construction du system_message avec contexte + recommandations
    system_message = construire_system_message_rag(user_query, top_chunks)

    # 3) Assembler la conversation : on place le system_message en tête
    prompt_history = construire_prompt_session()
    full_prompt = [system_message] + prompt_history

    # 4) On appelle Mistral pour générer la réponse
    try:
        # Pour minimiser les hallucinations, on fixe temperature à 0.0 et top_p à 1.0
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=full_prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        logging.info("Réponse générée par Mistral (longueur=%d).", len(answer))
        return answer

    except Exception as e:
        logging.error("Erreur lors de l'appel à Mistral Chat : %s", e)
        return "Désolé, une erreur interne est survenue. Veuillez réessayer."

# === INTERFACE STREAMLIT ===

st.title("Assistant Virtuel Puls-Events")

# 1) Afficher tout l'historique (chat bubbles)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2) Gérer la saisie de l'utilisateur
if user_input := st.chat_input("Comment puis-je vous aider ?"):
    # a) On mémorise d'abord la question de l'utilisateur
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # b) On affiche la bulle « assistant réfléchit… »
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.text("Recherche dans la base de données et réflexion…")

        # c) On génère la réponse RAG
        answer = generer_reponse_rag(user_input)

        # d) On remplace l'indicateur par la réponse finale
        placeholder.write(answer)

    # e) On ajoute la réponse à l'historique
    st.session_state["messages"].append({"role": "assistant", "content": answer})
