import streamlit as st
import os
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage
import logging
import datetime

load_dotenv()

# === Configuration du logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# === Paramètres ===
API_KEY    = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = "mistral-large-latest"

# === Fonctions ===
def init_mistral_client(api_key: str) -> Mistral:
    """
    Initialise le client Mistral.
    """
    if not api_key:
        logging.error("La variable d'environnement MISTRAL_API_KEY est manquante")
        raise RuntimeError("MISTRAL_API_KEY is missing")
    logging.info("Initialisation du client Mistral")
    return Mistral(api_key=api_key)

client = init_mistral_client(API_KEY)

# === Initialisation Session State ===
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": (
            "Bonjour et bienvenue sur Puls-Events ! Je suis votre assistant virtuel dédié à la découverte et à la recommandation "
            "d'événements culturels. Vous pouvez me demander des suggestions d'activités par lieu, date ou type d'événement."
        )
    }]

def construire_prompt_session(max_messages: int = 10) -> list[dict]:
    """
    Extrait les derniers messages de st.session_state["messages"],
    les formate au format attendu par Mistral (role + content).
    """
    history = st.session_state["messages"]
    recent = history[-max_messages:] if len(history) > max_messages else history
    return [{"role": msg["role"], "content": msg["content"]} for msg in recent]

def generer_reponse(prompt_messages: list[dict]) -> str:
    """
    Envoie les messages (prompt_messages) à Mistral et renvoie le texte généré.
    """
    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=prompt_messages
        )
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse : {e}")
        return "Je suis désolé, j'ai rencontré un problème. Veuillez réessayer."

# === Interface Streamlit ===

st.title("Assistant Virtuel Puls-Events")

# 1) Affichage de tout l'historique
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 2) Lorsqu’un utilisateur envoie un nouveau message
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # a) On enregistre d'abord son message dans l'historique
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # b) On reconstruit le prompt complet (derniers messages)
    prompt_messages = construire_prompt_session()

    # c) On affiche la bulle "assistant est en train de réfléchir..."
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.text("En train de réfléchir…")

        # d) On génère la réponse
        response = generer_reponse(prompt_messages)

        # e) On remplace l’indicateur par le vrai texte
        placeholder.write(response)

    # f) On ajoute la réponse de l’assistant à l’historique
    st.session_state["messages"].append({"role": "assistant", "content": response})
