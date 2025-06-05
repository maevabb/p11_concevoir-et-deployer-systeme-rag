import streamlit as st
import os
from mistralai import Mistral, UserMessage
import logging
import datetime

# === Configuration du logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# === Paramètres ===
API_KEY        = os.getenv("MISTRAL_API_KEY")
MODEL_NAME     = "mistral-embed"

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