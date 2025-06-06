# Puls-Events RAG Chatbot POC
✨ Author: Maëva Beauvillain 📅 Start Date: May 2025 📅  Last Updated: June 6, 2025

Un Proof of Concept (POC) d’un chatbot de recommandation d’événements culturels, basé sur un système Retrieval-Augmented Generation (RAG) utilisant Faiss pour l’indexation vectorielle et Mistral pour la génération de texte. Ce projet inclut l’extraction, le nettoyage et la vectorisation des données Open Agenda, ainsi qu’un chatbot Streamlit exposant les recommandations en temps réel.

---

## Table des matières

1. [Contexte & Objectifs](#contexte--objectifs)  
2. [Prérequis](#prérequis)  
3. [Installation](#installation)  
4. [Arborescence du projet](#arborescence-du-projet)  
5. [Description des scripts](#description-des-scripts)  
   - [1. fetch_openagenda.py](#1-fetch_openagendapy)  
   - [2. clean_data.py](#2-clean_datapy)  
   - [3. vectorize.py](#3-vectorizepy)  
   - [4. search_index.py](#4-search_indexpy)  
   - [5. puls_events_chatbot.py](#5-puls_events_chatbotpy)  
   - [6. run_pipeline.py](#6-run_pipelinepy)  
6. [Exécution du pipeline complet](#exécution-du-pipeline-complet)  
7. [Tests unitaires](#tests-unitaires)  
8. [Fichier `.env`](#fichier-env)  
9. [Paramètres techniques](#paramètres-techniques)  

---

## 1. Contexte & Objectifs

Puls-Events souhaite tester une version fonctionnelle d’un POC RAG : un chatbot intelligent, capable de recommander des événements culturels récents (moins d’un an) de l’Île-de-France, en exploitant une base vectorielle Faiss alimentée par les données publiques d’Open Agenda.  

Le POC comprend :
- Un **script d’extraction** (`fetch_openagenda.py`) pour récupérer les événements Open Agenda filtrés par région et dates.
- Un **script de nettoyage** (`clean_data.py`) pour normaliser, enrichir (titre, dates, ville) et formater les données.
- Un **script de vectorisation** (`vectorize.py`) qui découpe les descriptions en chunks, génère des embeddings Mistral, construit un index Faiss et sauvegarde les métadonnées.
- Un **bot Streamlit** (`puls_events_chatbot.py`) qui orchestre la logique RAG : recherche Faiss, construction du prompt système enrichi du contexte, appel au modèle Mistral-chat pour générer la réponse, et affichage en interface chat.
- Un **script de tests** (`tests/test_rag_pipeline.py`) qui vérifie que :
  - Tous les événements nettoyés sont situés à ±1 an de la date courante.  
  - Tous les événements proviennent de la région Île-de-France.  
  - Le nombre de chunks générés correspond à la taille du metadata JSON.  
  - L’index Faiss contient autant de vecteurs que de chunks.  
  - Une recherche Faiss sur un chunk donné renvoie bien sa métadonnée associée.  

---

## 2. Prérequis
- **Python 3.10 +**  
- **Poetry** (gestionaire de dépendances recommandé)
- Clé d’API Mistral (fichier `.env` requis)  
- Accès internet pour appeler l’API Open Agenda et l’API Mistral

---

## 3. Installation
1. Clonez ce dépôt :  
```bash
git clone https://github.com/maevabb/p11_concevoir-et-deployer-systeme-rag
cd p11_concevoir-et-deployer-systeme-rag
```

2. Installez les dépendances avec Poetry :
```bash
poetry install
```

3. Créez un fichier `.env` à la racine (voir section [Fichier .env](#fichier-env))

---

## 4. Arborescence du projet
```markdown
C:.
│   .env
│   .gitignore
│   poetry.lock
│   pyproject.toml
│   pytest.ini
│   README.md
│   puls_events_chatbot.py
│   run_pipeline.py
│
├───data
│       events_raw.json
│       events_clean.json
│       faiss_index.idx
│       faiss_metadata.json
│
├───scripts
│   │   fetch_openagenda.py
│   │   clean_data.py
│   │   vectorize.py
│   │   search_index.py
│   │   __init__.py
│
├───tests
│   │   test_rag_pipeline.py
│   │   test_fetch_openagenda.py
│   │   test_clean_data.py
```
---

## 5. Description des scripts
### 1. `scripts/fetch_openagenda.py`
#### But :  
Récupérer les événements publics Open Agenda filtrés sur la région « Île-de-France » et la période ± 1 an.

### 2. `scripts/clean_data.py`

#### But : 
Prendre `data/events_raw.json`, supprimer doublons et enregistrements invalides, concaténer et formater les champs pour produire `data/events_clean.json` avec tous les événements publics d’Île-de-France (±1 an).

#### Étapes :

- Supprime les doublons sur la colonne `uid`.
- Supprime les colonnes 100 % NaN.
- Supprime les enregistrements sans `title_fr` ni `description_fr`.
-  Crée un champ `description` incluant :
  1. **Titre**  
  2. **Dates** (`firstdate_begin → firstdate_end`) et **Ville**  
  3. **Résumé**  
  4. **Détails** (HTML nettoyé)
- Conserve uniquement les colonnes listées dans `SELECT_COLUMNS`.
- Reformate les dates (`firstdate_begin` et `firstdate_end`) en `YYYY-MM-DD HH:MM`.
- Sauvegarde le DataFrame nettoyé au format JSON indenté dans `data/events_clean.json`.

### 3. `scripts/vectorize.py`

#### But :  
Découper chaque description en chunks, calculer les embeddings Mistral pour chaque chunk, construire un index Faiss et sauvegarder l’index + métadonnées dans `data/`.

#### Étape :
- Découpe chaque description en chunks (1000 caractères + chevauchement 200)
- Génère embeddings avec mistral-embed
- Normalise (L2) et crée un index Faiss FlatIP,
- Écrit data/faiss_index.idx et data/faiss_metadata.json.

Les logs (logging.info) informent du nombre de chunks, des lots vectorisés et de la taille finale de l’index.


### 4. `scripts/search_index.py`

> **Note :** Ce script peut être utilisé en ligne de commande pour interroger l’index Faiss sans passer par Streamlit.

#### But :  
Charger l’index Faiss, charger les métadonnées, générer l’embedding d’une requête utilisateur et afficher les k meilleurs résultats.

### 5. `puls_events_chatbot.py`

#### But : 
Interface Streamlit qui orchestre la logique RAG (Retrieval‐Augmented Generation) en temps réel.

#### Fichiers utilisés :

- `data/faiss_index.idx`  
- `data/faiss_metadata.json`  
  (faits lors de l’étape de vectorisation)

- `.env`  
  (doit contenir `MISTRAL_API_KEY`)

#### Étapes :

- **Session State initiale**   
    ```
    Bonjour et bienvenue sur Puls-Events ! Je suis votre assistant virtuel dédié à la découverte
    et à la recommandation d’événements culturels. Vous pouvez me demander des suggestions 
    d’activités par lieu, date ou type d’événement.
    ```
- L’app génère embedding, récupère les Top-K chunks (K=5),
- Logs console détaillés sur les chunks retournés (UID, chunk_id, score, titre, dates, lieu, extrait),
- Construit un prompt système RAG complet (instructions Puls-EventsBot + extraits),
- Appelle client.chat.complete (Mistral) avec temperature=0.0, top_p=1.0, max_tokens=1000,
- Affiche la réponse chat (liste numérotée, titres, dates, lieux, descriptif).

### 4. `run_pipeline.py`
#### But :
Enchaîner automatiquement toutes les étapes (extraction, nettoyage, vectorisation, tests).

#### Étapes :
1. Vérifie que le script est lancé depuis la racine du projet (présence du dossier `scripts/`).
2. Lance successivement :
    - `scripts/fetch_openagenda.py::main()`
    - `scripts/clean_data.py::main()`
    - `scripts/vectorize.py::main()`
3. Exécute les tests unitaires
4. Si une étape échoue, interrompt la pipeline avec un message d’erreur explicite.

---

## 6. Exécution du pipeline complet

Il est recommandé d’utiliser Poetry pour isoler et reproduire l’environnement.
Depuis la racine du projet :
```bash
poetry install
poetry shell
run python run_pipeline.py
```
Puis pour lancer l'assistant virtuel : 
```bash
streamlit run puls_events_chatbot.py
```

---

## 7. Tests unitaires
Les tests se trouvent dans `tests/test_rag_pipeline.py`. Ils valident :

1. `test_events_within_one_year()`
    Vérifie que chaque `firstdate_begin` dans `events_clean.json` est à ± 1 an de la date du jour.

2. `test_events_in_idf_region()`
    Vérifie que chaque ligne de `events_clean.json` a `location_region == "Île-de-France"`.

3. `test_chunks_count_matches_metadata()`
    Appelle `chunk_events()` sur les événements nettoyés, compare :
    `len(chunks) == len(metadata) == len(saved_metadata_json)`

4. `test_faiss_index_vector_count()`
    Vérifie que `faiss_index.idx.ntotal == len(chunks)`.

5. `test_faiss_search_returns_correct_metadata()`
    Sélectionne un chunk aléatoire, génère son embedding via `embed_batch()`, interroge l’index Faiss et confirme que :
    - L’ID retourné ≡ index du chunk choisi
    - Sa metadata contient bien text, firstdate_begin, location_city

Pour lancer l’ensemble des tests :
```bash
poetry run pytest
```

---

## 8. Fichier `.env`
Le projet utilise l’API Mistral pour la génération d’embeddings et de réponses.
Créez un fichier `.env` à la racine du projet contenant :
```env
MISTRAL_API_KEY=VotreCléAPI_Mistral
```

---

## 9. Paramètres techniques
### Modèles Mistral 
- **EMBEDDING_MODEL :** `"mistral-embed"`
- **MODEL_NAME :** `"mistral-large-latest"` 

### Faiss
- Index FlatIP pour similarité cosinus
- Normalisation L2 appliquée à chaque vecteur

### LangChain Text Splitter
- `chunk_size` = 1000
- `chunk_overlap` = 200

### Paramètres du chat Mistral
- `temperature = 0.0`
- `top_p = 1.0`
- `max_tokens = 1000`

---

Projet développé dans le cadre du parcours **Data Engineer** par OpenClassrooms.