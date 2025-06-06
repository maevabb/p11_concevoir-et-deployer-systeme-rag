# run_pipeline.py
import sys
import logging
from pathlib import Path

from scripts.fetch_openagenda import main as fetch_main
from scripts.clean_data      import main as clean_main
from scripts.vectorize       import main as vectorize_main

# === Configuration du logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

if __name__ == "__main__":
    base = Path(__file__).parent
    if not (base / "scripts").exists():
        logging.error("⚠️  Exécutez run_pipeline.py depuis la racine du projet.")
        sys.exit(1)

    try:
        logging.info("=== Étape 1 : fetch_openagenda.py ===")
        fetch_main()

        logging.info("=== Étape 2 : clean_data.py ===")
        clean_main()

        logging.info("=== Étape 3 : vectorize.py ===")
        vectorize_main()

        logging.info("=== Étape 4 : pytest tests/test_rag_pipeline.py ===")
        import pytest
        sys.exit(pytest.main(["tests/test_rag_pipeline.py"]))

    except Exception as e:
        logging.error("❌ Pipeline interrompue : %s", e)
        sys.exit(1)
