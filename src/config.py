"""
Carga de configuración desde variables de entorno.
Un solo punto de verdad: nadie más en el proyecto debe leer os.environ directamente.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

# Rutas de datos
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
CHROMA_PERSIST_DIR = "data/processed/chroma_db"

# Pesos de la fórmula de relevancia R(v|q)
WEIGHT_SEMANTIC = 0.5
WEIGHT_GRAPH_PROXIMITY = 0.3
WEIGHT_CENTRALITY = 0.2

# Umbrales de Hidden Connection Discovery (validados en experiments/scoring_poc.py)
HIDDEN_CONNECTION_SEMANTIC_THRESHOLD = 0.85
HIDDEN_CONNECTION_DISTANCE_THRESHOLD = 3  # saltos en el grafo

# Cuántos resultados top-N pasar a Gemini en la explicación final
TOP_N_FOR_EXPLANATION = 5


def validate_config() -> None:
    """Falla rápido y con un mensaje claro si falta la API key, en vez de
    fallar más tarde con un error críptico del SDK de Gemini."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY no está configurada. "
            "Copia .env.example a .env y coloca tu API key ahí."
        )
