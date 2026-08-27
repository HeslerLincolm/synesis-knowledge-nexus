"""
Embedder GENÉRICO de marcador de posición (placeholder), NO semánticamente
significativo. Existe únicamente para poder probar el cableado completo del
pipeline (RAG -> grafo -> clustering -> scoring -> hidden connections) sobre
Data V1.0 oficial ANTES de decidir la fuente real de embeddings (Gemini API
o sentence-transformers local -- pendiente, ver conversación).

A diferencia de src.rag.synthetic_embeddings (diseñado a mano para el
dominio matemática/nefrología del dataset sintético, con una dimensión
"bridge" deliberada), este módulo funciona para CUALQUIER texto en
cualquier dominio -- pero solo captura señal léxica superficial (bolsa de
palabras con hashing), no similitud semántica real. Dos textos sobre el
mismo tema con vocabulario distinto NO se verán similares aquí, y eso
significa que Hidden Connection Discovery no va a encontrar nada
interesante todavía con este embedder -- es esperado, no es un bug.

REEMPLAZAR este módulo es el siguiente paso obligatorio antes de confiar en
resultados de descubrimiento sobre Data V1.0 real.
"""

from __future__ import annotations
import hashlib
import re
import numpy as np

EMBEDDING_DIM = 256

_STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "para",
    "con", "por", "que", "se", "su", "sus", "del", "al", "es", "como",
    "más", "sobre", "entre", "the", "of", "and", "a", "to", "in", "for",
    "on", "with",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    return [w for w in words if w not in _STOPWORDS_ES and len(w) > 2]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Bolsa de palabras con hashing: cada palabra cae en una posición fija
    del vector (determinista, sin vocabulario que mantener). Vector
    normalizado L2 para que la similitud coseno tenga sentido básico."""
    vec = np.zeros(dim, dtype=np.float64)
    for word in _tokenize(text):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_generic_embeddings(document_texts: dict[str, str], dim: int = EMBEDDING_DIM) -> dict[str, np.ndarray]:
    return {entity_id: embed_text(text, dim) for entity_id, text in document_texts.items()}


def build_generic_query_embedding(query: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    return embed_text(query, dim)
