"""
Embeddings semánticos reales vía sentence-transformers (local, gratis, sin
depender de la API de Gemini para indexar). Modelo multilingüe -- el
dataset oficial mezcla español predominante con inglés controlado en
algunos abstracts/títulos (ver dataset_manifest_block_C.json).

Con ~1,330 documentos reales (proyectos+tesis+publicaciones), recalcular
embeddings en cada arranque de la app sería lento e innecesario -- se
cachean a disco por entity_id + hash del texto, así que solo se
recalculan los documentos nuevos o modificados entre corridas.

IMPORTANTE (compatibilidad de espacio vectorial): el embedding de la
CONSULTA del usuario debe generarse con este MISMO modelo, nunca con
Gemini -- la similitud coseno solo es válida si ambos lados vienen del
mismo espacio vectorial. Gemini se reserva para comprensión/explicación
de la consulta, no para el embedding de búsqueda (ver orchestrator.py).
"""

from __future__ import annotations
import hashlib
import os
import pickle
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_PATH = "data/processed/sentence_transformer_cache.pkl"

_model = None  # carga perezosa: solo se instancia si realmente se usa


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache_path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def build_document_embeddings(document_texts, cache_path=DEFAULT_CACHE_PATH):
    """
    Calcula embeddings para todos los documentos, reusando la caché a disco
    para los que no cambiaron desde la última corrida (comparando un hash
    del texto, no solo el entity_id -- si el CSV fuente cambia, se recalcula
    automáticamente ese documento puntual).
    """
    cache = _load_cache(cache_path)  # {entity_id: (text_hash, vector)}
    result = {}
    to_compute = []

    for entity_id, text in document_texts.items():
        h = _text_hash(text)
        cached = cache.get(entity_id)
        if cached and cached[0] == h:
            result[entity_id] = cached[1]
        else:
            to_compute.append(entity_id)

    if to_compute:
        model = _get_model()
        texts_to_encode = [document_texts[i] for i in to_compute]
        vectors = model.encode(texts_to_encode, show_progress_bar=len(to_compute) > 50, normalize_embeddings=True)
        for entity_id, vector in zip(to_compute, vectors):
            vector = np.asarray(vector, dtype=np.float64)
            result[entity_id] = vector
            cache[entity_id] = (_text_hash(document_texts[entity_id]), vector)
        _save_cache(cache_path, cache)

    return result


def build_query_embedding(query):
    """Mismo modelo que los documentos -- obligatorio para que la similitud
    coseno tenga sentido. No se cachea (una sola consulta, barato calcularla)."""
    model = _get_model()
    vector = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(vector, dtype=np.float64)
