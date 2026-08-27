"""
Embeddings sintéticos hechos a mano (DEMO_DATA), para validar clustering y
Hidden Connection Discovery ANTES de conectar la API real de Gemini.

Diseño: espacio de 7 dimensiones.
  [0] A1 Matemática Aplicada
  [1] A2 Teoría de Aproximación y Redes Neuronales
  [2] A3 Deep Learning en Medicina
  [3] A4 Nefrología Computacional
  [4] A5 Visión por Computadora
  [5] A6 Bioestadística
  [6] BRIDGE -- "capacidad de una red neuronal de aproximar una función
      compleja/de predicción". Alto en los papers de teoría de aproximación
      universal (mundo matemático) Y en los papers que aplican deep learning
      para *aproximar una función de diagnóstico/riesgo* (mundo médico),
      aunque esos dos mundos no comparten ninguna arista en el grafo.

Cada entidad recibe: vector base de sus área(s) + boost manual en BRIDGE si
su abstract habla explícitamente de "aproximar una función" / "capacidad
expresiva" (mundo matemático) o de deep learning prediciendo/aproximando un
resultado clínico complejo (mundo médico). El resto de dimensiones llevan un
ruido determinista pequeño (basado en un hash del id) para que no haya dos
vectores idénticos, sin introducir aleatoriedad real -> reproducible entre
corridas.

Cuando se conecte Gemini embeddings real (Fase 3 completa), este módulo se
reemplaza sin tocar clustering.py ni hidden_connections.py, que solo esperan
un dict[str, np.ndarray].
"""

from __future__ import annotations
import hashlib
import numpy as np

AREA_BASE_VECTORS = {
    "A1": np.array([1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "A2": np.array([0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "A3": np.array([0.0, 0.2, 1.0, 0.3, 0.2, 0.1, 0.0]),
    "A4": np.array([0.0, 0.1, 0.3, 1.0, 0.2, 0.3, 0.0]),
    "A5": np.array([0.0, 0.0, 0.3, 0.2, 1.0, 0.0, 0.0]),
    "A6": np.array([0.0, 0.0, 0.1, 0.3, 0.0, 1.0, 0.0]),
}

# Boost manual en la dimensión BRIDGE (índice 6), calibrado a mano para que
# la similitud coseno entre el cluster matemático y P4 supere el umbral 0.85
# de src/config.HIDDEN_CONNECTION_SEMANTIC_THRESHOLD.
BRIDGE_BOOST = {
    "P1": 2.6,    # Teoremas de aproximación universal
    "P2": 2.0,    # Cotas de aproximación
    "P14": 2.0,   # Universal approximation con ReLU
    "P19": 3.0,   # Capacidad expresiva de redes profundas (el más explícito)
    "T1": 2.6,    # Tesis de Elena Rojas, mismo tema que P1
    "P4": 2.4,    # Deep learning para diagnóstico renal -- "aproxima la función de riesgo"
    "P6": 1.0,    # Predicción de progresión -- relacionado pero menos explícito
    "T2": 0.8,    # Tesis de Sofia Vega, deep learning para ERC
}


def _deterministic_noise(entity_id: str, dims: int = 7, scale: float = 0.05) -> np.ndarray:
    """Ruido pequeño pero determinista (mismo id -> mismo ruido siempre),
    para que los vectores no queden perfectamente colineales dentro de un
    área sin depender de random.seed global."""
    h = hashlib.sha256(entity_id.encode()).digest()
    raw = np.frombuffer(h[:dims * 4], dtype=np.uint32).astype(np.float64)
    normalized = (raw / raw.max()) * 2 - 1  # rango [-1, 1]
    return normalized[:dims] * scale


def _base_vector_for_areas(area_ids: list[str]) -> np.ndarray:
    if not area_ids:
        return np.zeros(7)
    vectors = [AREA_BASE_VECTORS[a] for a in area_ids if a in AREA_BASE_VECTORS]
    if not vectors:
        return np.zeros(7)
    return np.mean(vectors, axis=0)


def build_synthetic_embeddings(dataset) -> dict[str, np.ndarray]:
    """Genera embeddings de 7-D para researchers, papers, theses, projects
    y datasets del dataset cargado (src.ingestion.loader.Dataset_)."""
    embeddings: dict[str, np.ndarray] = {}

    for p in dataset.papers.values():
        vec = _base_vector_for_areas([p.area_id] if p.area_id else [])
        vec = vec.copy()
        vec[6] = BRIDGE_BOOST.get(p.id, 0.05)
        vec = vec + _deterministic_noise(p.id)
        embeddings[p.id] = vec

    for t in dataset.theses.values():
        # Las tesis no tienen area_id explícito en el esquema; se infiere del
        # autor -- es una aproximación razonable para datos sintéticos.
        author = dataset.researchers.get(t.author_id)
        areas = author.research_areas if author else []
        vec = _base_vector_for_areas(areas).copy()
        vec[6] = BRIDGE_BOOST.get(t.id, 0.05)
        vec = vec + _deterministic_noise(t.id)
        embeddings[t.id] = vec

    for pr in dataset.projects.values():
        vec = _base_vector_for_areas([pr.area_id] if pr.area_id else []).copy()
        vec[6] = BRIDGE_BOOST.get(pr.id, 0.05)
        vec = vec + _deterministic_noise(pr.id)
        embeddings[pr.id] = vec

    for d in dataset.datasets.values():
        # Los datasets heredan el área del primer proyecto que los usa.
        area_id = None
        if d.used_by_project_ids:
            proj = dataset.projects.get(d.used_by_project_ids[0])
            area_id = proj.area_id if proj else None
        vec = _base_vector_for_areas([area_id] if area_id else []).copy()
        vec[6] = BRIDGE_BOOST.get(d.id, 0.05)
        vec = vec + _deterministic_noise(d.id)
        embeddings[d.id] = vec

    # Los researchers se embeben como el promedio de sus propios papers/tesis
    # ya calculados (si tiene), o del vector base de sus áreas si no publicó nada.
    for r in dataset.researchers.values():
        own_docs = [
            embeddings[p.id] for p in dataset.papers.values()
            if r.id in p.author_ids and p.id in embeddings
        ]
        own_docs += [
            embeddings[t.id] for t in dataset.theses.values()
            if t.author_id == r.id and t.id in embeddings
        ]
        if own_docs:
            vec = np.mean(own_docs, axis=0)
        else:
            vec = _base_vector_for_areas(r.research_areas).copy()
            vec[6] = 0.05
        embeddings[r.id] = vec + _deterministic_noise(r.id, scale=0.02)

    return embeddings


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from sklearn.metrics.pairwise import cosine_similarity

    dataset = load_dataset()
    embeddings = build_synthetic_embeddings(dataset)
    print(f"Embeddings generados: {len(embeddings)}")

    # Verificación rápida: similitud coseno del par diseñado como conexión oculta
    for bridge_id in ["P1", "P19", "T1"]:
        sim = cosine_similarity(
            embeddings[bridge_id].reshape(1, -1), embeddings["P4"].reshape(1, -1)
        )[0, 0]
        print(f"cos({bridge_id}, P4) = {sim:.3f}")
