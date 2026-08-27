"""
Mathematical Intelligence: R(v|q) = 0.5*SemanticSimilarity + 0.3*GraphProximity + 0.2*Centrality

Lógica validada de forma aislada en experiments/scoring_poc.py antes de integrarse
aquí. Decisiones tomadas durante esa validación:

- Centrality usa degree centrality (min-max normalizado), NO PageRank. En grafos
  pequeños y poco densos (10-20 investigadores, como el dataset del MVP), PageRank
  converge a valores casi uniformes y no discrimina entre nodos (rango medido: 0.000
  en el grafo de prueba). Degree centrality sí discrimina (rango medido: 1.000) y
  además es más fácil de explicar en una demo ("cuántas conexiones directas tiene").
"""

from __future__ import annotations
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import WEIGHT_SEMANTIC, WEIGHT_GRAPH_PROXIMITY, WEIGHT_CENTRALITY
from src.graph.queries import get_trace_ids


def semantic_similarity(query_embedding: np.ndarray, node_embedding: np.ndarray) -> float:
    q = np.asarray(query_embedding).reshape(1, -1)
    v = np.asarray(node_embedding).reshape(1, -1)
    return float(cosine_similarity(q, v)[0, 0])


def graph_proximity(graph: nx.Graph, source_id: str, target_id: str) -> float:
    """1 / (1 + shortest_path_distance). Sin camino -> proximidad 0."""
    try:
        dist = nx.shortest_path_length(graph, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.0
    return 1.0 / (1.0 + dist)


def _minmax_normalize(d: dict) -> dict:
    if not d:
        return {}
    values = list(d.values())
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def compute_centrality(graph: nx.Graph) -> dict[str, float]:
    """Degree centrality normalizada min-max. Ver docstring del módulo."""
    return _minmax_normalize(nx.degree_centrality(graph))


def relevance(
    graph: nx.Graph,
    node_embeddings: dict[str, np.ndarray],
    query_embedding: np.ndarray,
    anchor_id: str,
    node_id: str,
    centrality_scores: dict[str, float] | None = None,
) -> dict:
    """
    Calcula R(v|q) para node_id respecto a un nodo ancla en el grafo
    (típicamente el resultado más relevante del RAG para la query textual).

    centrality_scores puede precalcularse una vez por consulta (compute_centrality)
    y pasarse aquí para no recalcular PageRank/degree por cada nodo evaluado.
    """
    if centrality_scores is None:
        centrality_scores = compute_centrality(graph)

    sem = semantic_similarity(query_embedding, node_embeddings[node_id])
    prox = graph_proximity(graph, anchor_id, node_id)
    cent = centrality_scores.get(node_id, 0.0)

    score = (
        WEIGHT_SEMANTIC * sem
        + WEIGHT_GRAPH_PROXIMITY * prox
        + WEIGHT_CENTRALITY * cent
    )

    return {
        "node_id": node_id,
        "semantic_similarity": round(sem, 3),
        "graph_proximity": round(prox, 3),
        "centrality": round(cent, 3),
        "final_relevance": round(score, 3),
        # Trazabilidad estricta (punto 1 del contrato del reto): el scorer
        # nunca devuelve solo un score, siempre acompañado de la ruta
        # completa de IDs canónicos oficiales.
        "trace_ids": get_trace_ids(graph, node_id),
    }


def rank_nodes(
    graph: nx.Graph,
    node_embeddings: dict[str, np.ndarray],
    query_embedding: np.ndarray,
    anchor_id: str,
    candidate_ids: list[str],
) -> list[dict]:
    """Calcula R(v|q) para varios nodos candidatos y los devuelve ordenados
    de mayor a menor relevancia."""
    centrality_scores = compute_centrality(graph)
    results = [
        relevance(graph, node_embeddings, query_embedding, anchor_id, node_id, centrality_scores)
        for node_id in candidate_ids
    ]
    return sorted(results, key=lambda r: r["final_relevance"], reverse=True)
