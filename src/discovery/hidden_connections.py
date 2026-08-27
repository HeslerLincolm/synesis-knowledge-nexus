"""
Hidden Connection Discovery.

Regla (validada en experiments/scoring_poc.py sobre un grafo de juguete de 5 nodos):
dos entidades son candidatas a conexión oculta si son semánticamente muy similares
(alta similitud coseno de embeddings) PERO están lejos -o no conectadas- en el grafo
explícito. Se etiquetan como:

  - EXPLICIT_CONNECTION: existe un camino corto (< HIDDEN_CONNECTION_DISTANCE_THRESHOLD
    saltos) en el grafo. La similitud semántica alta simplemente confirma algo que ya
    era visible en las relaciones explícitas.
  - INFERRED_CONNECTION: no hay camino corto (o no hay camino alguno). La conexión
    solo emerge del espacio semántico, no de las relaciones explícitas del grafo.
    NUNCA se presenta como un hecho comprobado -- es una hipótesis a explorar.

Nota de la validación aislada: se corrigió un bug de redondeo donde una similitud de
0.996 se mostraba como "1.00" en el texto de explicación (2 decimales insuficientes
para distinguir "muy similar" de "idéntico"). Aquí se usan 3 decimales.
"""

from __future__ import annotations
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    HIDDEN_CONNECTION_SEMANTIC_THRESHOLD,
    HIDDEN_CONNECTION_DISTANCE_THRESHOLD,
)
from src.models.schemas import EXPLICIT_CONNECTION, INFERRED_CONNECTION


def discover_hidden_connections(
    graph: nx.Graph,
    node_embeddings: dict[str, np.ndarray],
    node_labels: dict[str, str],
    candidate_ids: list[str] | None = None,
) -> list[dict]:
    """
    Busca pares de nodos candidatos con alta similitud semántica y baja
    proximidad en el grafo explícito.

    candidate_ids: si se pasa, solo se comparan pares dentro de este subconjunto
    (útil para no comparar todo-contra-todo en grafos grandes -- p.ej. limitar
    a los top-N resultados del RAG más los miembros de su cluster).
    """
    ids = candidate_ids if candidate_ids is not None else list(node_embeddings.keys())
    findings = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a not in node_embeddings or b not in node_embeddings:
                continue

            sem = float(cosine_similarity(
                np.asarray(node_embeddings[a]).reshape(1, -1),
                np.asarray(node_embeddings[b]).reshape(1, -1),
            )[0, 0])
            if sem < HIDDEN_CONNECTION_SEMANTIC_THRESHOLD:
                continue

            try:
                dist = nx.shortest_path_length(graph, source=a, target=b)
                path = nx.shortest_path(graph, source=a, target=b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = None
                path = None

            if dist == 1:
                continue  # edge directo: no es una conexión "oculta"

            is_explicit = dist is not None and dist < HIDDEN_CONNECTION_DISTANCE_THRESHOLD
            label = EXPLICIT_CONNECTION if is_explicit else INFERRED_CONNECTION

            label_a, label_b = node_labels.get(a, a), node_labels.get(b, b)
            if path:
                distance_txt = f"están a {dist} saltos en el grafo explícito"
            else:
                distance_txt = "no tienen camino explícito conocido en el grafo"

            findings.append({
                "entity_a": a,
                "entity_b": b,
                "entity_a_label": label_a,
                "entity_b_label": label_b,
                "semantic_similarity": round(sem, 3),
                "graph_distance": dist,
                "path": path,
                "label": label,
                "explanation": (
                    f"{label_a} y {label_b} comparten alta similitud semántica "
                    f"({sem:.3f}) pero {distance_txt}."
                ),
            })

    return findings
