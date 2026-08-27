"""
Hidden Connection Detector -- versión "cross-faculty" (punto 3 del contrato).

A diferencia de src/discovery/hidden_connections.py (que compara pares de
nodos por similitud coseno pura), este módulo cruza dos señales distintas
como pide el reto explícitamente:

  1. Clústeres semánticos (K-Means, ver src/clustering/kmeans_communities.py)
  2. Topología del grafo (a qué Facultad pertenece cada nodo, vía
     src/graph/queries.get_trace_ids)

Lógica: para cada cluster K-Means, se calcula la Facultad "dominante" (la
más frecuente entre sus miembros, según su traza en el grafo). Cualquier
miembro cuya propia Facultad NO coincida con la dominante es un candidato a
salto entre facultades: semánticamente pertenece a esa comunidad, pero
estructuralmente (en el grafo oficial) es de otra Facultad.

Esto es exactamente el caso ya validado en Fase 5: P4 (Facultad de
Ingeniería, F2) cae en el cluster dominado por Facultad de Ciencias (F1).

Para cada outlier se identifica además su socio semántico más cercano
DENTRO del cluster (el miembro "nativo" de la facultad dominante con mayor
similitud coseno), que sirve como ancla para construir la cadena de valor
en src/discovery/value_chain.py.
"""

from __future__ import annotations
from collections import Counter
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from src.graph.queries import get_trace_ids, get_shortest_path_length
from src.models.schemas import EXPLICIT_CONNECTION, INFERRED_CONNECTION
from src.config import HIDDEN_CONNECTION_DISTANCE_THRESHOLD


def _dominant_faculty(member_ids: list[str], graph: nx.Graph) -> str | None:
    faculties = []
    for m in member_ids:
        trace = get_trace_ids(graph, m)
        if "faculty_id" in trace:
            faculties.append(trace["faculty_id"])
    if not faculties:
        return None
    return Counter(faculties).most_common(1)[0][0]


def detect_cross_faculty_connections(
    graph: nx.Graph,
    cluster_assignments: dict[str, int],
    node_embeddings: dict[str, np.ndarray],
    node_labels: dict[str, str],
) -> list[dict]:
    """
    Devuelve una lista de hallazgos, uno por cada nodo cuyo cluster semántico
    y facultad topológica no coinciden. Cada hallazgo incluye el socio
    semántico más afín dentro del cluster (miembro nativo de la facultad
    dominante), la etiqueta EXPLICIT/INFERRED según si existe camino corto
    en el grafo entre ambos, y los faculty_id de cada lado para que el
    llamador pueda armar la Ruta de Trazabilidad.
    """
    clusters: dict[int, list[str]] = {}
    for node_id, cluster_id in cluster_assignments.items():
        clusters.setdefault(cluster_id, []).append(node_id)

    findings = []

    for cluster_id, members in clusters.items():
        dominant_faculty = _dominant_faculty(members, graph)
        if dominant_faculty is None:
            continue

        native_members = [
            m for m in members
            if get_trace_ids(graph, m).get("faculty_id") == dominant_faculty
        ]

        for member_id in members:
            trace = get_trace_ids(graph, member_id)
            own_faculty = trace.get("faculty_id")
            if own_faculty is None or own_faculty == dominant_faculty:
                continue  # no es un salto de facultad

            best_partner, best_sim = None, -1.0
            for native_id in native_members:
                if native_id not in node_embeddings or member_id not in node_embeddings:
                    continue
                sim = float(cosine_similarity(
                    node_embeddings[member_id].reshape(1, -1),
                    node_embeddings[native_id].reshape(1, -1),
                )[0, 0])
                if sim > best_sim:
                    best_sim, best_partner = sim, native_id

            if best_partner is None:
                continue

            dist = get_shortest_path_length(graph, member_id, best_partner)
            is_explicit = dist is not None and dist < HIDDEN_CONNECTION_DISTANCE_THRESHOLD
            label = EXPLICIT_CONNECTION if is_explicit else INFERRED_CONNECTION

            partner_trace = get_trace_ids(graph, best_partner)

            findings.append({
                "outlier_id": member_id,
                "outlier_label": node_labels.get(member_id, member_id),
                "outlier_faculty_id": own_faculty,
                "cluster_id": cluster_id,
                "dominant_faculty_id": dominant_faculty,
                "partner_id": best_partner,
                "partner_label": node_labels.get(best_partner, best_partner),
                "semantic_similarity": round(best_sim, 3),
                "graph_distance": dist,
                "label": label,
                "crosses_faculty": True,
                "outlier_trace": trace,
                "partner_trace": partner_trace,
            })

    return findings


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.graph.builder import build_graph
    from src.rag.synthetic_embeddings import build_synthetic_embeddings
    from src.clustering.kmeans_communities import run_kmeans

    dataset = load_dataset()
    graph = build_graph(dataset)
    embeddings = build_synthetic_embeddings(dataset)

    labels = {}
    for p in dataset.papers.values():
        labels[p.id] = p.title
    for t in dataset.theses.values():
        labels[t.id] = t.title
    for pr in dataset.projects.values():
        labels[pr.id] = pr.name

    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}

    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)

    print(f"Saltos entre facultades detectados: {len(findings)}")
    for f in findings:
        print(f"\n[{f['label']}] {f['outlier_label']}")
        print(f"  Facultad propia: {f['outlier_faculty_id']} | Facultad dominante del cluster: {f['dominant_faculty_id']}")
        print(f"  Socio semantico mas afin: {f['partner_label']} (sim={f['semantic_similarity']})")
        print(f"  Distancia en grafo: {f['graph_distance']}")
