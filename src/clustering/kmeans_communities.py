"""
Unsupervised ML: K-Means sobre embeddings para descubrir comunidades temáticas.

Pipeline: Embeddings -> Clustering -> Communities (según lo pedido en el
prompt original). No se usan etiquetas predefinidas de área -- el objetivo
es ver si el clustering redescubre las áreas por sí solo (validación de
sanidad) y, más interesante, si algún documento cae fuera del cluster
'esperado' por su área -- eso es justamente candidato a conexión oculta,
en la misma línea de src/discovery/hidden_connections.py pero visto desde
el ángulo de comunidades en vez de pares.
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


def run_kmeans(
    embeddings: dict[str, np.ndarray],
    n_clusters: int = 5,
    random_state: int = 42,
) -> dict[str, int]:
    """Devuelve {entity_id: cluster_label} para todas las entidades en embeddings."""
    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in ids])

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    return {entity_id: int(label) for entity_id, label in zip(ids, labels)}


def summarize_clusters(
    cluster_assignments: dict[str, int],
    entity_labels: dict[str, str],
    entity_areas: dict[str, str] | None = None,
) -> dict[int, dict]:
    """Agrupa por cluster y devuelve, para cada uno, la lista de entidades
    (id + label legible) y, si se provee entity_areas, la distribución de
    áreas reales dentro del cluster -- útil para ver qué tan bien el
    clustering no supervisado redescubrió las áreas conocidas."""
    summary: dict[int, dict] = {}
    for entity_id, cluster_id in cluster_assignments.items():
        summary.setdefault(cluster_id, {"members": [], "area_distribution": {}})
        summary[cluster_id]["members"].append({
            "id": entity_id,
            "label": entity_labels.get(entity_id, entity_id),
        })
        if entity_areas:
            area = entity_areas.get(entity_id, "desconocida")
            summary[cluster_id]["area_distribution"][area] = (
                summary[cluster_id]["area_distribution"].get(area, 0) + 1
            )
    return summary


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.rag.synthetic_embeddings import build_synthetic_embeddings

    dataset = load_dataset()
    all_embeddings = build_synthetic_embeddings(dataset)

    # Clustering solo sobre "documentos" (papers, tesis, proyectos), no
    # sobre researchers/datasets, siguiendo el pipeline pedido: "embeddings
    # de documentos, investigadores o temas".
    document_ids = (
        list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    )
    document_embeddings = {i: all_embeddings[i] for i in document_ids}

    entity_labels = {}
    entity_areas = {}
    for p in dataset.papers.values():
        entity_labels[p.id] = p.title
        entity_areas[p.id] = p.area_id or "?"
    for t in dataset.theses.values():
        entity_labels[t.id] = t.title
        author = dataset.researchers.get(t.author_id)
        entity_areas[t.id] = (author.research_areas[0] if author and author.research_areas else "?")
    for pr in dataset.projects.values():
        entity_labels[pr.id] = pr.name
        entity_areas[pr.id] = pr.area_id or "?"

    assignments = run_kmeans(document_embeddings, n_clusters=5)
    summary = summarize_clusters(assignments, entity_labels, entity_areas)

    for cluster_id in sorted(summary.keys()):
        info = summary[cluster_id]
        print(f"\n=== Cluster {cluster_id} ({len(info['members'])} miembros) ===")
        print(f"Distribución de áreas reales: {info['area_distribution']}")
        for m in info["members"]:
            print(f"  [{m['id']}] {m['label']}")

    print("\n--- Dónde cayó P4 (deep learning + enfermedad renal) ---")
    print(f"P4 asignado al cluster: {assignments['P4']}")
    print(f"Compañeros de cluster de P4: {[m['id'] for m in summary[assignments['P4']]['members']]}")
