"""
Valida el diseño de embeddings sintéticos (Fase 5): confirma que la
conexión oculta diseñada (P1/P19/T1 <-> P4) cruza el umbral de similitud
semántica, y que el clustering K-Means la revela agrupando a P4 con el
cluster de teoría de aproximación en vez de con su propia área (A4).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion.loader import load_dataset
from src.rag.synthetic_embeddings import build_synthetic_embeddings
from src.clustering.kmeans_communities import run_kmeans
from src.config import HIDDEN_CONNECTION_SEMANTIC_THRESHOLD


def _embeddings():
    return build_synthetic_embeddings(load_dataset())


def test_bridge_similarity_crosses_threshold():
    emb = _embeddings()
    for bridge_id in ["P1", "P19", "T1"]:
        sim = cosine_similarity(emb[bridge_id].reshape(1, -1), emb["P4"].reshape(1, -1))[0, 0]
        assert sim >= HIDDEN_CONNECTION_SEMANTIC_THRESHOLD, (
            f"cos({bridge_id}, P4)={sim:.3f} no cruza el umbral {HIDDEN_CONNECTION_SEMANTIC_THRESHOLD}"
        )


def test_kmeans_groups_p4_with_math_cluster_not_own_area():
    """El hallazgo clave de Fase 5: K-Means agrupa a P4 con los papers de
    teoría de aproximación (P1, P19, T1), no con sus vecinos de área A4
    (P5, P6, P8, P12, P16, P20, T2)."""
    dataset = load_dataset()
    all_embeddings = _embeddings()
    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: all_embeddings[i] for i in document_ids}

    assignments = run_kmeans(document_embeddings, n_clusters=5)
    p4_cluster = assignments["P4"]

    assert assignments["P19"] == p4_cluster
    assert assignments["P1"] == p4_cluster
    # P4 NO debe compartir cluster con sus vecinos directos de área A4
    assert assignments["P5"] != p4_cluster
    assert assignments["P12"] != p4_cluster


if __name__ == "__main__":
    test_bridge_similarity_crosses_threshold()
    test_kmeans_groups_p4_with_math_cluster_not_own_area()
    print("Todos los tests de embeddings/clustering pasaron.")
