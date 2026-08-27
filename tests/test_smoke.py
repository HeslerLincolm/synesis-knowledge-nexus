"""
Test de humo: valida que src/scoring/relevance.py y src/discovery/hidden_connections.py
(los módulos reales, no el PoC) reproducen el comportamiento validado en
experiments/scoring_poc.py sobre el mismo grafo de juguete de 5 nodos.

No requiere GEMINI_API_KEY ni red -- corre 100% local y rápido.
Ejecutar: python -m pytest tests/test_smoke.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import numpy as np

from src.scoring.relevance import rank_nodes, compute_centrality
from src.discovery.hidden_connections import discover_hidden_connections

NODE_LABELS = {
    "n1": "Dr. Rojas (Matemática Aplicada)",
    "n2": "ML aplicado a diagnóstico renal",
    "n3": "Teoría de aproximación universal",
    "n4": "Dra. Vega (Medicina Computacional)",
    "n5": "Dataset de imágenes renales",
}

EMBEDDINGS = {
    "n1": np.array([0.90, 0.10, 0.40, 0.05]),
    "n2": np.array([0.85, 0.15, 0.35, 0.10]),
    "n3": np.array([0.95, 0.05, 0.05, 0.02]),
    "n4": np.array([0.10, 0.90, 0.20, 0.80]),
    "n5": np.array([0.05, 0.85, 0.10, 0.85]),
}

EDGES = [
    ("n1", "n3", "COLLABORATES_WITH"),
    ("n2", "n4", "AUTHORED"),
    ("n2", "n5", "USES"),
    ("n4", "n5", "WORKS_ON"),
]


def build_toy_graph() -> nx.Graph:
    graph = nx.Graph()
    for node_id in NODE_LABELS:
        graph.add_node(node_id)
    for src, dst, rel in EDGES:
        graph.add_edge(src, dst, relation=rel)
    return graph


def test_centrality_discriminates():
    """Degree centrality debe dar rango > 0 (a diferencia de PageRank en este grafo)."""
    graph = build_toy_graph()
    centrality = compute_centrality(graph)
    spread = max(centrality.values()) - min(centrality.values())
    assert spread > 0, "Centrality no está discriminando entre nodos"


def test_relevance_ranking_makes_sense():
    """Anclado en n1, n3 (conectado directo) debe rankear más alto que n4/n5 (mundo ajeno)."""
    graph = build_toy_graph()
    query_embedding = EMBEDDINGS["n1"]
    candidates = list(NODE_LABELS.keys())

    ranked = rank_nodes(graph, EMBEDDINGS, query_embedding, anchor_id="n1", candidate_ids=candidates)
    ranking_order = [r["node_id"] for r in ranked]

    assert ranking_order[0] == "n1"  # la query es literalmente n1
    n3_pos = ranking_order.index("n3")
    n4_pos = ranking_order.index("n4")
    n5_pos = ranking_order.index("n5")
    assert n3_pos < n4_pos
    assert n3_pos < n5_pos


def test_hidden_connection_detected():
    """La conexión oculta diseñada (n1 <-> n2) debe detectarse como INFERRED_CONNECTION."""
    graph = build_toy_graph()
    findings = discover_hidden_connections(graph, EMBEDDINGS, NODE_LABELS)

    pairs_found = {frozenset([f["entity_a"], f["entity_b"]]) for f in findings}
    assert frozenset(["n1", "n2"]) in pairs_found

    match = next(f for f in findings if frozenset([f["entity_a"], f["entity_b"]]) == frozenset(["n1", "n2"]))
    assert match["label"] == "INFERRED_CONNECTION"
    assert match["graph_distance"] is None


if __name__ == "__main__":
    test_centrality_discriminates()
    test_relevance_ranking_makes_sense()
    test_hidden_connection_detected()
    print("Todos los tests de humo pasaron.")
