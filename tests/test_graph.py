"""
Valida la construcción del Knowledge Graph con el dataset sintético real
(no el juguete de 5 nodos). Confirma en particular que la separación entre
el cluster matemático y el cluster médico -diseñada a mano en Fase 2- se
refleja correctamente en el grafo, y que la conexión oculta (R1 <-> P4)
no tiene camino explícito.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from src.ingestion.loader import load_dataset
from src.graph.builder import build_graph
from src.graph.queries import get_shortest_path_length, get_neighbors
from src.scoring.relevance import compute_centrality


def _graph():
    return build_graph(load_dataset())


def test_graph_has_expected_size():
    graph = _graph()
    # 54 nodos base (Fase 4) + 28 de la jerarquía oficial del reto
    # (4 Faculty + 4 Program + 4 Subject + 4 Capability + 5 Group + 7 ResearchLine)
    assert graph.number_of_nodes() == 82
    assert graph.number_of_edges() > 0


def test_math_and_medical_clusters_are_disconnected():
    """El cluster matemático (R1, R2, R8, R11 y su entorno) debe quedar en
    una componente conexa distinta al cluster médico -- así lo diseñamos
    en Fase 2 para que la conexión R1<->P4 sea realmente 'oculta'."""
    graph = _graph()
    assert not nx.is_connected(graph)
    assert nx.node_connected_component(graph, "R1") != nx.node_connected_component(graph, "P4")


def test_hidden_connection_pair_has_no_path():
    graph = _graph()
    dist = get_shortest_path_length(graph, "R1", "P4")
    assert dist is None


def test_centrality_discriminates_more_than_pagerank():
    """Confirma a escala real (54 nodos) lo observado en el grafo de juguete:
    degree centrality discrimina más que PageRank crudo."""
    graph = _graph()
    pr = nx.pagerank(graph)
    pr_spread = max(pr.values()) - min(pr.values())

    degree_centrality = compute_centrality(graph)
    degree_spread = max(degree_centrality.values()) - min(degree_centrality.values())

    assert degree_spread > pr_spread


if __name__ == "__main__":
    test_graph_has_expected_size()
    test_math_and_medical_clusters_are_disconnected()
    test_hidden_connection_pair_has_no_path()
    test_centrality_discriminates_more_than_pagerank()
    print("Todos los tests del grafo pasaron.")
