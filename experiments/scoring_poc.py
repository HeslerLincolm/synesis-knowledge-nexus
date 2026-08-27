"""
Synesis — PoC aislado de Mathematical Intelligence + Hidden Connection Discovery.

Objetivo: validar, sin RAG/Gemini/UI, que la fórmula
    R(v|q) = 0.5*SemanticSimilarity + 0.3*GraphProximity + 0.2*Centrality
y el criterio de "hidden connection" (mismo vecindario semántico pero lejos
en el grafo explícito) se comportan como esperamos, sobre un grafo de
juguete construido a mano.

No requiere API keys ni dependencias externas más allá de numpy/networkx/sklearn.
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Datos de juguete: 5 nodos con embeddings hechos a mano (dim=4, simplificado)
#    Diseñados a propósito para que exista UNA conexión oculta clara:
#    - n1 (Investigador, área: Matemática Aplicada)
#    - n2 (Paper, área: Machine Learning en Medicina)
#    - n3 (Paper, área: Matemática Pura)
#    - n4 (Investigador, área: Medicina Computacional)
#    - n5 (Dataset, área: Imágenes médicas)
#
#    n1 y n2 son semánticamente cercanos (comparten "Matemática Aplicada" /
#    fundamentos matemáticos de ML) pero NO tienen ningún camino corto
#    explícito en el grafo -> candidata a INFERRED_CONNECTION.
# ---------------------------------------------------------------------------

NODES = {
    "n1": {"type": "Researcher", "label": "Dr. Rojas (Matemática Aplicada)"},
    "n2": {"type": "Paper", "label": "ML aplicado a diagnóstico renal"},
    "n3": {"type": "Paper", "label": "Teoría de aproximación universal"},
    "n4": {"type": "Researcher", "label": "Dra. Vega (Medicina Computacional)"},
    "n5": {"type": "Dataset", "label": "Dataset de imágenes renales"},
}

# Embeddings de juguete (normalmente vendrían de Gemini). Vectores elegidos
# a mano para que n1 y n2 tengan alta similitud coseno.
EMBEDDINGS = {
    "n1": np.array([0.90, 0.10, 0.40, 0.05]),  # Matemática Aplicada
    "n2": np.array([0.85, 0.15, 0.35, 0.10]),  # ML medicina (cerca de n1)
    "n3": np.array([0.95, 0.05, 0.05, 0.02]),  # Matemática pura (cerca de n1 tambien)
    "n4": np.array([0.10, 0.90, 0.20, 0.80]),  # Medicina computacional (lejos de n1)
    "n5": np.array([0.05, 0.85, 0.10, 0.85]),  # Dataset imágenes (lejos de n1)
}

# Grafo explícito: n1 conectado solo a n3 (su mundo de matemática pura).
# n2 conectado solo a n4 y n5 (su mundo médico). NO hay edge directo ni
# camino corto entre n1 y n2 -> esa es la conexión que el sistema debe
# "descubrir" vía similitud semántica, no vía el grafo.
EDGES = [
    ("n1", "n3", "COLLABORATES_WITH"),
    ("n2", "n4", "AUTHORED"),
    ("n2", "n5", "USES"),
    ("n4", "n5", "WORKS_ON"),
]

G = nx.Graph()
for node_id, attrs in NODES.items():
    G.add_node(node_id, **attrs)
for src, dst, rel in EDGES:
    G.add_edge(src, dst, relation=rel)


# ---------------------------------------------------------------------------
# 2. Mathematical Intelligence: R(v|q)
# ---------------------------------------------------------------------------

def semantic_similarity(query_emb: np.ndarray, node_id: str) -> float:
    v = EMBEDDINGS[node_id].reshape(1, -1)
    q = query_emb.reshape(1, -1)
    return float(cosine_similarity(q, v)[0, 0])


def graph_proximity(source_id: str, target_id: str) -> float:
    """1 / (1 + shortest_path_distance). Si no hay camino, distancia = infinito -> proximidad = 0."""
    try:
        dist = nx.shortest_path_length(G, source=source_id, target=target_id)
    except nx.NetworkXNoPath:
        return 0.0
    return 1.0 / (1.0 + dist)


def minmax_normalize(d: dict) -> dict:
    values = list(d.values())
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def centrality_scores_max_only() -> dict:
    """Versión original: PageRank dividido por el máximo. Se vuelve poco discriminante
    en grafos pequeños/poco densos porque todos los nodos quedan cerca de 1."""
    pr = nx.pagerank(G)
    max_pr = max(pr.values()) or 1.0
    return {k: v / max_pr for k, v in pr.items()}


def centrality_scores_minmax() -> dict:
    """PageRank con min-max normalization. No ayuda si el PageRank crudo ya es uniforme."""
    return minmax_normalize(nx.pagerank(G))


def centrality_scores_degree() -> dict:
    """Degree centrality: número de conexiones directas, normalizado min-max.
    Más simple e interpretable ('cuántos vecinos directos tiene')."""
    return minmax_normalize(nx.degree_centrality(G))


def centrality_scores_betweenness() -> dict:
    """Betweenness centrality: qué tan seguido un nodo está en el camino más corto
    entre otros dos nodos. Buena para detectar nodos "puente" entre comunidades."""
    return minmax_normalize(nx.betweenness_centrality(G))


CENTRALITY_MAX_ONLY = centrality_scores_max_only()
CENTRALITY_MINMAX = centrality_scores_minmax()
CENTRALITY_DEGREE = centrality_scores_degree()
CENTRALITY_BETWEENNESS = centrality_scores_betweenness()

# Decisión final tras comparar métricas: PageRank no discrimina en grafos
# pequeños/poco densos (rango 0.000 en el grafo de juguete). Degree centrality
# sí discrimina (rango 1.000) y es más simple de explicar en demo.
CENTRALITY = CENTRALITY_DEGREE


def relevance(query_emb: np.ndarray, anchor_id: str, node_id: str) -> dict:
    """R(v|q) respecto a un nodo ancla (ej. el nodo más relevante a la query textual)."""
    sem = semantic_similarity(query_emb, node_id)
    prox = graph_proximity(anchor_id, node_id)
    cent = CENTRALITY[node_id]
    score = 0.5 * sem + 0.3 * prox + 0.2 * cent
    return {
        "node": node_id,
        "label": NODES[node_id]["label"],
        "semantic_similarity": round(sem, 3),
        "graph_proximity": round(prox, 3),
        "centrality": round(cent, 3),
        "final_relevance": round(score, 3),
    }


# ---------------------------------------------------------------------------
# 3. Hidden Connection Discovery (heurística simple, explicable)
#
#    Regla: dos nodos son candidatos a conexión si
#      - semantic_similarity(a, b) >= SEM_THRESHOLD   (cercanos en embedding)
#      - AND graph_distance(a, b) >= DIST_THRESHOLD (o sin camino)  (lejos en el grafo)
#
#    Si además existe un camino explícito corto -> EXPLICIT_CONNECTION.
#    Si no hay camino corto pero sí similitud alta -> INFERRED_CONNECTION.
# ---------------------------------------------------------------------------

SEM_THRESHOLD = 0.85
DIST_THRESHOLD = 3  # saltos


def discover_hidden_connections() -> list:
    findings = []
    node_ids = list(NODES.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a, b = node_ids[i], node_ids[j]
            sem = float(cosine_similarity(
                EMBEDDINGS[a].reshape(1, -1), EMBEDDINGS[b].reshape(1, -1)
            )[0, 0])
            if sem < SEM_THRESHOLD:
                continue

            try:
                dist = nx.shortest_path_length(G, source=a, target=b)
                path = nx.shortest_path(G, source=a, target=b)
            except nx.NetworkXNoPath:
                dist = float("inf")
                path = None

            if dist == 1:
                continue  # ya es un edge directo, no es "oculta"

            label = "EXPLICIT_CONNECTION" if dist < DIST_THRESHOLD else "INFERRED_CONNECTION"
            findings.append({
                "entities": (NODES[a]["label"], NODES[b]["label"]),
                "semantic_similarity": round(sem, 3),
                "graph_distance": dist if dist != float("inf") else None,
                "path": path,
                "label": label,
                "explanation": (
                    f"{NODES[a]['label']} y {NODES[b]['label']} comparten alta similitud "
                    f"semántica ({sem:.2f}) pero "
                    + (f"están a {dist} saltos en el grafo explícito." if path else
                       "no tienen camino explícito conocido en el grafo.")
                ),
            })
    return findings


# ---------------------------------------------------------------------------
# 4. Ejecución de prueba
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PRUEBA 0: Comparación de métricas/normalizaciones de Centrality")
    print("=" * 70)
    raw_pr = nx.pagerank(G)
    print(f"{'nodo':<6}{'PR crudo':<11}{'PR max-only':<14}{'PR min-max':<13}{'degree':<10}{'betweenness':<12}")
    for node_id in NODES:
        print(
            f"{node_id:<6}"
            f"{raw_pr[node_id]:<11.4f}"
            f"{CENTRALITY_MAX_ONLY[node_id]:<14.3f}"
            f"{CENTRALITY_MINMAX[node_id]:<13.3f}"
            f"{CENTRALITY_DEGREE[node_id]:<10.3f}"
            f"{CENTRALITY_BETWEENNESS[node_id]:<12.3f}"
        )
    for name, scores in [
        ("PR max-only", CENTRALITY_MAX_ONLY),
        ("PR min-max", CENTRALITY_MINMAX),
        ("degree", CENTRALITY_DEGREE),
        ("betweenness", CENTRALITY_BETWEENNESS),
    ]:
        spread = max(scores.values()) - min(scores.values())
        print(f"Rango cubierto por {name}: {spread:.3f}")

    print()
    print("=" * 70)
    print("PRUEBA 1: Mathematical Relevance Score R(v|q)")
    print("=" * 70)
    # Simulamos una query cuyo embedding es idéntico al de n1 (el investigador
    # de Matemática Aplicada), y calculamos relevancia de todos los nodos
    # respecto a n1 como ancla en el grafo.
    query_embedding = EMBEDDINGS["n1"]
    anchor = "n1"
    for node_id in NODES:
        result = relevance(query_embedding, anchor, node_id)
        print(
            f"{result['node']} ({result['label']}): "
            f"sem={result['semantic_similarity']} "
            f"prox={result['graph_proximity']} "
            f"cent={result['centrality']} "
            f"-> R={result['final_relevance']}"
        )

    print()
    print("=" * 70)
    print("PRUEBA 2: Hidden Connection Discovery")
    print("=" * 70)
    findings = discover_hidden_connections()
    if not findings:
        print("No se detectaron conexiones candidatas con los umbrales actuales.")
    for f in findings:
        print(f"\n[{f['label']}] {f['entities'][0]}  <->  {f['entities'][1]}")
        print(f"  Similitud semántica: {f['semantic_similarity']}")
        print(f"  Distancia en grafo: {f['graph_distance']}")
        print(f"  Explicación: {f['explanation']}")

    print()
    print("=" * 70)
    print("Resumen de validación")
    print("=" * 70)
    n1_n2_found = any(
        set(f["entities"]) == {NODES["n1"]["label"], NODES["n2"]["label"]}
        for f in findings
    )
    print(f"¿Se detectó la conexión oculta diseñada (n1 <-> n2)?: {n1_n2_found}")
