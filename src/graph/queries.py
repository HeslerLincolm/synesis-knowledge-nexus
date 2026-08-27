"""
Utilidades de consulta sobre el Knowledge Graph. Envuelve funciones de
NetworkX con manejo de errores consistente (nodo inexistente / sin camino
devuelven None en vez de lanzar excepción) para que el orquestador (Fase 8)
no tenga que hacer try/except en cada llamada.
"""

from __future__ import annotations
import networkx as nx


def get_neighbors(graph: nx.Graph, node_id: str) -> list[str]:
    if node_id not in graph:
        return []
    return list(graph.neighbors(node_id))


def get_shortest_path(graph: nx.Graph, source_id: str, target_id: str) -> list[str] | None:
    try:
        return nx.shortest_path(graph, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def get_shortest_path_length(graph: nx.Graph, source_id: str, target_id: str) -> int | None:
    try:
        return nx.shortest_path_length(graph, source=source_id, target=target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def get_nodes_by_type(graph: nx.Graph, node_type: str) -> list[str]:
    return [n for n, data in graph.nodes(data=True) if data.get("type") == node_type]


# Mapea cada tipo de nodo al nombre del campo canónico que debe aparecer en
# la traza (punto 1 del contrato: faculty_id, program_id, group_id,
# researcher_id, capability_id, project_id, etc.)
_TYPE_TO_ID_KEY = {
    "University": "institution_id",
    "Faculty": "faculty_id",
    "Program": "program_id",
    "Subject": "subject_id",
    "Capability": "capability_id",
    "Group": "group_id",
    "ResearchLine": "line_id",
    "Researcher": "researcher_id",
    "Paper": "paper_id",
    "Thesis": "thesis_id",
    "Project": "project_id",
    "Dataset": "dataset_id",
}

# Para cada tipo, qué tipo(s) de vecino puede ser un "padre" hacia arriba en
# la jerarquía. A diferencia de una lista simple, aquí un nodo puede tener
# VARIOS padres directos a la vez (ej. en Data V1.0 oficial, un Project trae
# faculty_id + program_id + group_id embebidos, los tres como aristas
# directas) -- se capturan todos los que existan, no solo el primero.
_UPWARD_NEIGHBOR_TYPES = {
    "Paper": ["Faculty", "Program", "Group", "Researcher"],
    "Thesis": ["Faculty", "Program", "Group", "Researcher"],
    "Project": ["Faculty", "Program", "Group", "Researcher"],
    "Researcher": ["Faculty", "Group", "ResearchLine"],
    "ResearchLine": ["Group"],
    "Group": ["Faculty"],
    "Faculty": ["University"],
}

# Al elegir cuál de los padres encontrados usar como siguiente escalón para
# seguir subiendo, se prefiere el más específico primero (para no saltarse
# niveles intermedios que aporten más contexto en la traza).
_CLIMB_PRIORITY = ["Researcher", "ResearchLine", "Group", "Faculty", "University"]


def get_trace_ids(graph: nx.Graph, node_id: str) -> dict[str, str]:
    """
    Recorre el grafo hacia arriba desde node_id y devuelve un diccionario con
    los IDs canónicos de toda la cadena jerárquica que lo contiene, p.ej. para
    un Paper: {paper_id, researcher_id, group_id, faculty_id, institution_id}.

    Diseñado para funcionar tanto con el dataset sintético (cadena lineal
    Researcher->ResearchLine->Group->Faculty) como con Data V1.0 oficial
    (Project conectado directo a Faculty+Program+Group a la vez, Researcher
    conectado directo a Group sin Línea intermedia). En cada escalón solo se
    exploran los tipos de vecino explícitamente permitidos por
    _UPWARD_NEIGHBOR_TYPES -- nunca se hace una búsqueda genérica de todos
    los vecinos, para no contaminar la traza con nodos "hermanos" (ej. otro
    investigador del mismo grupo, u otro paper del mismo autor).

    Esto es lo que el Mathematical Scorer debe adjuntar a cada resultado
    (punto 1 del contrato de trazabilidad): nunca se devuelve solo texto,
    siempre acompañado de la ruta completa de IDs oficiales.
    """
    if node_id not in graph:
        return {}

    trace: dict[str, str] = {}
    node_type = graph.nodes[node_id].get("type")
    key = _TYPE_TO_ID_KEY.get(node_type)
    if key:
        trace[key] = node_id

    visited_as_current = set()
    current = node_id

    while current and current not in visited_as_current:
        visited_as_current.add(current)
        current_type = graph.nodes[current].get("type")
        allowed_types = _UPWARD_NEIGHBOR_TYPES.get(current_type, [])
        if not allowed_types:
            break

        found_this_hop = []
        for neighbor in graph.neighbors(current):
            neighbor_type = graph.nodes[neighbor].get("type")
            if neighbor_type not in allowed_types:
                continue
            neighbor_key = _TYPE_TO_ID_KEY.get(neighbor_type)
            if neighbor_key and neighbor_key not in trace:
                trace[neighbor_key] = neighbor
            found_this_hop.append((neighbor_type, neighbor))

        next_current = None
        for preferred_type in _CLIMB_PRIORITY:
            match = next((n for t, n in found_this_hop if t == preferred_type), None)
            if match:
                next_current = match
                break
        current = next_current

    # Competencia curricular: no es ancestro directo del Investigador/Grupo
    # en el grafo (vive en la rama curricular), se adjunta vía la arista
    # explícita APPLIES_CAPABILITY del Grupo, cuando exista.
    if "group_id" in trace:
        group_id = trace["group_id"]
        for neighbor in graph.neighbors(group_id):
            if graph.nodes[neighbor].get("type") == "Capability":
                trace["capability_id"] = neighbor
                break

    return trace


def describe_path(graph: nx.Graph, path: list[str]) -> str:
    """Convierte un camino de ids en una descripción legible usando los
    labels de los nodos y el tipo de relación de cada arista, p.ej.:
    'Dra. Sofia Vega -[AUTHORED]-> P4 -[BELONGS_TO]-> Nefrología Computacional'
    """
    if not path or len(path) < 2:
        label = graph.nodes[path[0]].get("label", path[0]) if path else ""
        return label

    parts = [graph.nodes[path[0]].get("label", path[0])]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        relation = graph.edges[a, b].get("relation", "?")
        parts.append(f"-[{relation}]-> {graph.nodes[b].get('label', b)}")
    return " ".join(parts)


def subgraph_around(graph: nx.Graph, node_id: str, radius: int = 1) -> nx.Graph:
    """Subgrafo de vecinos hasta 'radius' saltos de node_id. Útil para no
    pasar el grafo completo a la UI o al LLM cuando solo importa el vecindario
    de un resultado concreto."""
    if node_id not in graph:
        return nx.Graph()
    nodes_in_radius = nx.single_source_shortest_path_length(graph, node_id, cutoff=radius).keys()
    return graph.subgraph(nodes_in_radius).copy()
