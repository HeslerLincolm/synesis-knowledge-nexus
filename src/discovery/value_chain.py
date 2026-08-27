"""
Construye la cadena de valor exigida por el reto:

  NECESIDAD -> ANTECEDENTES -> PROYECTOS/TESIS -> INVESTIGADORES/GRUPOS
  -> CAPACIDADES -> CURRÍCULO -> OPORTUNIDAD

A partir de un hallazgo de src/discovery/cross_faculty.py (un salto entre
facultades detectado cruzando clustering + grafo). Todo el contenido de
"OPORTUNIDAD" que arma este módulo es una plantilla determinista (NO se
llama a Gemini aquí) -- Gemini solo redacta la versión fluida en el prompt
final (Fase 8 / src/llm/prompts.py), tomando esta cadena ya armada como
insumo. Esto respeta la restricción de no agregar llamadas extra al LLM
durante la fase de búsqueda.
"""

from __future__ import annotations
import networkx as nx

from src.graph.queries import get_neighbors, get_shortest_path, describe_path
from src.ingestion.loader import Dataset_


def _entity_label(graph: nx.Graph, node_id: str | None) -> str | None:
    if node_id is None or node_id not in graph:
        return None
    return graph.nodes[node_id].get("label", node_id)


MAX_RELATED_ITEMS_PER_SIDE = 5


def _projects_and_theses_near(graph: nx.Graph, node_id: str) -> list[dict]:
    """Proyectos/tesis conectados directamente al nodo o a su investigador.

    Sin límite, un investigador prolífico (ej. un asesor con decenas de
    tesis dirigidas a lo largo de los años) puede inundar la Ruta de
    Trazabilidad con entidades que no tienen relación real con el hallazgo
    puntual -- se trunca a MAX_RELATED_ITEMS_PER_SIDE en build_value_chain,
    esta función solo recolecta el universo completo."""
    results = []
    for neighbor in get_neighbors(graph, node_id):
        node_type = graph.nodes[neighbor].get("type")
        if node_type in ("Project", "Thesis"):
            results.append({"id": neighbor, "label": _entity_label(graph, neighbor), "type": node_type})
        elif node_type == "Researcher":
            for n2 in get_neighbors(graph, neighbor):
                t2 = graph.nodes[n2].get("type")
                if t2 in ("Project", "Thesis") and n2 != node_id:
                    results.append({"id": n2, "label": _entity_label(graph, n2), "type": t2})
    # dedupe conservando orden
    seen = set()
    unique = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    return unique


def _capability_and_curriculum(graph: nx.Graph, dataset: Dataset_, faculty_id: str, capability_id: str | None) -> dict:
    curriculum = {"faculty_id": faculty_id, "faculty_label": _entity_label(graph, faculty_id)}
    if not capability_id:
        return curriculum
    capability = dataset.capabilities.get(capability_id)
    curriculum["capability_id"] = capability_id
    curriculum["capability_label"] = capability.name if capability else capability_id
    if capability:
        subject = dataset.subjects.get(capability.subject_id)
        if subject:
            curriculum["subject_id"] = subject.id
            curriculum["subject_label"] = subject.name
            program = dataset.programs.get(subject.program_id)
            if program:
                curriculum["program_id"] = program.id
                curriculum["program_label"] = program.name
    return curriculum


def build_value_chain(
    graph: nx.Graph,
    dataset: Dataset_,
    query: str,
    finding: dict,
) -> dict:
    """
    finding: un elemento devuelto por
    src.discovery.cross_faculty.detect_cross_faculty_connections
    (tiene outlier_id/partner_id, sus trazas, label, similitud, etc.)
    """
    outlier_id = finding["outlier_id"]
    partner_id = finding["partner_id"]
    outlier_trace = finding["outlier_trace"]
    partner_trace = finding["partner_trace"]

    chain = {}

    # 1. NECESIDAD -- la consulta original del usuario
    chain["NECESIDAD"] = {"query": query}

    # 2. ANTECEDENTES -- los dos documentos que originaron el hallazgo
    chain["ANTECEDENTES"] = [
        {"id": outlier_id, "label": finding["outlier_label"], "faculty_id": finding["outlier_faculty_id"]},
        {"id": partner_id, "label": finding["partner_label"], "faculty_id": finding["dominant_faculty_id"]},
    ]

    # 3. PROYECTOS/TESIS -- conectados a cada antecedente. Se trunca a
    # MAX_RELATED_ITEMS_PER_SIDE para no inundar la Ruta de Trazabilidad con
    # entidades del investigador que no tienen relación real con este
    # hallazgo puntual (ej. un asesor con decenas de tesis a lo largo de los
    # años). El conteo total se conserva para que no se pierda información,
    # solo se deja de listar cada id individualmente.
    outlier_related_all = _projects_and_theses_near(graph, outlier_id)
    partner_related_all = _projects_and_theses_near(graph, partner_id)
    chain["PROYECTOS_TESIS"] = {
        "lado_outlier": outlier_related_all[:MAX_RELATED_ITEMS_PER_SIDE],
        "lado_outlier_total": len(outlier_related_all),
        "lado_partner": partner_related_all[:MAX_RELATED_ITEMS_PER_SIDE],
        "lado_partner_total": len(partner_related_all),
    }

    # 4. INVESTIGADORES/GRUPOS -- de cada lado, vía la traza ya calculada
    chain["INVESTIGADORES_GRUPOS"] = {
        "lado_outlier": {
            "researcher_id": outlier_trace.get("researcher_id"),
            "researcher_label": _entity_label(graph, outlier_trace.get("researcher_id")),
            "group_id": outlier_trace.get("group_id"),
            "group_label": _entity_label(graph, outlier_trace.get("group_id")),
        },
        "lado_partner": {
            "researcher_id": partner_trace.get("researcher_id"),
            "researcher_label": _entity_label(graph, partner_trace.get("researcher_id")),
            "group_id": partner_trace.get("group_id"),
            "group_label": _entity_label(graph, partner_trace.get("group_id")),
        },
    }

    # 5 y 6. CAPACIDADES + CURRÍCULO -- de cada lado
    chain["CAPACIDADES"] = {
        "lado_outlier": _capability_and_curriculum(
            graph, dataset, outlier_trace.get("faculty_id"), outlier_trace.get("capability_id")
        ),
        "lado_partner": _capability_and_curriculum(
            graph, dataset, partner_trace.get("faculty_id"), partner_trace.get("capability_id")
        ),
    }
    chain["CURRICULO"] = chain["CAPACIDADES"]  # mismo contenido, alias explícito pedido por el contrato

    # 7. OPORTUNIDAD -- texto plantilla determinista (NO generado por LLM aquí)
    cap_outlier = chain["CAPACIDADES"]["lado_outlier"].get("capability_label", "su especialidad")
    cap_partner = chain["CAPACIDADES"]["lado_partner"].get("capability_label", "su especialidad")
    group_outlier = chain["INVESTIGADORES_GRUPOS"]["lado_outlier"].get("group_label", "el grupo A")
    group_partner = chain["INVESTIGADORES_GRUPOS"]["lado_partner"].get("group_label", "el grupo B")
    faculty_outlier = _entity_label(graph, finding["outlier_faculty_id"])
    faculty_partner = _entity_label(graph, finding["dominant_faculty_id"])

    chain["OPORTUNIDAD"] = {
        "resumen_deterministico": (
            f"{group_outlier} ({faculty_outlier}) trabaja en '{finding['outlier_label']}', "
            f"semánticamente afín (similitud {finding['semantic_similarity']}) a "
            f"'{finding['partner_label']}' de {group_partner} ({faculty_partner}), "
            f"sin conexión explícita en el grafo institucional. "
            f"Posible sinergia: la competencia '{cap_partner}' de {faculty_partner} "
            f"podría fortalecer la línea de {group_outlier}, que aplica '{cap_outlier}'."
        ),
        "crosses_faculty": True,
        "connection_label": finding["label"],
    }

    return chain


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.graph.builder import build_graph
    from src.rag.synthetic_embeddings import build_synthetic_embeddings
    from src.clustering.kmeans_communities import run_kmeans
    from src.discovery.cross_faculty import detect_cross_faculty_connections
    import json

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

    p4_finding = next(f for f in findings if f["outlier_id"] == "P4")
    chain = build_value_chain(graph, dataset, "deep learning applied to kidney disease", p4_finding)
    print(json.dumps(chain, indent=2, ensure_ascii=False))
