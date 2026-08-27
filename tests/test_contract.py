"""
Valida el "contrato funcional" agregado sobre la arquitectura base:
1. Trazabilidad estricta de IDs canónicos (grafo + Chroma + scorer)
2. Jerarquía oficial (Institución->Facultad->Programa->Asignatura->Competencia,
   Grupo->Línea->Investigador->Proyecto/Publicación)
3. Hidden Connection Detector cruzando clustering + topología (cross-faculty)
4. Prompt final con los 3 bloques exactos exigidos
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import load_dataset
from src.graph.builder import build_graph
from src.graph.queries import get_trace_ids
from src.rag.synthetic_embeddings import build_synthetic_embeddings
from src.clustering.kmeans_communities import run_kmeans
from src.discovery.cross_faculty import detect_cross_faculty_connections
from src.discovery.value_chain import build_value_chain
from src.llm.prompts import build_explanation_prompt, validate_output_structure
from src.llm.gemini_client import generate_explanation
from src.scoring.relevance import relevance, compute_centrality


def _setup():
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
    return dataset, graph, embeddings, labels


def test_trace_ids_include_official_canonical_ids():
    """Punto 1: el trace de un Paper debe incluir los IDs canónicos oficiales."""
    _, graph, _, _ = _setup()
    trace = get_trace_ids(graph, "P4")
    for key in ["paper_id", "researcher_id", "group_id", "faculty_id", "institution_id", "capability_id"]:
        assert key in trace, f"falta '{key}' en la traza de P4"


def test_scorer_attaches_trace_ids():
    """Punto 1: el Mathematical Scorer debe traer consigo los IDs canónicos."""
    dataset, graph, embeddings, _ = _setup()
    centrality = compute_centrality(graph)
    result = relevance(graph, embeddings, embeddings["P1"], anchor_id="P1", node_id="P4", centrality_scores=centrality)
    assert "trace_ids" in result
    assert "faculty_id" in result["trace_ids"]
    assert "capability_id" in result["trace_ids"]


def test_hierarchy_edges_exist():
    """Punto 2: las dos cadenas jerárquicas deben existir como aristas explícitas."""
    _, graph, _, _ = _setup()
    # Institución -> Facultad -> Programa -> Asignatura -> Competencia
    assert graph.has_edge("F1", "U1")
    assert graph.has_edge("PG1", "F1")
    assert graph.has_edge("S1", "PG1")
    assert graph.has_edge("C1", "S1")
    # Grupo -> Línea -> Investigador -> Proyecto/Publicación
    assert graph.has_edge("L1", "G1")
    assert graph.has_edge("R1", "L1")
    assert graph.has_edge("R1", "PR2")  # WORKS_ON, ya existente
    assert graph.has_edge("R1", "P1")   # AUTHORED, ya existente


def test_cross_faculty_detector_finds_p4():
    """Punto 3: el detector cross-faculty debe marcar P4 como salto entre
    Facultad de Ingeniería (F2, propia) y Facultad de Ciencias (F1, cluster dominante)."""
    dataset, graph, embeddings, labels = _setup()
    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}
    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)

    p4_findings = [f for f in findings if f["outlier_id"] == "P4"]
    assert len(p4_findings) == 1
    f = p4_findings[0]
    assert f["outlier_faculty_id"] == "F2"
    assert f["dominant_faculty_id"] == "F1"
    assert f["label"] == "INFERRED_CONNECTION"
    assert f["crosses_faculty"] is True


def test_value_chain_has_all_seven_links():
    """Punto 3: la cadena NECESIDAD..OPORTUNIDAD debe estar completa."""
    dataset, graph, embeddings, labels = _setup()
    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}
    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)
    p4_finding = next(f for f in findings if f["outlier_id"] == "P4")

    chain = build_value_chain(graph, dataset, "deep learning applied to kidney disease", p4_finding)

    required_keys = [
        "NECESIDAD", "ANTECEDENTES", "PROYECTOS_TESIS", "INVESTIGADORES_GRUPOS",
        "CAPACIDADES", "CURRICULO", "OPORTUNIDAD",
    ]
    for key in required_keys:
        assert key in chain, f"falta el eslabón '{key}' en la cadena de valor"

    assert chain["OPORTUNIDAD"]["crosses_faculty"] is True
    assert chain["CAPACIDADES"]["lado_outlier"]["capability_id"] == "C2"
    assert chain["CAPACIDADES"]["lado_partner"]["capability_id"] == "C1"


def test_value_chain_truncates_projects_and_theses():
    """La Ruta de Trazabilidad no debe inundarse con TODAS las tesis/proyectos
    de un investigador prolífico -- se trunca a un máximo por lado, con el
    conteo total conservado (no se pierde información, solo no se lista todo)."""
    from src.discovery.value_chain import MAX_RELATED_ITEMS_PER_SIDE

    dataset, graph, embeddings, labels = _setup()
    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}
    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)
    p4_finding = next(f for f in findings if f["outlier_id"] == "P4")

    chain = build_value_chain(graph, dataset, "deep learning applied to kidney disease", p4_finding)

    for side in ("lado_outlier", "lado_partner"):
        assert len(chain["PROYECTOS_TESIS"][side]) <= MAX_RELATED_ITEMS_PER_SIDE
        assert f"{side}_total" in chain["PROYECTOS_TESIS"]
        # el total nunca debe ser menor a lo listado
        assert chain["PROYECTOS_TESIS"][f"{side}_total"] >= len(chain["PROYECTOS_TESIS"][side])


def test_prompt_contains_exact_three_headers():
    """Punto 4: el prompt debe instruir los 3 bloques exactos."""
    dataset, graph, embeddings, labels = _setup()
    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}
    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)
    p4_finding = next(f for f in findings if f["outlier_id"] == "P4")
    chain = build_value_chain(graph, dataset, "deep learning applied to kidney disease", p4_finding)

    prompt = build_explanation_prompt("deep learning applied to kidney disease", chain)
    assert "## Oportunidad Accionable" in prompt
    assert "## Evidencia y Naturaleza de la Relación" in prompt
    assert "## Ruta de Trazabilidad" in prompt
    assert "<contenido_recuperado>" in prompt  # separación untrusted data


def test_dry_run_response_matches_required_structure():
    """El cliente en dry_run (sin llamada real) también debe cumplir el formato."""
    response = generate_explanation("prompt de prueba", dry_run=True)
    validation = validate_output_structure(response)
    assert validation["all_sections_present"]
    assert validation["sections_in_order"]


if __name__ == "__main__":
    test_trace_ids_include_official_canonical_ids()
    test_scorer_attaches_trace_ids()
    test_hierarchy_edges_exist()
    test_cross_faculty_detector_finds_p4()
    test_value_chain_has_all_seven_links()
    test_value_chain_truncates_projects_and_theses()
    test_prompt_contains_exact_three_headers()
    test_dry_run_response_matches_required_structure()
    print("Todos los tests del contrato funcional pasaron.")
