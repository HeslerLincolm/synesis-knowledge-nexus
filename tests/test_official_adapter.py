"""
Valida el adaptador de Data V1.0 oficial contra cualquier data que esté
puesta en data/official/ (tu fixture de prueba original, o tus 13 CSV
reales completos). A diferencia de la primera versión de este archivo,
NO asume valores específicos (no verifica que exista 'FAC-004' ni
'PRJ-001') -- valida la FORMA de los datos: que cargan, que la jerarquía
tiene sentido, que las trazas de IDs canónicos se arman correctamente
para cualquier entidad real que exista.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.official_adapter import load_official_dataset
from src.graph.builder import build_graph
from src.graph.queries import get_trace_ids


def test_official_dataset_loads_without_raising():
    ds = load_official_dataset()
    assert len(ds.faculties) > 0, "No se cargó ninguna Facultad -- revisa data/official/01_institution/faculties.csv"
    assert len(ds.researchers) > 0, "No se cargó ningún Investigador -- revisa researchers.csv"


def test_at_least_one_project_thesis_or_publication_loaded():
    ds = load_official_dataset()
    total_documents = len(ds.projects) + len(ds.theses) + len(ds.papers)
    assert total_documents > 0, "No se cargó ningún Project/Thesis/Publication -- revisa 03_knowledge_needs/"


def test_graph_builds_without_error():
    ds = load_official_dataset()
    graph = build_graph(ds)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_every_project_trace_includes_faculty_when_declared():
    """Todo Project con faculty_id no vacío en el CSV debe verse reflejado
    en su traza (verifica que la arista directa Project->Faculty se creó)."""
    ds = load_official_dataset()
    graph = build_graph(ds)
    for project in ds.projects.values():
        if project.faculty_id:
            trace = get_trace_ids(graph, project.id)
            assert trace.get("faculty_id") == project.faculty_id, (
                f"Project {project.id} declara faculty_id={project.faculty_id} "
                f"pero la traza del grafo no lo refleja: {trace}"
            )


def test_every_publication_with_authors_has_researcher_in_trace():
    """Toda Publication con al menos un autor conocido debe traer
    researcher_id en su traza (vía publication_researcher.csv / columna researchers)."""
    ds = load_official_dataset()
    graph = build_graph(ds)
    known_researcher_ids = set(ds.researchers.keys())
    for paper in ds.papers.values():
        valid_authors = [a for a in paper.author_ids if a in known_researcher_ids]
        if valid_authors:
            trace = get_trace_ids(graph, paper.id)
            assert "researcher_id" in trace, (
                f"Publication {paper.id} tiene autores {valid_authors} pero "
                f"la traza no encontró ningún researcher_id: {trace}"
            )


def test_referential_integrity_check_runs_without_crashing():
    """La validación de integridad (avisos, no excepciones) debe correr sin
    reventar el proceso, sin importar cuántas advertencias produzca."""
    ds = load_official_dataset()  # no debe lanzar excepción
    assert ds is not None


if __name__ == "__main__":
    test_official_dataset_loads_without_raising()
    test_at_least_one_project_thesis_or_publication_loaded()
    test_graph_builds_without_error()
    test_every_project_trace_includes_faculty_when_declared()
    test_every_publication_with_authors_has_researcher_in_trace()
    test_referential_integrity_check_runs_without_crashing()
    print("Todos los tests del adaptador oficial pasaron.")
