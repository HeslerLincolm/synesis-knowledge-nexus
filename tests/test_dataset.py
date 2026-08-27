"""
Valida que el dataset sintético en data/raw/ carga correctamente y tiene
integridad referencial. Si alguien edita a mano los JSON y rompe una
referencia, este test falla con un mensaje claro en vez de que el error
aparezca más tarde y de forma confusa al construir el grafo (Fase 4).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import load_dataset


def test_dataset_loads_without_errors():
    dataset = load_dataset()
    assert len(dataset.universities) == 3
    assert len(dataset.research_areas) == 6
    assert len(dataset.researchers) == 12
    assert len(dataset.papers) == 20
    assert len(dataset.theses) == 4
    assert len(dataset.projects) == 5
    assert len(dataset.datasets) == 4


def test_hidden_connection_entities_exist():
    """El investigador puente (R1) y el paper con el que debería conectar
    semánticamente (P4, deep learning + enfermedad renal) deben existir,
    y R1 NO debe estar en ningún proyecto/paper del cluster médico -- si
    lo estuviera, la conexión dejaría de ser 'oculta' y sería explícita."""
    dataset = load_dataset()
    assert "R1" in dataset.researchers
    assert "P4" in dataset.papers

    medical_project_researcher_ids = set()
    for pr in dataset.projects.values():
        if pr.id != "PR2":  # PR2 es el proyecto matemático de R1, se excluye
            medical_project_researcher_ids.update(pr.researcher_ids)
    assert "R1" not in medical_project_researcher_ids

    medical_paper_author_ids = set()
    for p in dataset.papers.values():
        if p.area_id == "A4":  # Nefrología Computacional
            medical_paper_author_ids.update(p.author_ids)
    assert "R1" not in medical_paper_author_ids


if __name__ == "__main__":
    test_dataset_loads_without_errors()
    test_hidden_connection_entities_exist()
    print("Todos los tests del dataset pasaron.")
