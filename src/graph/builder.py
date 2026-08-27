"""
Construye el Knowledge Graph G = (V, E) a partir del dataset cargado por
src/ingestion/loader.py.

Tipos de nodo (atributo 'type'): Researcher, Paper, Thesis, Project, Dataset,
University, ResearchArea.

Tipos de relación (atributo 'relation' en cada edge), tomados de
src/models/schemas.RELATION_TYPES:
  - AUTHORED: Researcher -> Paper | Thesis
  - WORKS_ON: Researcher -> Project
  - USES: Project -> Dataset
  - BELONGS_TO: Researcher -> University, Paper/Thesis/Project -> ResearchArea
  - COLLABORATES_WITH: Researcher <-> Researcher (coautoría de al menos un paper/tesis)

RELATED_TO (por similitud semántica) NO se agrega aquí -- ese tipo de arista
se calcula en Fase 7 (Hidden Connection Discovery) a partir de embeddings,
no es una relación explícita del dataset.
"""

from __future__ import annotations
import itertools
import networkx as nx

from src.ingestion.loader import Dataset_


def build_graph(dataset: Dataset_) -> nx.Graph:
    graph = nx.Graph()

    # --- Nodos: entidades base ---
    for u in dataset.universities.values():
        graph.add_node(u.id, type="University", label=u.name)

    for a in dataset.research_areas.values():
        graph.add_node(a.id, type="ResearchArea", label=a.name)

    for r in dataset.researchers.values():
        graph.add_node(r.id, type="Researcher", label=r.name)

    for p in dataset.papers.values():
        graph.add_node(p.id, type="Paper", label=p.title)

    for t in dataset.theses.values():
        graph.add_node(t.id, type="Thesis", label=t.title)

    for pr in dataset.projects.values():
        graph.add_node(pr.id, type="Project", label=pr.name)

    for d in dataset.datasets.values():
        graph.add_node(d.id, type="Dataset", label=d.name)

    # --- Nodos: jerarquía oficial del reto ---
    for f in dataset.faculties.values():
        graph.add_node(f.id, type="Faculty", label=f.name)

    for pg in dataset.programs.values():
        graph.add_node(pg.id, type="Program", label=pg.name)

    for s in dataset.subjects.values():
        graph.add_node(s.id, type="Subject", label=s.name)

    for c in dataset.capabilities.values():
        graph.add_node(c.id, type="Capability", label=c.name)

    for g in dataset.groups.values():
        graph.add_node(g.id, type="Group", label=g.name)

    for rl in dataset.research_lines.values():
        graph.add_node(rl.id, type="ResearchLine", label=rl.name)

    # --- Aristas: cadena curricular Institución -> Facultad -> Programa -> Asignatura -> Competencia ---
    for f in dataset.faculties.values():
        graph.add_edge(f.id, f.institution_id, relation="BELONGS_TO")
    for pg in dataset.programs.values():
        graph.add_edge(pg.id, pg.faculty_id, relation="BELONGS_TO")
    for s in dataset.subjects.values():
        graph.add_edge(s.id, s.program_id, relation="BELONGS_TO")
    for c in dataset.capabilities.values():
        if c.subject_id:
            graph.add_edge(c.id, c.subject_id, relation="BELONGS_TO")

    # --- Aristas: cadena de investigación Grupo -> Línea -> Investigador -> Proyecto/Publicación ---
    for g in dataset.groups.values():
        graph.add_edge(g.id, g.faculty_id, relation="BELONGS_TO")
        for cap_id in g.capability_ids:
            graph.add_edge(g.id, cap_id, relation="APPLIES_CAPABILITY")
    for rl in dataset.research_lines.values():
        graph.add_edge(rl.id, rl.group_id, relation="BELONGS_TO")
    for r in dataset.researchers.values():
        if r.line_id:
            graph.add_edge(r.id, r.line_id, relation="BELONGS_TO")
        # Data V1.0 oficial conecta al investigador directo con su grupo,
        # sin línea de investigación individual intermedia.
        if r.group_id and r.group_id in dataset.groups:
            graph.add_edge(r.id, r.group_id, relation="BELONGS_TO")

    # --- Aristas: BELONGS_TO (Researcher -> University/Area, legado del MVP inicial) ---
    for r in dataset.researchers.values():
        graph.add_edge(r.id, r.university_id, relation="BELONGS_TO")
        for area_id in r.research_areas:
            graph.add_edge(r.id, area_id, relation="BELONGS_TO")

    # --- Aristas: AUTHORED (Researcher -> Paper) + BELONGS_TO (Paper -> Area) ---
    for p in dataset.papers.values():
        for author_id in p.author_ids:
            graph.add_edge(author_id, p.id, relation="AUTHORED")
        if p.area_id:
            graph.add_edge(p.id, p.area_id, relation="BELONGS_TO")

    # --- Aristas: AUTHORED (Researcher -> Thesis) + BELONGS_TO (Thesis -> University) ---
    for t in dataset.theses.values():
        graph.add_edge(t.author_id, t.id, relation="AUTHORED")
        graph.add_edge(t.id, t.university_id, relation="BELONGS_TO")

    # --- Aristas: WORKS_ON (Researcher -> Project) + BELONGS_TO (Project -> Area) ---
    for pr in dataset.projects.values():
        for researcher_id in pr.researcher_ids:
            graph.add_edge(researcher_id, pr.id, relation="WORKS_ON")
        if pr.area_id:
            graph.add_edge(pr.id, pr.area_id, relation="BELONGS_TO")
        # Data V1.0 oficial: el proyecto ya declara su facultad/programa/grupo
        # directamente (no hay que inferirlos vía el investigador).
        if pr.faculty_id and pr.faculty_id in dataset.faculties:
            graph.add_edge(pr.id, pr.faculty_id, relation="BELONGS_TO")
        if pr.program_id and pr.program_id in dataset.programs:
            graph.add_edge(pr.id, pr.program_id, relation="BELONGS_TO")
        if pr.group_id and pr.group_id in dataset.groups:
            graph.add_edge(pr.id, pr.group_id, relation="BELONGS_TO")

    # --- Aristas: USES (Project -> Dataset) ---
    for d in dataset.datasets.values():
        for project_id in d.used_by_project_ids:
            graph.add_edge(project_id, d.id, relation="USES")

    # --- Aristas: COLLABORATES_WITH (Researcher <-> Researcher, coautoría) ---
    for p in dataset.papers.values():
        for r1, r2 in itertools.combinations(p.author_ids, 2):
            graph.add_edge(r1, r2, relation="COLLABORATES_WITH")

    return graph


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset

    dataset = load_dataset()
    graph = build_graph(dataset)
    print(f"Nodos: {graph.number_of_nodes()}")
    print(f"Aristas: {graph.number_of_edges()}")
    print(f"Grafo conexo: {nx.is_connected(graph)}")
    print(f"Número de componentes conexas: {nx.number_connected_components(graph)}")
