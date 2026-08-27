"""
Carga los JSON de data/raw/ y los valida contra los esquemas de src/models/schemas.py.

Además de la validación de tipos que ya hace Pydantic, corre una validación de
integridad referencial explícita (todo author_id, university_id, area_id, etc.
debe apuntar a un id que realmente existe) -- Pydantic por sí solo no detecta
una referencia a un id inexistente, solo valida que el campo sea del tipo correcto.
Detectarlo aquí, en ingestión, evita errores confusos más adelante al construir
el grafo (Fase 4), donde un id huérfano rompería silenciosamente una arista.
"""

from __future__ import annotations
import json
import os

from src.config import DATA_RAW_DIR
from src.models.schemas import (
    University, ResearchArea, Researcher, Paper, Thesis, Project, Dataset,
    Faculty, Program, Subject, Capability, Group, ResearchLine, Need,
)


class Dataset_:  # evita choque de nombre con el schema Dataset
    """Contenedor simple con todas las entidades cargadas, indexadas por id."""

    def __init__(self):
        self.universities: dict[str, University] = {}
        self.research_areas: dict[str, ResearchArea] = {}
        self.researchers: dict[str, Researcher] = {}
        self.papers: dict[str, Paper] = {}
        self.theses: dict[str, Thesis] = {}
        self.projects: dict[str, Project] = {}
        self.datasets: dict[str, Dataset] = {}
        # Jerarquía oficial del reto
        self.faculties: dict[str, Faculty] = {}
        self.programs: dict[str, Program] = {}
        self.subjects: dict[str, Subject] = {}
        self.capabilities: dict[str, Capability] = {}
        self.groups: dict[str, Group] = {}
        self.research_lines: dict[str, ResearchLine] = {}
        self.needs: dict[str, Need] = {}

    def all_entity_ids(self) -> set[str]:
        return (
            set(self.universities) | set(self.research_areas) | set(self.researchers)
            | set(self.papers) | set(self.theses) | set(self.projects) | set(self.datasets)
            | set(self.faculties) | set(self.programs) | set(self.subjects)
            | set(self.capabilities) | set(self.groups) | set(self.research_lines)
        )


def _load_json(filename: str) -> list[dict]:
    path = os.path.join(DATA_RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(raw_dir: str | None = None) -> Dataset_:
    """Carga y valida todas las entidades. Lanza ValueError con un mensaje
    claro si encuentra una referencia rota."""
    global DATA_RAW_DIR
    if raw_dir:
        DATA_RAW_DIR = raw_dir

    ds = Dataset_()

    for raw in _load_json("universities.json"):
        u = University(**raw)
        ds.universities[u.id] = u

    for raw in _load_json("research_areas.json"):
        a = ResearchArea(**raw)
        ds.research_areas[a.id] = a

    for raw in _load_json("researchers.json"):
        r = Researcher(**raw)
        ds.researchers[r.id] = r

    for raw in _load_json("papers.json"):
        p = Paper(**raw)
        ds.papers[p.id] = p

    for raw in _load_json("theses.json"):
        t = Thesis(**raw)
        ds.theses[t.id] = t

    for raw in _load_json("projects.json"):
        pr = Project(**raw)
        ds.projects[pr.id] = pr

    for raw in _load_json("datasets.json"):
        d = Dataset(**raw)
        ds.datasets[d.id] = d

    # --- Jerarquía oficial del reto ---
    for raw in _load_json("faculties.json"):
        f = Faculty(**raw)
        ds.faculties[f.id] = f

    for raw in _load_json("programs.json"):
        pg = Program(**raw)
        ds.programs[pg.id] = pg

    for raw in _load_json("subjects.json"):
        s = Subject(**raw)
        ds.subjects[s.id] = s

    for raw in _load_json("capabilities.json"):
        c = Capability(**raw)
        ds.capabilities[c.id] = c

    for raw in _load_json("groups.json"):
        g = Group(**raw)
        ds.groups[g.id] = g

    for raw in _load_json("research_lines.json"):
        rl = ResearchLine(**raw)
        ds.research_lines[rl.id] = rl

    _validate_referential_integrity(ds)
    return ds


def _validate_referential_integrity(ds: Dataset_) -> None:
    errors: list[str] = []

    for r in ds.researchers.values():
        if r.university_id not in ds.universities:
            errors.append(f"Researcher {r.id}: university_id '{r.university_id}' no existe")
        for area_id in r.research_areas:
            if area_id not in ds.research_areas:
                errors.append(f"Researcher {r.id}: research_area '{area_id}' no existe")
        if r.faculty_id and r.faculty_id not in ds.faculties:
            errors.append(f"Researcher {r.id}: faculty_id '{r.faculty_id}' no existe")
        if r.group_id and r.group_id not in ds.groups:
            errors.append(f"Researcher {r.id}: group_id '{r.group_id}' no existe")
        if r.line_id and r.line_id not in ds.research_lines:
            errors.append(f"Researcher {r.id}: line_id '{r.line_id}' no existe")

    for p in ds.papers.values():
        for author_id in p.author_ids:
            if author_id not in ds.researchers:
                errors.append(f"Paper {p.id}: author_id '{author_id}' no existe")
        if p.area_id and p.area_id not in ds.research_areas:
            errors.append(f"Paper {p.id}: area_id '{p.area_id}' no existe")

    for t in ds.theses.values():
        if t.author_id not in ds.researchers:
            errors.append(f"Thesis {t.id}: author_id '{t.author_id}' no existe")
        if t.university_id not in ds.universities:
            errors.append(f"Thesis {t.id}: university_id '{t.university_id}' no existe")

    for pr in ds.projects.values():
        for researcher_id in pr.researcher_ids:
            if researcher_id not in ds.researchers:
                errors.append(f"Project {pr.id}: researcher_id '{researcher_id}' no existe")
        if pr.area_id and pr.area_id not in ds.research_areas:
            errors.append(f"Project {pr.id}: area_id '{pr.area_id}' no existe")

    for d in ds.datasets.values():
        for project_id in d.used_by_project_ids:
            if project_id not in ds.projects:
                errors.append(f"Dataset {d.id}: used_by_project_id '{project_id}' no existe")

    for f in ds.faculties.values():
        if f.institution_id not in ds.universities:
            errors.append(f"Faculty {f.id}: institution_id '{f.institution_id}' no existe")

    for pg in ds.programs.values():
        if pg.faculty_id not in ds.faculties:
            errors.append(f"Program {pg.id}: faculty_id '{pg.faculty_id}' no existe")

    for s in ds.subjects.values():
        if s.program_id not in ds.programs:
            errors.append(f"Subject {s.id}: program_id '{s.program_id}' no existe")

    for c in ds.capabilities.values():
        if c.subject_id not in ds.subjects:
            errors.append(f"Capability {c.id}: subject_id '{c.subject_id}' no existe")

    for g in ds.groups.values():
        if g.faculty_id not in ds.faculties:
            errors.append(f"Group {g.id}: faculty_id '{g.faculty_id}' no existe")
        for cap_id in g.capability_ids:
            if cap_id not in ds.capabilities:
                errors.append(f"Group {g.id}: capability_id '{cap_id}' no existe")

    for rl in ds.research_lines.values():
        if rl.group_id not in ds.groups:
            errors.append(f"ResearchLine {rl.id}: group_id '{rl.group_id}' no existe")

    if errors:
        raise ValueError(
            "Integridad referencial rota en data/raw/:\n" + "\n".join(f"  - {e}" for e in errors)
        )


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Universidades: {len(dataset.universities)}")
    print(f"Áreas de investigación: {len(dataset.research_areas)}")
    print(f"Investigadores: {len(dataset.researchers)}")
    print(f"Papers: {len(dataset.papers)}")
    print(f"Tesis: {len(dataset.theses)}")
    print(f"Proyectos: {len(dataset.projects)}")
    print(f"Datasets: {len(dataset.datasets)}")
    print(f"Facultades: {len(dataset.faculties)}")
    print(f"Programas: {len(dataset.programs)}")
    print(f"Asignaturas: {len(dataset.subjects)}")
    print(f"Competencias: {len(dataset.capabilities)}")
    print(f"Grupos: {len(dataset.groups)}")
    print(f"Líneas de investigación: {len(dataset.research_lines)}")
    print("Integridad referencial: OK")
