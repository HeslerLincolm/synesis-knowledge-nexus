"""
Adaptador de Data V1.0 (Knowledge Nexus LATAM) -- el dataset OFICIAL del reto.

Lee los archivos originales tal cual (nunca los modifica, cumpliendo el
requisito "Datos originales: Conservar sin sobrescribir") y los traduce a
los mismos objetos Pydantic que ya usa todo el pipeline (grafo, RAG,
scoring, hidden connections). El resto del sistema no distingue si el
Dataset_ vino del adaptador sintético (src.ingestion.loader) o de este
adaptador -- misma forma, misma interfaz.

Recorte deliberado (13 de los 22 archivos disponibles, decidido junto con
el usuario): se usan las relaciones normalizadas (*_researcher, *_advisor,
*_group) como fuente de verdad para las aristas del grafo; se dejan fuera
subjects/competencies/learning_outcomes/researcher_expertise (rama
curricular fina) y project_group/publication_project/researcher_project
(confirmadas redundantes con columnas ya embebidas en projects.csv /
publications.csv, o no esenciales para el flujo de descubrimiento).

Archivos leídos:
  01_institution/     faculties.csv, programs.csv, research_groups.csv,
                       research_lines.csv, institutional_capabilities.csv
  02_people_curriculum/  researchers.csv, researcher_group.csv
  03_knowledge_needs/    projects.csv, theses.csv, thesis_advisor.csv,
                       publications.csv, publication_researcher.csv,
                       institutional_needs.csv
"""

from __future__ import annotations
import csv
import os

from src.ingestion.loader import Dataset_
from src.models.schemas import (
    Faculty, Program, Group, ResearchLine, Capability,
    Researcher, Project, Thesis, Paper, Need, University,
)


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró '{path}'. Verifica que copiaste los CSV oficiales "
            f"sin renombrarlos dentro de data/official/<capa>/."
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel  # fallback: coma estándar
        reader = csv.DictReader(f, dialect=dialect)
        return [row for row in reader]


def _split_multivalue(value):
    """Varios campos oficiales empaquetan IDs múltiples separados por ';'
    (ej. publications.researchers = 'INV-101;INV-117')."""
    if not value:
        return []
    return [v.strip() for v in value.split(";") if v.strip()]


def load_official_dataset(base_dir: str = "data/official") -> Dataset_:
    """
    base_dir debe contener las 3 subcarpetas oficiales:
      01_institution/, 02_people_curriculum/, 03_knowledge_needs/
    con los CSV originales sin renombrar.
    """
    inst_dir = os.path.join(base_dir, "01_institution")
    people_dir = os.path.join(base_dir, "02_people_curriculum")
    needs_dir = os.path.join(base_dir, "03_knowledge_needs")

    ds = Dataset_()

    # No hay entidad "Institución" explícita en Data V1.0 más allá del
    # nombre ficticio (UNEXA); se modela como una University sintética
    # única para mantener compatibilidad con el resto del pipeline.
    ds.universities["UNEXA"] = University(id="UNEXA", name="UNEXA (ficticia) — Knowledge Nexus LATAM")

    # --- 01_institution ---
    for row in _read_csv(os.path.join(inst_dir, "faculties.csv")):
        ds.faculties[row["faculty_id"]] = Faculty(
            id=row["faculty_id"], name=row.get("name", row["faculty_id"]), institution_id="UNEXA",
        )

    for row in _read_csv(os.path.join(inst_dir, "programs.csv")):
        ds.programs[row["program_id"]] = Program(
            id=row["program_id"],
            name=row.get("program_name") or (row.get("description") or row["program_id"])[:120],
            faculty_id=row["faculty_id"],
        )

    for row in _read_csv(os.path.join(inst_dir, "research_groups.csv")):
        ds.groups[row["group_id"]] = Group(
            id=row["group_id"], name=row.get("group_name") or row["group_id"],
            faculty_id=row.get("faculty_id") or "",
        )

    for row in _read_csv(os.path.join(inst_dir, "research_lines.csv")):
        ds.research_lines[row["line_id"]] = ResearchLine(
            id=row["line_id"], name=row.get("line_name", row["line_id"]), group_id=row["group_id"],
        )

    for row in _read_csv(os.path.join(inst_dir, "institutional_capabilities.csv")):
        ds.capabilities[row["capability_id"]] = Capability(
            id=row["capability_id"], name=row.get("name", row["capability_id"]),
            type=row.get("type"), description=row.get("description"),
        )

    # --- 02_people_curriculum ---
    for row in _read_csv(os.path.join(people_dir, "researchers.csv")):
        ds.researchers[row["researcher_id"]] = Researcher(
            id=row["researcher_id"], name=(row.get("profile_summary") or row["researcher_id"])[:80],
            university_id="UNEXA", bio=row.get("profile_summary"),
        )

    for row in _read_csv(os.path.join(people_dir, "researcher_group.csv")):
        researcher = ds.researchers.get(row["researcher_id"])
        if researcher and not researcher.group_id:  # primer grupo encontrado = principal
            researcher.group_id = row["group_id"]
            group = ds.groups.get(row["group_id"])
            researcher.faculty_id = group.faculty_id if group else None

    # --- 03_knowledge_needs ---
    for row in _read_csv(os.path.join(needs_dir, "institutional_needs.csv")):
        ds.needs[row["need_id"]] = Need(
            id=row["need_id"], title=row.get("title", row["need_id"]),
            description=row.get("description"), originating_unit=row.get("originating_unit"),
            context=row.get("context"), expected_impact=row.get("expected_impact"),
            priority=row.get("priority"),
        )

    for row in _read_csv(os.path.join(needs_dir, "projects.csv")):
        ds.projects[row["project_id"]] = Project(
            id=row["project_id"], name=row.get("title", row["project_id"]),
            description=row.get("abstract") or row.get("problem_statement") or "",
            keywords=_split_multivalue(row.get("keywords")),
            faculty_id=row.get("faculty_id") or None,
            program_id=row.get("program_id") or None,
            group_id=row.get("group_id") or None,
        )

    for row in _read_csv(os.path.join(needs_dir, "theses.csv")):
        ds.theses[row["thesis_id"]] = Thesis(
            id=row["thesis_id"], title=row.get("title", row["thesis_id"]),
            abstract=row.get("abstract", ""),
            keywords=_split_multivalue(row.get("keywords")),
            author_id="",  # se completa abajo con thesis_advisor.csv
            university_id="UNEXA",
        )

    for row in _read_csv(os.path.join(needs_dir, "thesis_advisor.csv")):
        thesis = ds.theses.get(row["thesis_id"])
        if thesis and not thesis.author_id:  # primer asesor = principal
            thesis.author_id = row["researcher_id"]

    for row in _read_csv(os.path.join(needs_dir, "publications.csv")):
        author_ids = _split_multivalue(row.get("researchers"))
        ds.papers[row["publication_id"]] = Paper(
            id=row["publication_id"], title=row.get("title", row["publication_id"]),
            abstract=row.get("abstract", ""), author_ids=author_ids,
            keywords=_split_multivalue(row.get("keywords")),
            year=int(row["year"]) if str(row.get("year", "")).isdigit() else None,
        )

    for row in _read_csv(os.path.join(needs_dir, "publication_researcher.csv")):
        paper = ds.papers.get(row["publication_id"])
        if paper and row["researcher_id"] not in paper.author_ids:
            paper.author_ids.append(row["researcher_id"])

    # Tesis sin asesor asignado quedan con author_id vacío -- Pydantic
    # exige el campo, así que se completa con un placeholder trazable en
    # vez de fallar la carga completa por un registro incompleto.
    for thesis in ds.theses.values():
        if not thesis.author_id:
            thesis.author_id = "UNKNOWN"

    _validate_referential_integrity(ds)
    return ds


def _validate_referential_integrity(ds: Dataset_) -> None:
    """A diferencia del dataset sintético (donde el loader.py original ya
    validaba esto), la data real llega con volumen suficiente (180
    investigadores, 320 proyectos...) como para que un ID mal escrito en un
    CSV real pase desapercibido silenciosamente. Se avisa (no se detiene la
    carga) para no bloquear el trabajo por un puñado de registros sucios --
    consistente con lo que el propio documento técnico anticipa
    ("heterogeneidad y valores nulos controlados")."""
    warnings = []

    for f in ds.faculties.values():
        if f.institution_id not in ds.universities:
            warnings.append(f"Faculty {f.id}: institution_id '{f.institution_id}' no existe")
    for pg in ds.programs.values():
        if pg.faculty_id not in ds.faculties:
            warnings.append(f"Program {pg.id}: faculty_id '{pg.faculty_id}' no existe")
    for g in ds.groups.values():
        if g.faculty_id not in ds.faculties:
            warnings.append(f"Group {g.id}: faculty_id '{g.faculty_id}' no existe")
    for rl in ds.research_lines.values():
        if rl.group_id not in ds.groups:
            warnings.append(f"ResearchLine {rl.id}: group_id '{rl.group_id}' no existe")
    for r in ds.researchers.values():
        if r.group_id and r.group_id not in ds.groups:
            warnings.append(f"Researcher {r.id}: group_id '{r.group_id}' no existe")
    for pr in ds.projects.values():
        if pr.faculty_id and pr.faculty_id not in ds.faculties:
            warnings.append(f"Project {pr.id}: faculty_id '{pr.faculty_id}' no existe")
        if pr.program_id and pr.program_id not in ds.programs:
            warnings.append(f"Project {pr.id}: program_id '{pr.program_id}' no existe")
        if pr.group_id and pr.group_id not in ds.groups:
            warnings.append(f"Project {pr.id}: group_id '{pr.group_id}' no existe")
    for t in ds.theses.values():
        if t.author_id != "UNKNOWN" and t.author_id not in ds.researchers:
            warnings.append(f"Thesis {t.id}: author_id '{t.author_id}' no existe (via thesis_advisor.csv)")
    for p in ds.papers.values():
        for author_id in p.author_ids:
            if author_id not in ds.researchers:
                warnings.append(f"Paper {p.id}: author_id '{author_id}' no existe (via publication_researcher.csv)")

    if warnings:
        print(f"⚠️  Integridad referencial: {len(warnings)} advertencia(s) encontradas")
        for w in warnings[:15]:
            print(f"   - {w}")
        if len(warnings) > 15:
            print(f"   ... y {len(warnings) - 15} más")


def summarize(ds: Dataset_) -> None:
    print(f"Facultades: {len(ds.faculties)}")
    print(f"Programas: {len(ds.programs)}")
    print(f"Grupos: {len(ds.groups)}")
    print(f"Líneas de investigación: {len(ds.research_lines)}")
    print(f"Capacidades institucionales: {len(ds.capabilities)}")
    print(f"Investigadores: {len(ds.researchers)}")
    print(f"Necesidades: {len(ds.needs)}")
    print(f"Proyectos: {len(ds.projects)}")
    print(f"Tesis: {len(ds.theses)}")
    print(f"Publicaciones: {len(ds.papers)}")

    orphan_researchers = [r.id for r in ds.researchers.values() if not r.group_id]
    if orphan_researchers:
        print(f"⚠️  {len(orphan_researchers)} investigador(es) sin grupo asignado: {orphan_researchers[:5]}")

    orphan_theses = [t.id for t in ds.theses.values() if t.author_id == "UNKNOWN"]
    if orphan_theses:
        print(f"⚠️  {len(orphan_theses)} tesis sin asesor asignado: {orphan_theses[:5]}")


if __name__ == "__main__":
    dataset = load_official_dataset()
    summarize(dataset)
