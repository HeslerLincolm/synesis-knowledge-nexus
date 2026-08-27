"""
Esquema de entidades del dominio académico. Todo el pipeline (ingestion, RAG,
graph, clustering) trabaja sobre estos tipos, no sobre dicts sueltos.

Los embeddings se agregan en tiempo de ingestión (Fase 2/3), por eso son
Optional aquí: un objeto recién cargado del JSON crudo aún no tiene embedding.
"""

from typing import Optional
from pydantic import BaseModel


class University(BaseModel):
    id: str
    name: str


class ResearchArea(BaseModel):
    id: str
    name: str


class Researcher(BaseModel):
    id: str
    name: str
    university_id: str
    research_areas: list[str] = []  # ids de ResearchArea
    bio: Optional[str] = None
    embedding: Optional[list[float]] = None
    # Jerarquía oficial del reto (Grupo -> Línea -> Investigador)
    faculty_id: Optional[str] = None
    group_id: Optional[str] = None
    line_id: Optional[str] = None


class Faculty(BaseModel):
    """Facultad. Institución -> Facultad -> Programa -> Asignatura -> Competencia."""
    id: str
    name: str
    institution_id: str  # apunta a University.id (Institución)


class Program(BaseModel):
    """Programa académico, dentro de una Facultad."""
    id: str
    name: str
    faculty_id: str


class Subject(BaseModel):
    """Asignatura, dentro de un Programa."""
    id: str
    name: str
    program_id: str


class Capability(BaseModel):
    """Competencia curricular O Capacidad institucional (según el dataset).
    En Data V1.0 oficial, la Capacidad institucional es una entidad
    independiente SIN llave foránea declarada -- su relevancia a un Grupo o
    Necesidad se DESCUBRE semánticamente (RAG/embeddings), no se declara por
    grafo. subject_id queda opcional para el dataset sintético (donde sí la
    atamos a una Asignatura a mano)."""
    id: str
    name: str
    subject_id: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None


class Need(BaseModel):
    """Necesidad institucional -- el punto de partida real de la cadena de
    valor NECESIDAD -> ANTECEDENTES -> ... -> OPORTUNIDAD del contrato oficial."""
    id: str
    title: str
    description: Optional[str] = None
    originating_unit: Optional[str] = None
    context: Optional[str] = None
    expected_impact: Optional[str] = None
    priority: Optional[str] = None


class Group(BaseModel):
    """Grupo de investigación. Grupo -> Línea de investigación -> Investigador -> Proyecto/Publicación."""
    id: str
    name: str
    faculty_id: str
    # Competencias curriculares que el grupo aplica en su investigación
    # (edge explícita APPLIES_CAPABILITY en el grafo).
    capability_ids: list[str] = []


class ResearchLine(BaseModel):
    """Línea de investigación, dentro de un Grupo."""
    id: str
    name: str
    group_id: str


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    author_ids: list[str] = []  # ids de Researcher
    area_id: Optional[str] = None
    year: Optional[int] = None
    embedding: Optional[list[float]] = None
    keywords: list[str] = []


class Thesis(BaseModel):
    id: str
    title: str
    abstract: str
    author_id: str  # Researcher
    university_id: str
    year: Optional[int] = None
    embedding: Optional[list[float]] = None
    keywords: list[str] = []


class Project(BaseModel):
    id: str
    name: str
    description: str
    researcher_ids: list[str] = []
    area_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    keywords: list[str] = []
    # Data V1.0 oficial trae estos IDs embebidos directamente en projects.csv,
    # a diferencia del dataset sintético (donde se infieren vía el investigador).
    faculty_id: Optional[str] = None
    program_id: Optional[str] = None
    group_id: Optional[str] = None


class Dataset(BaseModel):
    id: str
    name: str
    description: str
    used_by_project_ids: list[str] = []
    embedding: Optional[list[float]] = None


# Tipos de relación permitidos en el grafo. Centralizarlos aquí evita
# strings mágicos repartidos por src/graph/builder.py y src/graph/queries.py.
RELATION_TYPES = [
    "AUTHORED",         # Researcher -> Paper | Thesis
    "WORKS_ON",         # Researcher -> Project
    "USES",             # Project -> Dataset
    "RELATED_TO",       # entidad <-> entidad, inferido por similitud semántica
    "BELONGS_TO",       # relación jerárquica genérica (contención): University->Faculty->Program->Subject->Capability, Group->ResearchLine->Researcher, Researcher->University/Area
    "COLLABORATES_WITH",  # Researcher <-> Researcher
    "APPLIES_CAPABILITY",  # Group -> Capability: la competencia curricular que el grupo aplica en su investigación
]

# Etiquetas de Hidden Connection Discovery (ver src/discovery/hidden_connections.py)
EXPLICIT_CONNECTION = "EXPLICIT_CONNECTION"
INFERRED_CONNECTION = "INFERRED_CONNECTION"
