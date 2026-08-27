"""
Orquestador: junta todo el pipeline detrás de una sola función, run_query().

Separación deliberada en dos fases:

  1. build_index() -- se ejecuta UNA VEZ (al arrancar la app, no por consulta):
     carga el dataset, construye el grafo, calcula embeddings, indexa en
     Chroma, corre K-Means y el detector cross-faculty sobre TODO el dataset.
     Esto es lo costoso; no tiene sentido repetirlo en cada consulta del usuario.

  2. answer_query(context, query) -- se ejecuta POR CONSULTA: solo hace
     retrieval + scoring + selección del hallazgo relevante + una llamada a
     Gemini. Rápido y barato en cuota.

Nota honesta sobre el "paralelismo de 3 búsquedas": RAG (retrieval semántico)
y Clustering/Grafo-estático (cluster de cada documento, centralidad) SÍ son
independientes entre sí y corren en paralelo con ThreadPoolExecutor. La
"búsqueda en el grafo" que depende de un ancla (ej. graph_proximity a partir
del mejor resultado del RAG) no puede ser independiente del RAG por
definición -- se calcula inmediatamente después, no en paralelo. Preferimos
ser precisos sobre esto en vez de fingir una independencia que no existe.
"""

from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx

from src.ingestion.loader import load_dataset, Dataset_
from src.ingestion.official_adapter import load_official_dataset
from src.graph.builder import build_graph
from src.rag.synthetic_embeddings import build_synthetic_embeddings, AREA_BASE_VECTORS
from src.rag.sentence_transformer_embeddings import (
    build_document_embeddings as build_real_document_embeddings,
    build_query_embedding as build_real_query_embedding,
)
from src.rag.vector_store import get_client, build_collection
from src.rag.retriever import retrieve
from src.clustering.kmeans_communities import run_kmeans
from src.scoring.relevance import rank_nodes, compute_centrality
from src.discovery.cross_faculty import detect_cross_faculty_connections
from src.discovery.value_chain import build_value_chain
from src.llm.prompts import build_explanation_prompt
from src.llm.gemini_client import generate_explanation
from src.config import TOP_N_FOR_EXPLANATION


@dataclass
class IndexContext:
    source: str  # "synthetic" | "official"
    dataset: Dataset_
    graph: nx.Graph
    embeddings: dict[str, np.ndarray]
    labels: dict[str, str]
    document_ids: list[str]
    collection: object
    cluster_assignments: dict[str, int]
    cross_faculty_findings: list[dict]
    centrality: dict[str, float]


def build_index(
    source: str = "official",
    official_base_dir: str = "data/official",
    persist_dir: str | None = None,
) -> IndexContext:
    """
    Construye todo el índice una sola vez. Costoso pero no depende de
    ninguna consulta -- se corre al arrancar la app (o al cambiar el dataset).

    source="synthetic": dataset de demo hecho a mano (matemática/nefrología),
        con embeddings sintéticos diseñados a propósito para mostrar la
        conexión oculta. Bueno para pitch/demo controlada.

    source="official": Data V1.0 real del reto, vía
        src.ingestion.official_adapter. Usa embeddings GENÉRICOS de
        marcador de posición (bolsa de palabras con hashing) mientras se
        decide la fuente real de embeddings (Gemini API o
        sentence-transformers) -- los resultados de descubrimiento sobre
        esta fuente no deben tomarse como semánticamente confiables todavía.
    """
    if source not in ("synthetic", "official"):
        raise ValueError(f"source debe ser 'synthetic' u 'official', recibido: {source}")

    if source == "synthetic":
        dataset = load_dataset()
    else:
        dataset = load_official_dataset(base_dir=official_base_dir)

    graph = build_graph(dataset)

    labels: dict[str, str] = {}
    texts: dict[str, str] = {}       # texto completo, para mostrar/citar (Chroma "documents")
    embedding_texts: dict[str, str] = {}  # texto usado SOLO para calcular el embedding
    for p in dataset.papers.values():
        labels[p.id] = p.title
        texts[p.id] = f"{p.title}\n{p.abstract}"
        embedding_texts[p.id] = f"{p.title}. {' '.join(p.keywords)}" if p.keywords else texts[p.id]
    for t in dataset.theses.values():
        labels[t.id] = t.title
        texts[t.id] = f"{t.title}\n{t.abstract}"
        embedding_texts[t.id] = f"{t.title}. {' '.join(t.keywords)}" if t.keywords else texts[t.id]
    for pr in dataset.projects.values():
        labels[pr.id] = pr.name
        texts[pr.id] = f"{pr.name}\n{pr.description}"
        embedding_texts[pr.id] = f"{pr.name}. {' '.join(pr.keywords)}" if pr.keywords else texts[pr.id]

    document_ids = list(texts.keys())

    if source == "synthetic":
        all_embeddings = build_synthetic_embeddings(dataset)
        document_embeddings = {i: all_embeddings[i] for i in document_ids}
    else:
        # Se calculan los embeddings sobre embedding_texts (título + keywords),
        # NO sobre texts (título + descripción). El dataset oficial genera sus
        # descripciones con plantillas de texto repetidas entre proyectos de
        # temas completamente distintos (confirmado empíricamente: ~9% de los
        # documentos comparten frases idénticas como "evidencia trazable y
        # validación contextual"); calcular similitud semántica sobre esas
        # descripciones infla artificialmente la similitud por redacción
        # compartida, no por contenido real. keywords es la señal más limpia
        # de contenido específico disponible en Data V1.0.
        document_embeddings = build_real_document_embeddings(embedding_texts)
        all_embeddings = document_embeddings  # investigadores no llevan embedding propio en esta fuente todavía

    if persist_dir is None:
        persist_dir = f"data/processed/chroma_db_{source}"
    client = get_client(persist_dir=persist_dir)
    collection = build_collection(client, graph, texts, document_embeddings)

    cluster_assignments = run_kmeans(document_embeddings, n_clusters=min(5, max(2, len(document_ids) // 3)))
    cross_faculty_findings = detect_cross_faculty_connections(graph, cluster_assignments, all_embeddings, labels)
    centrality = compute_centrality(graph)

    return IndexContext(
        source=source, dataset=dataset, graph=graph, embeddings=all_embeddings, labels=labels,
        document_ids=document_ids, collection=collection,
        cluster_assignments=cluster_assignments,
        cross_faculty_findings=cross_faculty_findings, centrality=centrality,
    )


_KEYWORD_TO_AREA = {
    "matemát": "A1", "optimizacion": "A1", "optimización": "A1", "algebra": "A1", "álgebra": "A1",
    "aproximaci": "A2", "approximation": "A2", "universal": "A2", "teorema": "A2",
    "deep learning": "A3", "aprendizaje profundo": "A3", "red neuronal": "A3",
    "redes neuronales": "A3", "neural network": "A3",
    "renal": "A4", "riñón": "A4", "riñon": "A4", "nefro": "A4", "kidney": "A4",
    "vision": "A5", "visión": "A5", "imagen": "A5", "imágenes": "A5", "imagenes": "A5",
    "computer vision": "A5",
    "bioestad": "A6", "biomarcador": "A6", "estadístic": "A6", "estadistic": "A6",
}


def build_query_embedding(query: str) -> np.ndarray:
    """Heurística de keywords -> vector en el mismo espacio sintético de
    src.rag.synthetic_embeddings. Al conectar Gemini embeddings real, esta
    función se reemplaza por una sola llamada a la API (1 embedding por
    consulta, dentro del presupuesto de cuota ya acordado)."""
    query_lower = query.lower()
    matched_areas = {area for kw, area in _KEYWORD_TO_AREA.items() if kw in query_lower}
    if not matched_areas:
        matched_areas = {"A3"}

    vec = np.mean([AREA_BASE_VECTORS[a] for a in matched_areas], axis=0).copy()
    if "A2" in matched_areas or len(matched_areas) >= 2:
        vec[6] = 1.5
    return vec


def _run_rag_and_static_graph_signals(context: IndexContext, query_embedding: np.ndarray) -> tuple[list[dict], dict]:
    """RAG retrieval y señales de grafo que NO dependen de la consulta
    (centralidad, ya precalculada) corren en paralelo -- son independientes."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(retrieve, context.collection, query_embedding, TOP_N_FOR_EXPLANATION)
        centrality_future = executor.submit(lambda: context.centrality)
        rag_hits = rag_future.result()
        centrality = centrality_future.result()
    return rag_hits, centrality


def answer_query(context: IndexContext, query: str, dry_run: bool = True) -> dict:
    """Pipeline completo por consulta. Una sola llamada a Gemini al final."""
    if context.source == "synthetic":
        query_embedding = build_query_embedding(query)
    else:
        query_embedding = build_real_query_embedding(query)

    rag_hits, centrality = _run_rag_and_static_graph_signals(context, query_embedding)
    if not rag_hits:
        return {"query": query, "rag_hits": [], "error": "Sin resultados de RAG para esta consulta."}

    anchor_id = rag_hits[0]["id"]

    candidate_ids = [h["id"] for h in rag_hits]
    ranking = rank_nodes(context.graph, context.embeddings, query_embedding, anchor_id, candidate_ids)

    relevant_findings = [
        f for f in context.cross_faculty_findings
        if f["outlier_id"] in candidate_ids or f["partner_id"] in candidate_ids
    ]

    value_chain = None
    explanation = None
    if relevant_findings:
        best_finding = max(relevant_findings, key=lambda f: f["semantic_similarity"])
        value_chain = build_value_chain(context.graph, context.dataset, query, best_finding)
        prompt = build_explanation_prompt(query, value_chain)
        explanation = generate_explanation(prompt, dry_run=dry_run)

    return {
        "query": query,
        "rag_hits": rag_hits,
        "ranking": ranking,
        "hidden_connections": relevant_findings,
        "value_chain": value_chain,
        "explanation": explanation,
    }


if __name__ == "__main__":
    demo_query = "identificar antecedentes y capacidades para permanencia estudiantil"

    context = build_index()
    result = answer_query(context, demo_query, dry_run=True)
    print("=== RAG hits ===")
    for h in result["rag_hits"]:
        print(f"  [{h['id']}] sim={h['semantic_similarity']}  faculty={h['trace_ids'].get('faculty_id')}")

    print("\n=== Ranking (Mathematical Scorer) ===")
    for r in result["ranking"]:
        print(f"  [{r['node_id']}] R={r['final_relevance']}  faculty={r['trace_ids'].get('faculty_id')}")

    print(f"\n=== Hidden connections relevantes: {len(result['hidden_connections'])} ===")
    for f in result["hidden_connections"]:
        print(f"  [{f['label']}] {f['outlier_label']} <-> {f['partner_label']}")

    if result["explanation"]:
        print("\n=== Explicación final (Gemini, dry_run) ===")
        print(result["explanation"])
    else:
        print("\nSin hallazgo cross-faculty relevante para esta consulta -- no se llamó a Gemini.")
